import copy
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar

from cost_het_cluster import get_estimated_cost
from parallelization.core.data_model import TaskResult
from parallelization.core.search_space import find_gpu_subsets_optimized
from parallelization.core.task_runner import Task, TaskRunner
from parallelization.core.workload import (DEFAULT_HOST_ENTRIES, Arguments,
                                           DeviceGroupInfo, JobInfo,
                                           ModelConfigurations, ModelInfo,
                                           NodeConfigurations, get_arguments, GPUType)

# Type aliases
SubsetType = Dict[int, int]
DeviceInfoType = List[Tuple[int, float]]
ResultType = List[Tuple[SubsetType, Optional[List[Any]]]]

@dataclass
class Config:
    """Configuration for cost estimation runs"""
    output_path: Path = Path("./parallelization/outputs/")
    output_filename: str = "results_hom_A100.pkl"
    max_workers: int = 12
    job_batch_sizes: Dict[str, List[int]] = None
    host_entries: Dict = None
    nodes_info: Dict = None 
    models_info: Dict = None
    jobs_info: Dict = None

    def __post_init__(self):
        if self.job_batch_sizes is None:
            self.job_batch_sizes = {
                "llama2": [128, 256, 512],
                "wresnet": [256, 512, 1024], 
                "moe": [256, 512, 1024],
            }
        if any(x is None for x in [self.host_entries, self.nodes_info, self.models_info, self.jobs_info]):
            raise ValueError("host_entries, nodes_info, models_info and jobs_info must be provided")

class ClusterManager:
    """Manages cluster configuration and subset operations"""
    
    @staticmethod
    def get_sub_host_nodes(
        subset: SubsetType,
        host_entries: Dict,
        nodes_info: Dict
    ) -> Tuple[Dict, Dict]:
        """Extract a subset of host nodes based on specified GPU allocation."""
        sub_host_entries = {}
        sub_nodes_info = {}
        
        for i, (node, count) in enumerate(subset.items()):
            if count <= 0:
                raise ValueError("GPU count must be greater than 0")
                
            sub_host_entries[i] = copy.deepcopy(host_entries[node])
            sub_host_entries[i]["num_device"] = count
            ip = host_entries[node]["ip"]
            sub_nodes_info[ip] = nodes_info[ip]
            
        return sub_host_entries, sub_nodes_info

class MemoryProfiler:
    """Handles memory profiling and requirements calculation"""
    
    @staticmethod
    def get_memory_requirements(
        job_info: JobInfo,
        profile_path: str,
        instance_type: str
    ) -> List[float]:
        """Calculate memory requirements from profile data"""
        
        def read_memory_from_profile(path: str) -> float:
            with open(path, "r") as f:
                memory = json.load(f)["execution_memory"]["total_memory_mb"]
                return int(memory) / 1024.0

        profile_path_min = os.path.join(
            profile_path, 
            f"DeviceType.{instance_type}_tp{1}_bs{job_info.min_prof_bs}.json"
        )
        profile_path_max = os.path.join(
            profile_path,
            f"DeviceType.{instance_type}_tp{1}_bs{job_info.max_prof_bs}.json"
        )

        min_memory = read_memory_from_profile(profile_path_min)
        max_memory = read_memory_from_profile(profile_path_max)
        
        return sorted([
            min_memory,
            max_memory,
            min_memory / job_info.min_prof_bs * job_info.gbs,
            max_memory / job_info.max_prof_bs * job_info.gbs,
        ])

class CostEstimator:
    """Handles cost estimation for different cluster configurations"""
    
    def __init__(self, config: Config):
        self.config = config

    def calculate_result_for_job(
        self,
        model_info: ModelInfo,
        job_info: JobInfo,
        device_info: DeviceInfoType,
    ) -> ResultType:
        """Evaluate model execution costs across different GPU cluster configurations."""
        
        # Get memory requirements
        profile_path = os.path.join(job_info.home_dir, job_info.profile_path)
        instance_type = list(self.config.nodes_info.values())[0]["instance_type"]
        memory_demand = MemoryProfiler.get_memory_requirements(
            job_info, profile_path, instance_type
        )

        # Find optimal GPU subsets
        cluster_subset = find_gpu_subsets_optimized(
            device_info, min(memory_demand), max(memory_demand)
        )
        print(f"Evaluating job {model_info.id} for {len(cluster_subset)} cluster subsets")

        # Prepare tasks
        tasks = []
        for i, subset in enumerate(cluster_subset):
            sub_host_entries, sub_nodes_info = ClusterManager.get_sub_host_nodes(
                subset, self.config.host_entries, self.config.nodes_info
            )
            device_group_info = DeviceGroupInfo(sub_host_entries, sub_nodes_info)
            args = get_arguments(model_info, device_group_info, job_info, subset)
            tasks.append(Task(i, args))

        # Run tasks and process results
        runner = TaskRunner(tasks, max_workers=self.config.max_workers)
        results: List[TaskResult] = runner.run_tasks(verbose=False)
        # Create a mapping of task results for easier lookup
        results_by_task = {
            result.task_id: (result.subset, result.cost) 
            for result in results
        }
        
        # Map each cluster subset to its corresponding result
        return [
            (subset, results_by_task.get(i, (None, None)))
            for i, subset in enumerate(cluster_subset)
        ]


def run_cost_estimation(
    config: Config
) -> None:
    """Main function to run cost estimation across all models and batch sizes"""
    
    device_info = [
        (v["num_device"], config.nodes_info[v["ip"]]["memory"])
        for k, v in config.host_entries.items()
    ]
    
    estimator = CostEstimator(config)
    results = []

    for model_info in config.models_info:
        job_info = config.jobs_info[model_info.id]
        print(f"Processing Model: {model_info.id}")
        
        for gbs in config.job_batch_sizes[model_info.model_name]:
            job_info.gbs = gbs
            print(f"Batch Size: {gbs}")
            
            try:
                tic = time.time()
                final_results = estimator.calculate_result_for_job(
                    model_info, job_info, device_info
                )
                results.append([model_info.id, gbs, final_results])
                print(f"Time: {time.time() - tic:.2f} sec")
            except Exception as e:
                print(f"Error processing {model_info.id} with batch size {gbs}: {e}")
                results.append([model_info.id, gbs, None])

    # Save results
    config.output_path.mkdir(parents=True, exist_ok=True)
    output_file = config.output_path / config.output_filename
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    jobs_info = ModelConfigurations.get_job_configs()
    models_info = ModelConfigurations.get_model_configs()
    nodes_info = NodeConfigurations.get_homogeneous_config(GPUType.A100)
    
    config = Config(job_batch_sizes=None,
                    host_entries=DEFAULT_HOST_ENTRIES,
                    nodes_info=nodes_info,
                    models_info=models_info,
                    jobs_info=jobs_info)
    # Run cost estimation
    run_cost_estimation(config)
