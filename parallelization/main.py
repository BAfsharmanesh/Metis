import copy
import json
import os
import pickle
import time

from cost_het_cluster import get_estimated_cost
from parallelization.search_space import find_gpu_subsets_optimized
from parallelization.task_runner import Task, TaskRunner
from parallelization.workload import (
    Arguments,
    DeviceGroupInfo,
    ModelInfo,
    JobInfo,
    get_arguments,
    host_entries,
    jobs_info,
    models_info,
    nodes_info, nodes_info_hom_A100, nodes_info_hom_A6000
)


def get_sub_host_nodes(subset, host_entries, nodes_info):
    sub_host_entries = {}
    sub_nodes_info = {}
    for i, (node, count) in enumerate(subset.items()):
        assert count > 0, "count should be greater than 0"
        sub_host_entries[i] = copy.deepcopy(host_entries[node])
        sub_host_entries[i]["num_device"] = count
        ip = host_entries[node]["ip"]
        sub_nodes_info[ip] = nodes_info[ip]
    return sub_host_entries, sub_nodes_info


def calculate_result_for_job(
    model_info: ModelInfo, job_info: JobInfo, device_info: list, max_workers: int
):

    profile_path = os.path.join(job_info.home_dir, job_info.profile_path)
    instance_type = list(nodes_info.values())[0]["instance_type"]
    profile_path_min = os.path.join(
        profile_path, f"DeviceType.{instance_type}_tp{1}_bs{job_info.min_prof_bs}.json"
    )
    profile_path_max = os.path.join(
        profile_path, f"DeviceType.{instance_type}_tp{1}_bs{job_info.max_prof_bs}.json"
    )

    # read json file as dictionary

    with open(profile_path_min, "r") as f:
        min_memory = json.load(f)["execution_memory"]["total_memory_mb"]
        min_memory = int(min_memory) / 1024.0
        # print(min_memory, min_memory / job_info.min_prof_bs * job_info.gbs)
    with open(profile_path_max, "r") as f:
        max_memory = json.load(f)["execution_memory"]["total_memory_mb"]
        max_memory = int(max_memory) / 1024.0
        # print(max_memory, max_memory / job_info.max_prof_bs * job_info.gbs)
    # find the optimal gpu subsets
    memory_demand = sorted(
        [
            min_memory,
            max_memory,
            min_memory / job_info.min_prof_bs * job_info.gbs,
            max_memory / job_info.max_prof_bs * job_info.gbs,
        ]
    )
    # print(memory_demand)
    cluster_subset = find_gpu_subsets_optimized(
        device_info, min(memory_demand), max(memory_demand)
    )
    print(
        f"Perform the job {model_info.id} for {len(cluster_subset)} subsets of the cluster"
    )

    tasks = []
    args_list = []
    for i, subset in enumerate(cluster_subset):
        sub_host_entries, sub_nodes_info = get_sub_host_nodes(
            subset, host_entries, nodes_info
        )
        device_group_info = DeviceGroupInfo(sub_host_entries, sub_nodes_info)

        args = get_arguments(model_info, device_group_info, job_info, subset)

        tasks.append(Task(i, args))
        args_list.append(args)

    # Create and run the TaskRunner
    runner = TaskRunner(
        tasks, max_workers=max_workers
    )  # Adjust max_workers for your CPU
    results = runner.run_tasks()
    # args = Arguments(model_name='llama2', model_size='26B', home_dir='/home/bahram/Projects/Metis/', host_entries={0: {'ip': 'IP1', 'num_device': 2}, 1: {'ip': 'IP2', 'num_device': 5}, 2: {'ip': 'IP3', 'num_device': 1}}, nodes_info={'IP1': {'instance_type': 'A6000', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 48}, 'IP2': {'instance_type': 'A100', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 80}, 'IP3': {'instance_type': 'RTX4090', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 24}}, profile_data_path='/home/bahram/Projects/Metis/profile/metis/llama2/26B', gbs=128, num_layers=80, min_group_scale_variance=1, max_permute_len=4, max_profiled_tp_degree=8, max_profiled_batch_size=4, min_profiled_batch_size=1)
    # args = args_list[-1]
    # results = get_estimated_cost(args)
    results = {item[0]: [item[1], item[2]] for item in results}
    final_results = []
    for i, subset in enumerate(cluster_subset):
        if i in results.keys():
            if results[i][1] is not None:
                # print(f"{i} subset: {subset}, Estimated Cost: {results[i][1][6]:.2f}, Strategy: {results[i][1][2]}")
                final_results.append([subset, results[i]])
            else:
                final_results.append([subset, None])
        else:
            final_results.append([subset, None])

    return final_results

def test_a_sample(models_info, jobs_info, host_entries, nodes_info):
    model_info = models_info[4]
    print(model_info.model_name)
    job_info = jobs_info[model_info.id]
    subset = {0: 4, 1: 4}
    sub_host_entries, sub_nodes_info = get_sub_host_nodes(
        subset, host_entries, nodes_info
    )
    device_group_info = DeviceGroupInfo(sub_host_entries, sub_nodes_info)
    args = get_arguments(model_info, device_group_info, job_info, subset)
    res = get_estimated_cost(args)
    print(res)

if __name__ == "__main__":

    device_info = [
        (v["num_device"], nodes_info_hom_A100[v["ip"]]["memory"])
        for k, v in host_entries.items()
    ]
    # job batch size
    job_batch_size = {
        "llama2": [128, 256, 512],
        "wresnet": [256, 512, 1024],
        "moe": [256, 512, 1024],
    }

    gather_results = []
    for model_info in models_info:
        job_info = jobs_info[model_info.id]
        print(f"Model: {model_info.id}")
        for gbs in job_batch_size[model_info.model_name]:
            job_info.gbs = gbs
            print(f"Batch Size: {gbs}")
            tic = time.time()
            try:
                final_results = calculate_result_for_job(model_info, job_info, device_info, max_workers=12)
                gather_results.append([model_info.id, gbs, final_results])
            except Exception as e:
                print(f"Error: {e}")
                gather_results.append([model_info.id, gbs, None])
            toc = time.time()
            print(f"Time: {toc-tic:.2f} sec")

    out_put_path = "./parallelization/outputs/"
    if not os.path.exists(out_put_path):
        os.makedirs(out_put_path)
    with open(out_put_path+"results_hom_A100.pkl", "wb") as f:
        pickle.dump(gather_results, f)
