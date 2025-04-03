import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path 


@dataclass
class GPUAllocation:
    node_allocations: Dict[int, int]
    total_memory: int
    total_gpus: int

    def __hash__(self):
        return hash(tuple(sorted(self.node_allocations.items())))

@dataclass
class Parameters:
    total_parameters_bytes: int
    parameters_per_layer_bytes: List[int]
    activation_parameters_bytes: List[int]


@dataclass
class Model:
    model_name: str
    num_layers: int
    parameters: Parameters


@dataclass
class ExecutionTime:
    total_time_ms: float
    forward_backward_time_ms: float
    batch_generator_time_ms: float
    layernorm_grads_all_reduce_time_ms: float
    embedding_grads_all_reduce_time_ms: float
    optimizer_time_ms: float
    layer_compute_total_ms: List[float]


@dataclass
class ExecutionMemory:
    total_memory_mb: float
    layer_memory_total_mb: List[float]


@dataclass
class ModelMetrics:
    model: Model
    execution_time: ExecutionTime
    execution_memory: ExecutionMemory


@dataclass
class ModelInfo:
    model_name: str
    model_size: str
    num_layers: int
    
    @property
    def id(self) -> str:
        return f"{self.model_name}_{self.model_size}"


@dataclass
class DeviceGroupInfo:
    host_entries: Dict
    nodes_info: Dict


@dataclass
class JobInfo:
    home_dir: Path  
    profile_path: Path 
    gbs: int
    max_prof_bs: int
    min_group_scale_variance: float = 1.0
    max_permute_len: int = 4
    max_prof_tpd: int = 1
    min_prof_bs: int = 1


@dataclass
class Arguments:
    model_name: str
    model_size: str
    home_dir: str
    host_entries: Dict
    nodes_info: Dict
    profile_data_path: str
    gbs: int
    num_layers: int
    min_group_scale_variance: float
    max_permute_len: int
    max_profiled_tp_degree: int
    max_profiled_batch_size: int
    min_profiled_batch_size: int
    subset: Dict

    def __post_init__(self):
        self.profile_data_path = Path(self.home_dir) / self.profile_data_path

    @property
    def id(self) -> str:
        return f"{self.model_name}_{self.model_size}_{self.gbs}_{list(self.subset.values())}"
    
    
    
# task_runner.py

@dataclass(frozen=True)
class TaskResult:
    """Represents the result of a task execution."""
    task_id: int
    subset: str
    cost: Optional[float]
    error: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        return self.error is None
    
    

# workload.py


    
