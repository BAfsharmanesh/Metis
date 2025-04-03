import os
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path
from enum import Enum, auto

from parallelization.core.data_model import (Arguments, DeviceGroupInfo,
                                           JobInfo, ModelInfo)


# Constants
BASE_HOME_DIR = "/home/bahram/projects/Metis/"
PROFILE_ROOT = "profile_test/metis"

## Default host configuration
DEFAULT_HOST_ENTRIES = {
    0: {"ip": "IP1", "num_device": 8},
    1: {"ip": "IP2", "num_device": 8},
    2: {"ip": "IP3", "num_device": 8},
}

# Add new enum classes at the top
class GPUType(Enum):
    A6000 = "A6000"
    A100 = "A100"
    RTX4090 = "RTX4090"

class ModelName(Enum):
    MOE = "moe"
    WRESNET = "wresnet"
    LLAMA2 = "llama2"

class GPUSpecs:
    """GPU specifications for different models"""
    SPECS = {
        GPUType.A6000: {
            "full_tflops": 38.7,
            "tensor_fp8_tflops": 309.7,
            "mem_bandwidth": 768,
            "memory": 48,
        },
        GPUType.A100: {
            "full_tflops": 19.5,
            "tensor_fp8_tflops": 624,
            "mem_bandwidth": 1935,
            "memory": 80,
        },
        GPUType.RTX4090: {
            "full_tflops": 82.6,
            "tensor_fp8_tflops": 660.6,
            "mem_bandwidth": 1008,
            "memory": 24,
        },
    }

    @classmethod
    def get_memory(cls, gpu_type: GPUType) -> int:
        return cls.SPECS[gpu_type]["memory"]

class NodeConfigurations:
    """Node configurations for different setups"""
    
    @staticmethod
    def _create_node_info(gpu_type: GPUType, 
                          inter_bandwidth: float=312500000.0, 
                          intra_bandwidth: float=5312500000.0) -> Dict[str, Any]:
        return {
            "instance_type": gpu_type.value,
            "inter_bandwidth": inter_bandwidth,
            "intra_bandwidth": intra_bandwidth,
            "memory": GPUSpecs.get_memory(gpu_type)
        }

    @classmethod
    def get_heterogeneous_config(cls) -> Dict[str, Dict]:
        return {
            "IP1": cls._create_node_info(GPUType.A6000),
            "IP2": cls._create_node_info(GPUType.A100),
            "IP3": cls._create_node_info(GPUType.RTX4090),
        }

    @classmethod
    def get_homogeneous_config(cls, gpu_type: GPUType) -> Dict[str, Dict]:
        return {
            f"IP{i}": cls._create_node_info(gpu_type)
            for i in range(1, 4)
        }

class ModelConfigurations:
    """Model configurations and specifications"""
    
    @staticmethod
    def get_model_configs() -> List[ModelInfo]:
        return [
            ModelInfo(model_name=ModelName.MOE.value, model_size="380M", num_layers=9),
            ModelInfo(model_name=ModelName.MOE.value, model_size="1.3B", num_layers=17),
            ModelInfo(model_name=ModelName.MOE.value, model_size="2.4B", num_layers=17),
            ModelInfo(model_name=ModelName.MOE.value, model_size="10B", num_layers=17),
            ModelInfo(model_name=ModelName.WRESNET.value, model_size="250M", num_layers=18),
            ModelInfo(model_name=ModelName.WRESNET.value, model_size="1B", num_layers=18),
            ModelInfo(model_name=ModelName.WRESNET.value, model_size="2B", num_layers=18),
            ModelInfo(model_name=ModelName.WRESNET.value, model_size="4B", num_layers=18),
            ModelInfo(model_name=ModelName.WRESNET.value, model_size="6.8B", num_layers=18),
            ModelInfo(model_name=ModelName.WRESNET.value, model_size="13B", num_layers=35),
            ModelInfo(model_name=ModelName.LLAMA2.value, model_size="271M", num_layers=18),
            ModelInfo(model_name=ModelName.LLAMA2.value, model_size="1B", num_layers=18),
            ModelInfo(model_name=ModelName.LLAMA2.value, model_size="7B", num_layers=32),
            ModelInfo(model_name=ModelName.LLAMA2.value, model_size="13B", num_layers=40),
            ModelInfo(model_name=ModelName.LLAMA2.value, model_size="26B", num_layers=80),
        ]

    @staticmethod
    def get_job_configs() -> Dict[str, JobInfo]:
        jobs = {}
        
        # Helper function to create profile path
        def get_profile_path(model: str, size: str) -> str:
            return os.path.join(PROFILE_ROOT, model, size)

        # Define base configurations for all models
        model_configs = {
            ModelName.MOE.value: {
                "380M": {"max_prof_bs": 256},
                "1.3B": {"max_prof_bs": 256},
                "2.4B": {"max_prof_bs": 128},
                "10B": {"max_prof_bs": 64}
            },
            ModelName.WRESNET.value: {
                "250M": {"min_prof_bs": 2, "max_prof_bs": 1024},
                "1B": {"min_prof_bs": 2, "max_prof_bs": 1024},
                "2B": {"min_prof_bs": 2, "max_prof_bs": 1024},
                "4B": {"min_prof_bs": 2, "max_prof_bs": 1024},
                "6.8B": {"min_prof_bs": 2, "max_prof_bs": 1024},
                "13B": {"min_prof_bs": 2, "max_prof_bs": 512},
            },
            ModelName.LLAMA2.value: {
                "271M": {"max_prof_tpd": 8, "max_prof_bs": 16},
                "1B": {"max_prof_tpd": 8, "max_prof_bs": 16},
                "7B": {"max_prof_tpd": 8, "max_prof_bs": 8},
                "13B": {"max_prof_tpd": 8, "max_prof_bs": 4},
                "26B": {"max_prof_tpd": 8, "max_prof_bs": 4},
            }
        }

        # Create JobInfo instances for all models
        for model, sizes in model_configs.items():
            for size, config in sizes.items():
                key = f"{model}_{size}"
                jobs[key] = JobInfo(
                    home_dir=BASE_HOME_DIR,
                    profile_path=get_profile_path(model, size),
                    gbs=128,
                    **config
                )

        return jobs

def get_arguments(
    model_info: ModelInfo,
    device_group_info: DeviceGroupInfo,
    job_info: JobInfo,
    subset: Any
) -> Arguments:
    """
    Create Arguments instance with the provided configurations.
    
    Args:
        model_info: Model configuration
        device_group_info: Device group configuration
        job_info: Job configuration
        subset: Subset specification
    
    Returns:
        Arguments instance with all configurations
    """
    return Arguments(
        model_name=model_info.model_name,
        model_size=model_info.model_size,
        home_dir=job_info.home_dir,
        host_entries=device_group_info.host_entries,
        nodes_info=device_group_info.nodes_info,
        profile_data_path=job_info.profile_path,
        gbs=job_info.gbs,
        num_layers=model_info.num_layers,
        min_group_scale_variance=job_info.min_group_scale_variance,
        max_permute_len=job_info.max_permute_len,
        max_profiled_tp_degree=job_info.max_prof_tpd,
        max_profiled_batch_size=job_info.max_prof_bs,
        min_profiled_batch_size=job_info.min_prof_bs,
        subset=subset
    )
