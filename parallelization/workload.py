import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ModelInfo:
    model_name: str
    model_size: str
    num_layers: int
    def __post_init__(self):
        self.id = f"{self.model_name}_{self.model_size}"

@dataclass
class DeviceGroupInfo:
    host_entries: Dict
    nodes_info: Dict

@dataclass
class JobInfo:
    home_dir: str    
    profile_path: str
    gbs: int
    max_prof_bs: int
    min_group_scale_variance: float = 1
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
    def __post_init__(self):
        self.profile_data_path = os.path.join(self.home_dir, self.profile_data_path)
        self.id = f"{self.model_name}_{self.model_size}_{self.gbs}"
        
def get_arguments(model_info: ModelInfo, device_group_info: DeviceGroupInfo, job_info: JobInfo) -> Arguments:
    return Arguments(model_name=model_info.model_name,
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
                     min_profiled_batch_size=job_info.min_prof_bs)


# workloads
models_info = [
    ModelInfo(model_name="moe", model_size="380M", num_layers=9),
    ModelInfo(model_name="moe", model_size="1.3B", num_layers=17),
    ModelInfo(model_name="moe", model_size="2.4B", num_layers=17),
    ModelInfo(model_name="moe", model_size="10B", num_layers=17),
    ModelInfo(model_name="wresnet", model_size="250M", num_layers=18),
    ModelInfo(model_name="wresnet", model_size="1B", num_layers=18),
    ModelInfo(model_name="wresnet", model_size="2B", num_layers=18),
    ModelInfo(model_name="wresnet", model_size="4B", num_layers=18),
    ModelInfo(model_name="wresnet", model_size="6.8B", num_layers=18),
    ModelInfo(model_name="wresnet", model_size="13B", num_layers=35),
    ModelInfo(model_name="llama2", model_size="271M", num_layers=18),
    ModelInfo(model_name="llama2", model_size="1B", num_layers=18),
    ModelInfo(model_name="llama2", model_size="7B", num_layers=32),
    ModelInfo(model_name="llama2", model_size="13B", num_layers=40),
    ModelInfo(model_name="llama2", model_size="26B", num_layers=80),
]

home_dir = "/home/bahram/Projects/Metis/"
moe_path = "profile/metis/moe"
wresnet_path = "profile/metis/wresnet"
llama2_path = "profile/metis/llama2"

jobs_info = {
    "moe_380M": JobInfo(
        home_dir=home_dir, profile_path=moe_path + "/380M", gbs=128, max_prof_bs=256
    ),
    "moe_1.3B": JobInfo(
        home_dir=home_dir, profile_path=moe_path + "/1.3B", gbs=128, max_prof_bs=256
    ),
    "moe_2.4B": JobInfo(
        home_dir=home_dir, profile_path=moe_path + "/2.4B", gbs=128, max_prof_bs=128
    ),
    "moe_10B": JobInfo(
        home_dir=home_dir, profile_path=moe_path + "/10B", gbs=128, max_prof_bs=64
    ),
    "wresnet_250M": JobInfo(
        home_dir=home_dir,
        profile_path=wresnet_path + "/250M",
        gbs=128,
        max_prof_bs=1024,
        min_prof_bs=2,
    ),
    "wresnet_1B": JobInfo(
        home_dir=home_dir,
        profile_path=wresnet_path + "/1B",
        gbs=128,
        max_prof_bs=1024,
        min_prof_bs=2,
    ),
    "wresnet_2B": JobInfo(
        home_dir=home_dir,
        profile_path=wresnet_path + "/2B",
        gbs=128,
        max_prof_bs=1024,
        min_prof_bs=2,
    ),
    "wresnet_4B": JobInfo(
        home_dir=home_dir,
        profile_path=wresnet_path + "/4B",
        gbs=128,
        max_prof_bs=1024,
        min_prof_bs=2,
    ),
    "wresnet_6.8B": JobInfo(
        home_dir=home_dir,
        profile_path=wresnet_path + "/6.8B",
        gbs=128,
        max_prof_bs=1024,
        min_prof_bs=2,
    ),
    "wresnet_13B": JobInfo(
        home_dir=home_dir,
        profile_path=wresnet_path + "/13B",
        gbs=128,
        max_prof_bs=512,
        min_prof_bs=2,
    ),
    "llama2_271M": JobInfo(
        home_dir=home_dir,
        profile_path=llama2_path + "/271M",
        gbs=128,
        max_prof_tpd=8,
        max_prof_bs=16,
    ),
    "llama2_1B": JobInfo(
        home_dir=home_dir,
        profile_path=llama2_path + "/1B",
        gbs=128,
        max_prof_tpd=8,
        max_prof_bs=16,
    ),
    "llama2_7B": JobInfo(
        home_dir=home_dir,
        profile_path=llama2_path + "/7B",
        gbs=128,
        max_prof_tpd=8,
        max_prof_bs=8,
    ),
    "llama2_13B": JobInfo(
        home_dir=home_dir,
        profile_path=llama2_path + "/13B",
        gbs=128,
        max_prof_tpd=8,
        max_prof_bs=4,
    ),
    "llama2_26B": JobInfo(
        home_dir=home_dir,
        profile_path=llama2_path + "/26B",
        gbs=128,
        max_prof_tpd=8,
        max_prof_bs=4,
    ),
}

gpus_info = {
    "A6000": {
        "full_tflops": 38.7,
        "tensor_fp8_tflops": 309.7,
        "mem_bandwidth": 768,
        "memory": 48,
    },
    "A100": {
        "full_tflops": 19.5,
        "tensor_fp8_tflops": 624,
        "mem_bandwidth": 1935,
        "memory": 80,
    },
    "RTX4090": {
        "full_tflops": 82.6,
        "tensor_fp8_tflops": 660.6,
        "mem_bandwidth": 1008,
        "memory": 24,
    },
}