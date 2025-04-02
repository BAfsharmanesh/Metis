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

from parallelization.main import calculate_result_for_job, get_sub_host_nodes

def run_all_possible_permutations():
    device_info = [
        (v["num_device"], nodes_info[v["ip"]]["memory"])
        for k, v in host_entries.items()
    ]
    # job batch size
    job_batch_size = {
        "llama2": [128, 256, 512],
        "wresnet": [256, 512, 1024],
        "moe": [256, 512, 1024],
    }
    
    model_info = models_info[-1]
    job_info = jobs_info[model_info.id]
    gbs = job_batch_size[model_info.model_name][-1]
    job_info.gbs = gbs
    print(f"Model: {model_info.id}")
    print(f"Batch Size: {gbs}")
    
    tic = time.time()
    final_results = calculate_result_for_job(model_info, job_info, device_info, max_workers=9)
    toc = time.time()
    print(f"Time: {toc-tic:.2f} sec")
    
    print(final_results)
    
def test_a_sample():
    model_info = models_info[-1]
    print(model_info.model_name)
    job_info = jobs_info[model_info.id]
    job_info.gbs = 512
    subset = {0: 5, 1: 7, 2:4}
    sub_host_entries, sub_nodes_info = get_sub_host_nodes(
        subset, host_entries, nodes_info
    )
    device_group_info = DeviceGroupInfo(sub_host_entries, sub_nodes_info)
    args = get_arguments(model_info, device_group_info, job_info, subset)
    res = get_estimated_cost(args)
    print(res)


    
if __name__ == "__main__":
    test_a_sample()

# Arguments(model_name='llama2', model_size='26B', home_dir='/home/bahram/projects/Metis/', 
#           host_entries={0: {'ip': 'IP1', 'num_device': 5}, 1: {'ip': 'IP2', 'num_device': 7}, 2: {'ip': 'IP3', 'num_device': 4}}, 
#           nodes_info={'IP1': {'instance_type': 'A6000', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 48}, 
#                       'IP2': {'instance_type': 'A100', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 80}, 
#                       'IP3': {'instance_type': 'RTX4090', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 24}}, 
#           profile_data_path='/home/bahram/projects/Metis/profile/metis/llama2/26B', 
#           gbs=512, num_layers=80, min_group_scale_variance=1, max_permute_len=4, max_profiled_tp_degree=8, max_profiled_batch_size=4, 
#           min_profiled_batch_size=1, subset={0: 5, 1: 7, 2: 4}): 'tp1_bs8'

# Task failed for args Arguments(model_name='llama2', model_size='26B', home_dir='/home/bahram/projects/Metis/', host_entries={0: {'ip': 'IP1', 'num_device': 6}, 1: {'ip': 'IP2', 'num_device': 2}, 2: {'ip': 'IP3', 'num_device': 8}}, nodes_info={'IP1': {'instance_type': 'A6000', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 48}, 'IP2': {'instance_type': 'A100', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 80}, 'IP3': {'instance_type': 'RTX4090', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 24}}, profile_data_path='/home/bahram/projects/Metis/profile/metis/llama2/26B', gbs=512, num_layers=80, min_group_scale_variance=1, max_permute_len=4, max_profiled_tp_degree=8, max_profiled_batch_size=4, min_profiled_batch_size=1, subset={0: 6, 1: 2, 2: 8}): 'tp1_bs8'
# Task failed for args Arguments(model_name='llama2', model_size='26B', home_dir='/home/bahram/projects/Metis/', host_entries={0: {'ip': 'IP1', 'num_device': 6}, 1: {'ip': 'IP2', 'num_device': 6}, 2: {'ip': 'IP3', 'num_device': 4}}, nodes_info={'IP1': {'instance_type': 'A6000', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 48}, 'IP2': {'instance_type': 'A100', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 80}, 'IP3': {'instance_type': 'RTX4090', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 24}}, profile_data_path='/home/bahram/projects/Metis/profile/metis/llama2/26B', gbs=512, num_layers=80, min_group_scale_variance=1, max_permute_len=4, max_profiled_tp_degree=8, max_profiled_batch_size=4, min_profiled_batch_size=1, subset={0: 6, 1: 6, 2: 4}): 'tp1_bs8'