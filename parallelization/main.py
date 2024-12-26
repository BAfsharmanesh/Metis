from parallelization.workload import Arguments
from parallelization.task_runner import Task, TaskRunner
from parallelization.workload import (
    Arguments,
    ModelInfo,
    DeviceGroupInfo,
    JobInfo,
    get_arguments,
    models_info,
    jobs_info,
    gpus_info,
)
from parallelization.search_space import find_gpu_subsets_optimized
from cost_het_cluster import get_estimated_cost

# Example usage
if __name__ == "__main__":

    nodes_info = {
        "IP1": {
            "instance_type": "A6000",
            "inter_bandwidth": 312500000.0,
            "intra_bandwidth": 5312500000.0,
            "memory": gpus_info["A6000"]["memory"],
        },
        "IP2": {
            "instance_type": "A100",
            "inter_bandwidth": 312500000.0,
            "intra_bandwidth": 5312500000.0,
            "memory": gpus_info["A100"]["memory"],
        },
        "IP3": {
            "instance_type": "RTX4090",
            "inter_bandwidth": 312500000.0,
            "intra_bandwidth": 5312500000.0,
            "memory": gpus_info["RTX4090"]["memory"],
        },
    }

    host_entries = {
        0: {"ip": "IP1", "num_device": 8},
        1: {"ip": "IP2", "num_device": 8},
        2: {"ip": "IP3", "num_device": 8},
    }

    # example task
    model_info = models_info[-1]
    job_info = jobs_info[model_info.id]

    device_info = [
        (v["num_device"], nodes_info[v["ip"]]["memory"])
        for k, v in host_entries.items()
    ]
    print(device_info)

    # for the model min_memory, go to the profiled data and get the memory of tp{1}_bs{min_prof_bs}
    import os

    profile_path = os.path.join(job_info.home_dir, job_info.profile_path)
    instance_type = list(nodes_info.values())[0]["instance_type"]
    profile_path_min = os.path.join(
        profile_path, f"DeviceType.{instance_type}_tp{1}_bs{job_info.min_prof_bs}.json"
    )
    profile_path_max = os.path.join(
        profile_path, f"DeviceType.{instance_type}_tp{1}_bs{job_info.max_prof_bs}.json"
    )

    # read json file as dictionary
    import json

    with open(profile_path_min, "r") as f:
        min_memory = json.load(f)["execution_memory"]["total_memory_mb"]
        min_memory = int(min_memory) / 1024.0
        print(min_memory, min_memory / job_info.min_prof_bs * job_info.gbs)
    with open(profile_path_max, "r") as f:
        max_memory = json.load(f)["execution_memory"]["total_memory_mb"]
        max_memory = int(max_memory) / 1024.0
        print(max_memory, max_memory / job_info.max_prof_bs * job_info.gbs)
    # find the optimal gpu subsets
    memory_demand = sorted(
        [min_memory,
         max_memory,
            min_memory / job_info.min_prof_bs * job_info.gbs,
            max_memory / job_info.max_prof_bs * job_info.gbs,
        ]
    )
    print(memory_demand)
    result = find_gpu_subsets_optimized(device_info, min(memory_demand), max(memory_demand))
    print(result, len(result))

    # {1: 1, 2: 3}
    import copy

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

    tasks = []
    args_list = []
    for subset in result:
        sub_host_entries, sub_nodes_info = get_sub_host_nodes(
            subset, host_entries, nodes_info
        )
        device_group_info = DeviceGroupInfo(sub_host_entries, sub_nodes_info)

        args = get_arguments(model_info, device_group_info, job_info)

        tasks.append(Task(args))
        args_list.append(args)

    # Create and run the TaskRunner
    # runner = TaskRunner(tasks, max_workers=4)  # Adjust max_workers for your CPU
    # results = runner.run_tasks()
    # args = Arguments(model_name='llama2', model_size='26B', home_dir='/home/bahram/Projects/Metis/', host_entries={0: {'ip': 'IP1', 'num_device': 2}, 1: {'ip': 'IP2', 'num_device': 5}, 2: {'ip': 'IP3', 'num_device': 1}}, nodes_info={'IP1': {'instance_type': 'A6000', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 48}, 'IP2': {'instance_type': 'A100', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 80}, 'IP3': {'instance_type': 'RTX4090', 'inter_bandwidth': 312500000.0, 'intra_bandwidth': 5312500000.0, 'memory': 24}}, profile_data_path='/home/bahram/Projects/Metis/profile/metis/llama2/26B', gbs=128, num_layers=80, min_group_scale_variance=1, max_permute_len=4, max_profiled_tp_degree=8, max_profiled_batch_size=4, min_profiled_batch_size=1)
    args = args_list[-1]
    # results = get_estimated_cost(args)
    # print(results[:10])
    # print(args, result[-1])


    # for each workload, create a task, e.g., LLAMA2_7B_128: workloads['llama2']['7B']
    # tasks = []
    # for model_name, model_workloads in workloads.items():
    #     for model_size, workload in model_workloads.items():
    #         args = Arguments(
    #             model_name=model_name,
    #             model_size=model_size,
    #             home_dir=base_args.home_dir,
    #             host_entries=base_args.host_entries,
    #             nodes_info=base_args.nodes_info,
    #             profile_path=f"profile/metis/{model_name}/{model_size}",
    #             gbs=workload["gbs"],
    #             num_layers=workload["num_layers"],
    #             max_prof_tpd=workload["max_prof_tpd"],
    #             max_prof_bs=workload["max_prof_bs"],
    #             min_prof_bs=workload["min_prof_bs"],
    #             min_group_scale_variance=base_args.min_group_scale_variance,
    #             max_permute_len=base_args.max_permute_len,
    #         )
    #         tasks.append(Task(args))

    # # Create and run the TaskRunner
    # runner = TaskRunner(tasks, max_workers=4)  # Adjust max_workers for your CPU
    # results = runner.run_tasks()

    # # Print final results
    # print("\nAll tasks completed. Results:")
    # for id, result in results:
    #     print(f"task_id: {id}, Estimated Cost: {result}")

    # print(tasks[-1].args)
    # res = get_estimated_cost(tasks[-1].args)
    # print(res)
