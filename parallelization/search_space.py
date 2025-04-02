from math import log2
from typing import List, Dict, Any, Tuple
from itertools import combinations
from collections import Counter

def find_unique(list):
    unique_list = []
    for x in list:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list 

def find_gpu_subsets_optimized(gpu_list: List[Tuple], min_memory: float, max_memory: float) -> List[Dict[int, int]]:
    """_summary_

    Args:
        gpu_list (List[Tuple]): a list of tuples, each tuple contains the number of GPUs and the memory of each node
                                e.g. [node1, node2, ...] = [(# of GPU1, memory of GPU1), (# of GPU2, memory of GPU2), ...]
        min_memory (int): memory needed for the task

    Returns:
        List[Dict[int, int]]: a list of dictionaries, each dictionary contains the number of GPUs for each node
                                e.g. [{node1: # of GPU1, node2: # of GPU2, ...}, ...]
    
    Example usage:
        gpu_list = [(8, 48), (8, 48), (8, 48)]  # 8*GPU1 with 48Gb memory, 8*GPU2 with 48Gb memory, 8*GPU3 with 48Gb memory
        min_memory = 768  
        max_memory = 1536
        result = find_gpu_subsets_optimized(gpu_list, min_memory, max_memory)    
    """

    def is_power_of_two(x):
        return x > 0 and (x & (x - 1)) == 0

    def generate_subsets():
        subsets_tmp = []
        for node_i, (count, memory) in enumerate(gpu_list):
            subsets_tmp.append((node_i, count, memory))
        return subsets_tmp

    def backtrack(index, current_memory, current_count, subset):
        if current_memory >= min_memory and is_power_of_two(current_count):
            valid_subsets.append(dict(subset))
            

        if index == len(subsets) or current_memory >= max_memory:
            return

        node_i, count, memory = subsets[index]
        for use_count in range(count + 1):
            if current_memory + use_count * memory >= max_memory:
                break

            subset[node_i] = subset.get(node_i, 0) + use_count
            backtrack(
                index + 1,
                current_memory + use_count * memory,
                current_count + use_count,
                subset,
            )
            if use_count > 0:
                subset[node_i] -= use_count
                if subset[node_i] == 0:
                    del subset[node_i]

    subsets = generate_subsets()
    valid_subsets = []
    backtrack(0, 0, 0, {})
    
    # iterate over all subsets, if number of gpu for a node is zero, remove it from the subset: {0: 0, 1: 8, 2: 8} -> {1: 8, 2: 8}
    for subset in valid_subsets:
        for node in list(subset.keys()):
            if subset[node] == 0:
                del subset[node]       
    return find_unique(valid_subsets)


if __name__ == "__main__":
    gpu_list = [(4, 32), (4, 24)]
    min_mem = 20
    result = find_gpu_subsets_optimized(gpu_list, min_memory=min_mem, max_memory=min_mem*3)    
    print(result)
    for res in result:
        total_mem = 0
        for node, count in res.items():
            total_mem += count * gpu_list[node][1]
        print(f"Node: {res}, Total memory: {total_mem}, GPU count: {sum(res.values())}")


