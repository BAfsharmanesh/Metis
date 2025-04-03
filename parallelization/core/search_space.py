from parallelization.core.data_model import GPUAllocation
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

def is_power_of_two(x: int) -> bool:
    """Check if a number is a power of 2."""
    return x > 0 and (x & (x - 1)) == 0

def validate_inputs(gpu_list: List[Tuple[int, int]], min_memory: float, max_memory: float) -> None:
    """Validate input parameters."""
    if not gpu_list:
        raise ValueError("GPU list cannot be empty")
    if min_memory <= 0:
        raise ValueError("Minimum memory must be positive")
    if max_memory < min_memory:
        raise ValueError("Maximum memory must be greater than or equal to minimum memory")
    
def find_gpu_subsets_optimized(
    gpu_list: List[Tuple[int, int]], 
    min_memory: float, 
    max_memory: float
) -> List[Dict[int, int]]:
    """Find valid GPU allocations across nodes that meet memory requirements.

    Args:
        gpu_list: List of tuples [(num_gpus, gpu_memory), ...] for each node
        min_memory: Minimum total memory required
        max_memory: Maximum total memory allowed
        
    Returns:
        List of valid GPU allocations, where each allocation is a dict mapping
        node index to number of GPUs to use from that node.
        
    Example:
        >>> gpu_list = [(8, 48), (8, 48)]  # Two nodes with 8 GPUs of 48GB each
        >>> result = find_gpu_subsets_optimized(gpu_list, min_memory=768, max_memory=1536)
    """
    validate_inputs(gpu_list, min_memory, max_memory)
    
    def backtrack(index: int, current_memory: int, current_count: int, subset: Dict[int, int]) -> None:
        if current_memory >= min_memory and is_power_of_two(current_count):
            # Only store allocations with non-zero GPU counts
            allocation = {k: v for k, v in subset.items() if v > 0}
            if allocation:
                valid_subsets.add(GPUAllocation(
                    node_allocations=allocation,
                    total_memory=current_memory,
                    total_gpus=current_count
                ))

        if index == len(gpu_list) or current_memory >= max_memory:
            return

        num_gpus, gpu_memory = gpu_list[index]
        for use_count in range(num_gpus + 1):
            new_memory = current_memory + use_count * gpu_memory
            if new_memory >= max_memory:
                break

            if use_count > 0:
                subset[index] = use_count
            backtrack(
                index + 1,
                new_memory,
                current_count + use_count,
                subset
            )
            if use_count > 0:
                del subset[index]

    valid_subsets: Set[GPUAllocation] = set()
    backtrack(0, 0, 0, {})
    
    return [allocation.node_allocations for allocation in valid_subsets]



def find_gpu_subsets_optimized_0(gpu_list: List[Tuple], min_memory: float, max_memory: float) -> List[Dict[int, int]]:
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
    def find_unique(list):
        unique_list = []
        for x in list:
            if x not in unique_list:
                unique_list.append(x)
        return unique_list     

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


def main() -> None:
    """Example usage of the GPU allocation function."""
    gpu_list = [(4, 32), (4, 24)]  # Two nodes: 4 GPUs x 32GB and 4 GPUs x 24GB
    min_mem = 20
    max_mem = min_mem * 3
    
    result = find_gpu_subsets_optimized(
        gpu_list=gpu_list,
        min_memory=min_mem,
        max_memory=max_mem
    )
    
    print("\nValid GPU Allocations:")
    for allocation in result:
        total_mem = sum(count * gpu_list[node][1] for node, count in allocation.items())
        total_gpus = sum(allocation.values())
        print(f"Allocation: {allocation}")
        print(f"Total Memory: {total_mem}GB")
        print(f"Total GPUs: {total_gpus}")
        print("-" * 40)

if __name__ == "__main__":
    main()


