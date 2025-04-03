import unittest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.search_space import (
    find_gpu_subsets_optimized,
    find_gpu_subsets_optimized_0,
)

class TestGPUSubsetFinders(unittest.TestCase):
    def setUp(self):
        # Common test cases
        self.test_cases = [
            {
                "gpu_list": [(8, 48), (8, 48)],  # Two nodes with 8 GPUs of 48GB each
                "min_memory": 40,
                "max_memory": 1536,
                "name": "two_identical_nodes"
            },
            {
                "gpu_list": [(4, 32), (4, 24)],  # Different memory GPUs
                "min_memory": 20,
                "max_memory": 60,
                "name": "different_memory_gpus"
            },
            {
                "gpu_list": [(2, 16), (2, 16), (2, 16)],  # Three identical small nodes
                "min_memory": 16,
                "max_memory": 96,
                "name": "three_small_nodes"
            },
            {
                "gpu_list": [(8, 48), (8, 32), (8, 24)],  # Three large nodes
                "min_memory": 30,
                "max_memory": 1536,
                "name": "three_large_nodes"
            }
        ]

    def test_functions_produce_same_results(self):
        """Test that both implementations produce the same results."""
        for case in self.test_cases:
            with self.subTest(case=case["name"]):
                result1 = find_gpu_subsets_optimized(
                    case["gpu_list"],
                    case["min_memory"],
                    case["max_memory"]
                )
                result2 = find_gpu_subsets_optimized_0(
                    case["gpu_list"],
                    case["min_memory"],
                    case["max_memory"]
                )
                
                # Convert results to sets of frozensets for comparison
                result1_set = {frozenset(d.items()) for d in result1}
                result2_set = {frozenset(d.items()) for d in result2}
                
                self.assertEqual(result1_set, result2_set,
                    f"Results differ for case {case['name']}")

    def test_power_of_two_constraint(self):
        """Test that all allocations use a power of 2 total GPUs."""
        for case in self.test_cases:
            with self.subTest(case=case["name"]):
                results = find_gpu_subsets_optimized(
                    case["gpu_list"],
                    case["min_memory"],
                    case["max_memory"]
                )
                
                for allocation in results:
                    total_gpus = sum(allocation.values())
                    self.assertEqual(bin(total_gpus).count('1'), 1,
                        f"Total GPUs {total_gpus} is not a power of 2")

    def test_memory_constraints(self):
        """Test that all allocations meet memory constraints."""
        for case in self.test_cases:
            with self.subTest(case=case["name"]):
                results = find_gpu_subsets_optimized(
                    case["gpu_list"],
                    case["min_memory"],
                    case["max_memory"]
                )
                
                for allocation in results:
                    total_memory = sum(
                        count * case["gpu_list"][node][1]
                        for node, count in allocation.items()
                    )
                    self.assertGreaterEqual(
                        total_memory,
                        case["min_memory"],
                        f"Allocation has less than minimum memory: {total_memory} < {case['min_memory']}"
                    )
                    self.assertLessEqual(
                        total_memory,
                        case["max_memory"],
                        f"Allocation exceeds maximum memory: {total_memory} > {case['max_memory']}"
                    )

    def test_invalid_inputs(self):
        """Test that invalid inputs raise appropriate exceptions."""
        with self.assertRaises(ValueError):
            find_gpu_subsets_optimized([], 100, 200)  # Empty GPU list
        
        with self.assertRaises(ValueError):
            find_gpu_subsets_optimized([(4, 32)], -1, 100)  # Negative min memory
        
        with self.assertRaises(ValueError):
            find_gpu_subsets_optimized([(4, 32)], 100, 50)  # max_memory < min_memory

if __name__ == '__main__':
    unittest.main() 