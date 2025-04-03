import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Dict, List

from parallelization.core.data_model import (ExecutionMemory, ExecutionTime,
                                             Model, ModelMetrics, Parameters)
from parallelization.core.workload import GPUSpecs, GPUType


# Suppress the output
def call_silently(func):
    """
    Decorator to suppress stdout output from a function.

    Args:
        func: The function whose output should be suppressed

    Returns:
        Wrapped function that executes silently
    """

    def wrapper(*args, **kwargs):
        with StringIO() as f, redirect_stdout(f):
            return func(*args, **kwargs)

    return wrapper


def json_2_model(json_data):
    """
    Convert JSON data to a structured ModelMetrics object.

    Args:
        json_data: Dictionary containing model data from JSON

    Returns:
        ModelMetrics object with parsed data
    """
    parameters = Parameters(**json_data["model"]["parameters"])
    model = Model(
        model_name=json_data["model"]["model_name"],
        parameters=parameters,
        num_layers=json_data["model"]["num_layers"],
    )
    execution_time = ExecutionTime(**json_data["execution_time"])
    execution_memory = ExecutionMemory(**json_data["execution_memory"])
    model_metrics = ModelMetrics(
        model=model, execution_time=execution_time, execution_memory=execution_memory
    )
    return model_metrics


def read_json_file(file_name):
    """
    Read and parse a JSON file into a Python dictionary.

    Args:
        file_name: Path to the JSON file

    Returns:
        Dictionary containing the parsed JSON data
    """
    with open(file_name, "r") as f:
        data = json.load(f)
    return dict(data)


import re


def extract_tp_bs(filename: str) -> tuple[int, int]:
    """
    Extract tensor parallelism and batch size numbers from filename.
    
    Args:
        filename: Input filename containing tp and bs numbers
        
    Returns:
        Tuple of (tp_number, bs_number)
        
    Raises:
        ValueError: If filename doesn't match expected pattern
    """
    match = re.search(r"_tp(\d+)_bs(\d+)\.json", filename)
    if match:
        return int(match.group(1)), int(match.group(2)) # (tp_number, bs_number)
    raise ValueError("Filename does not match the expected pattern")


def manipulate_write_new_file(json_file_path: str | Path, new_file_path: str | Path) -> None:
    """
    Manipulate JSON file by scaling memory values and write to new path.
    
    Args:
        json_file_path: Path to source JSON file
        new_file_path: Path to write modified JSON
    """
    json_file_path = Path(json_file_path)
    tp_value, _ = extract_tp_bs(json_file_path.name)
    
    json_data = json_2_model(read_json_file(json_file_path))
    
    # Scale memory values by tp_value
    json_data.execution_memory.layer_memory_total_mb = [
        mem * tp_value for mem in json_data.execution_memory.layer_memory_total_mb
    ]
    json_data.execution_memory.total_memory_mb *= tp_value
    
    with open(new_file_path, "w") as f:
        json.dump(asdict(json_data), f, indent=2)
    print(f"Corrected and Written: {new_file_path}")


def create_dummy_profile(json_file_path: str | Path, new_file_path: str | Path) -> None:
    """
    Create dummy profiles for different GPU types based on a reference profile.
    
    Args:
        json_file_path: Path to source JSON file
        new_file_path: Path template for new GPU profiles
    """
    NEW_GPUS = [GPUType.A100, GPUType.RTX4090]
    PROFILED_GPU = GPUType.A6000
    ALPHA, BETA = 0.7, 0.3  # Weights for throughput calculation
    
    # Calculate relative throughput for each new GPU
    new_gpus_throughput = {
        gpu: ALPHA * (GPUSpecs.SPECS[gpu]["tensor_fp8_tflops"] / GPUSpecs.SPECS[PROFILED_GPU]["tensor_fp8_tflops"]) +
             BETA * (GPUSpecs.SPECS[gpu]["mem_bandwidth"] / GPUSpecs.SPECS[PROFILED_GPU]["mem_bandwidth"])
        for gpu in NEW_GPUS
    }
    
    json_data = json_2_model(read_json_file(Path(json_file_path)))
    
    for gpu in NEW_GPUS:
        throughput = new_gpus_throughput[gpu]
        
        # Scale execution times by GPU throughput
        json_data.execution_time.total_time_ms /= throughput
        json_data.execution_time.forward_backward_time_ms /= throughput
        json_data.execution_time.batch_generator_time_ms /= throughput
        json_data.execution_time.optimizer_time_ms /= throughput
        json_data.execution_time.layer_compute_total_ms = [
            time / throughput for time in json_data.execution_time.layer_compute_total_ms
        ]
        
        # Write GPU-specific profile
        new_file_path_gpu = str(new_file_path).replace(PROFILED_GPU, gpu)
        with open(new_file_path_gpu, "w") as f:
            json.dump(asdict(json_data), indent=2)
        print(f"Dummy Data for: {new_file_path_gpu}")
