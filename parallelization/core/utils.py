import json
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Dict, List

from parallelization.core.data_model import (ExecutionMemory, ExecutionTime,
                                             Model, ModelMetrics, Parameters)
from parallelization.core.workload import gpus_info


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


def extract_tp_bs(filename):
    # Match the pattern for tp and bs
    match = re.search(r"_tp(\d+)_bs(\d+)\.json", filename)
    if match:
        tp_number = int(match.group(1))  # Extract tp number
        bs_number = int(match.group(2))  # Extract bs number
        return tp_number, bs_number
    else:
        raise ValueError("Filename does not match the expected pattern")


def manipulate_write_new_file(json_file_path, new_file_path):
    """Manipulate the JSON file and write to a new path."""
    json_file_path = Path(json_file_path)

    tp_tmp, bs_tmp = extract_tp_bs(json_file_path.name)
    json_data = read_json_file(json_file_path)
    json_data = json_2_model(json_data)

    tmp = json_data.execution_memory.layer_memory_total_mb
    json_data.execution_memory.layer_memory_total_mb = [i * tp_tmp for i in tmp]

    tmp = json_data.execution_memory.total_memory_mb
    json_data.execution_memory.total_memory_mb = tmp * tp_tmp
    json_data = json.dumps(asdict(json_data), indent=2)

    with open(new_file_path, "w") as f:
        f.write(json_data)
    print(f"Corrected and Written: {new_file_path}")


def create_dummy_profile(json_file_path, new_file_path):

    new_gpus = ["A100", "RTX4090"]

    alpha = 0.7
    beta = 0.3
    profiled_gpus = "A6000"
    new_gpus_throughput = {
        gpu: alpha
        * gpus_info[gpu]["tensor_fp8_tflops"]
        / gpus_info[profiled_gpus]["tensor_fp8_tflops"]
        + beta
        * gpus_info[gpu]["mem_bandwidth"]
        / gpus_info[profiled_gpus]["mem_bandwidth"]
        for gpu in new_gpus
    }

    # print(f"{new_gpus_throughput=}")

    json_file_path = Path(json_file_path)

    json_data = read_json_file(json_file_path)
    json_data = json_2_model(json_data)

    for gpu in new_gpus:
        json_data.execution_time.total_time_ms = (
            json_data.execution_time.total_time_ms / new_gpus_throughput[gpu]
        )
        json_data.execution_time.forward_backward_time_ms = (
            json_data.execution_time.forward_backward_time_ms / new_gpus_throughput[gpu]
        )
        json_data.execution_time.batch_generator_time_ms = (
            json_data.execution_time.batch_generator_time_ms / new_gpus_throughput[gpu]
        )
        # json_data.execution_time.layernorm_grads_all_reduce_time_ms /= new_gpus_throughput[gpu]
        # json_data.execution_time.embedding_grads_all_reduce_time_ms /= new_gpus_throughput[gpu]
        json_data.execution_time.optimizer_time_ms = (
            json_data.execution_time.optimizer_time_ms / new_gpus_throughput[gpu]
        )
        json_data.execution_time.layer_compute_total_ms = [
            i / new_gpus_throughput[gpu]
            for i in json_data.execution_time.layer_compute_total_ms
        ]
        json_data_dump = json.dumps(asdict(json_data), indent=2)

        new_file_path_gpu = new_file_path.replace("A6000", gpu)
        with open(new_file_path_gpu, "w") as f:
            f.write(json_data_dump)
        print(f"Dummy Data for: {new_file_path_gpu}")
