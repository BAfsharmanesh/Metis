import os
import subprocess
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from cost_het_cluster import get_estimated_cost
from parallelization.utils import call_silently
from typing import List, Dict, Any
from parallelization.workload import Arguments


class Task:
    """
    Represents a single task to calculate the estimated cost for a given set of arguments.
    """

    def __init__(self, id: int, args: Arguments):
        """
        Initializes the task with its arguments.
        """
        self.args = args
        self.result = None
        self.id = id

    def _silent_func(self, args: Any) -> Any:
        return call_silently(get_estimated_cost)(args)

    def run(self) -> float:
        """
        Executes the task by calling get_estimated_cost with the given arguments.
        """
        # print(f"Processing task with args: {self.args.id}")

        self.result = self._silent_func(self.args)
        # self.result = get_estimated_cost(self.args)
        # print(f"Task completed for args {self.args.id}")

        return self.result[0] if self.result else None


class TaskRunner:
    """
    Manages and runs multiple tasks in parallel using ProcessPoolExecutor.
    """

    def __init__(self, tasks: List[Task], max_workers: int = 4):
        """
        Initializes the TaskRunner with a list of tasks and worker count.
        """
        self.tasks = tasks
        self.max_workers = max_workers

    def _run_single_task(self, task: Task) -> float:
        """
        Helper method to run a single task. Needed for ProcessPoolExecutor.
        """
        return task.run()

    def run_tasks(self, verbose=False) -> List[float]:
        """
        Runs all tasks in parallel using ProcessPoolExecutor.
        Returns a list of results from the tasks.
        """
        print(
            f"Running {len(self.tasks)} tasks with {self.max_workers} workers (processes)..."
        )
        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._run_single_task, task): task
                for task in self.tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append((task.id, task.args.subset, result))
                    if verbose:
                        print(f"Task finished successfully for args {task.args.id}, {task.id}")
                except Exception as e:
                    print(f"Task failed for args {task.args}: {e}")
        return results