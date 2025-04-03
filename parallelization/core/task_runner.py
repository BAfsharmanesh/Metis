import os
import subprocess
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from cost_het_cluster import get_estimated_cost
from parallelization.core.utils import call_silently
from typing import List, Dict, Any, Tuple, Optional
from parallelization.core.workload import Arguments
from parallelization.core.data_model import TaskResult


@dataclass(frozen=True)
class Task:
    """Represents a single cost estimation task."""
    id: int
    args: Arguments

    def _execute_cost_estimation(self) -> Tuple[Optional[float], Optional[str]]:
        """Execute the cost estimation and return result or error."""
        try:
            result = call_silently(get_estimated_cost)(self.args)
            return (result[0] if result else None), None
        except Exception as e:
            return None, str(e)

    def run(self) -> TaskResult:
        """Execute the task and return a TaskResult."""
        cost, error = self._execute_cost_estimation()
        return TaskResult(
            task_id=self.id,
            subset=self.args.subset,
            cost=cost,
            error=error
        )

class TaskRunner:
    """Manages parallel execution of cost estimation tasks."""

    def __init__(self, tasks: List[Task], max_workers: int = 4):
        self.tasks = tasks
        self.max_workers = max_workers

    def run_tasks(self, verbose: bool = False) -> List[TaskResult]:
        """Execute all tasks in parallel and return results."""
        print(f"Running {len(self.tasks)} tasks with {self.max_workers} workers (processes)...")
        
        results: List[TaskResult] = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(task.run): task
                for task in self.tasks
            }
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    if verbose and result.is_success:
                        print(f"Task {task.id} completed successfully")
                    elif not result.is_success:
                        print(f"Task {task.id} failed: {result.error}, not is_success")
                except Exception as e:
                    print(f"Task {task.id} failed: {str(e)}, Exception")
                    results.append(TaskResult(
                        task_id=task.id,
                        subset=task.args.subset,
                        cost=None,
                        error=f"Execution error: {str(e)}"
                    ))
                    
        return results
