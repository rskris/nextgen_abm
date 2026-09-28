"""Multi-Processing Parallel Household MILP Solver.

Distributes independent household MILP optimizations across all available CPU cores
(e.g. 12 cores on Apple Silicon) using ProcessPoolExecutor for high throughput.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..config import MasterConfig, get_config
from .household_milp import (
    EscortSpec,
    HouseholdMILPResult,
    JointActivitySpec,
    MemberAgenda,
    UnifiedHouseholdMILP,
)
from .assets import VehicleAsset


@dataclass
class HouseholdTask:
    """Serializable task descriptor for a single household optimization."""
    household_id: str
    members: List[MemberAgenda]
    vehicles: List[VehicleAsset]
    joint_activities: List[JointActivitySpec] = field(default_factory=list)
    escorts: List[EscortSpec] = field(default_factory=list)


def _solve_single_household_worker(task: HouseholdTask, config_dump: Dict[str, Any]) -> HouseholdMILPResult:
    """Worker function executed inside a child process."""
    config = MasterConfig.model_validate(config_dump)
    solver = UnifiedHouseholdMILP(config=config)
    return solver.solve_household(
        household_id=task.household_id,
        members=task.members,
        vehicles=task.vehicles,
        joint_activities=task.joint_activities,
        escorts=task.escorts,
    )


class ParallelHouseholdSolver:
    """High-throughput multi-core household scheduling manager."""

    def __init__(self, config: Optional[MasterConfig] = None, max_workers: Optional[int] = None):
        self.config = config or get_config()
        self.max_workers = max_workers or (os.cpu_count() or 4)

    def solve_batch(
        self,
        tasks: List[HouseholdTask],
        chunksize: int = 50,
    ) -> Tuple[List[HouseholdMILPResult], float]:
        """Solve a list of household tasks in parallel across all worker processes.
        
        Returns:
            Tuple of (results_list, total_elapsed_seconds)
        """
        start_time = time.perf_counter()
        config_dump = self.config.model_dump()
        results: List[HouseholdMILPResult] = []

        if len(tasks) <= 2 or self.max_workers == 1:
            # Single process fast path for small sets
            solver = UnifiedHouseholdMILP(config=self.config)
            for t in tasks:
                results.append(
                    solver.solve_household(
                        household_id=t.household_id,
                        members=t.members,
                        vehicles=t.vehicles,
                        joint_activities=t.joint_activities,
                        escorts=t.escorts,
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(_solve_single_household_worker, task, config_dump)
                    for task in tasks
                ]
                for fut in as_completed(futures):
                    results.append(fut.result())

        elapsed = time.perf_counter() - start_time
        return results, elapsed
