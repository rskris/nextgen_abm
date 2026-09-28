"""Tests for DataSyncManager, ParallelHouseholdSolver, and CLI entrypoint."""

import pytest
from pathlib import Path
from nextgen_abm.config import get_config
from nextgen_abm.data_sync import DataSyncManager, SyncReport
from nextgen_abm.household.parallel_milp import ParallelHouseholdSolver, HouseholdTask
from nextgen_abm.household.household_milp import MemberAgenda
from nextgen_abm.household.assets import VehicleAsset
from nextgen_abm.scheduler.milp import ActivityDefinition
from nextgen_abm.cli import build_parser


def test_data_sync_pipeline(tmp_path):
    cfg = get_config()
    cfg.spatial.cache_dir = str(tmp_path / "cache")
    mgr = DataSyncManager(config=cfg)

    report = mgr.sync_all()
    assert report.status == "completed"
    assert len(report.files_synced) >= 4
    assert report.census_households_count > 0
    assert report.lodes_od_pairs_count > 0
    assert report.overture_buildings_count > 0


def test_parallel_household_solver():
    cfg = get_config()
    solver = ParallelHouseholdSolver(config=cfg, max_workers=2)

    home_loc = [("bldg_home", (-119.82, 34.43))]
    work_loc = [("bldg_work", (-119.70, 34.42))]

    tasks = [
        HouseholdTask(
            household_id=f"test_hh_{i}",
            members=[
                MemberAgenda(
                    person_id=f"p_{i}",
                    is_adult=True,
                    has_license=True,
                    activities=[
                        ActivityDefinition("h1", "home_morning", True, home_loc, 0.5, 3.0, 6.0, 9.0),
                        ActivityDefinition("w", "work", True, work_loc, 6.0, 9.0, 8.0, 18.0),
                        ActivityDefinition("h2", "home_night", True, home_loc, 4.0, 10.0, 17.0, 24.0),
                    ],
                )
            ],
            vehicles=[VehicleAsset(f"v_{i}", f"test_hh_{i}", "ev", "bldg_home", "bldg_home")],
        )
        for i in range(4)
    ]

    results, elapsed = solver.solve_batch(tasks)
    assert len(results) == 4
    assert elapsed > 0.0
    for r in results:
        assert r.status == "optimal"


def test_cli_subcommands_parser():
    parser = build_parser()

    # Test sync-data
    args_sync = parser.parse_args(["sync-data", "--config", "config.yaml"])
    assert args_sync.command == "sync-data"

    # Test run
    args_run = parser.parse_args(["run", "--sample", "0.1", "--iterations", "3", "--workers", "4"])
    assert args_run.command == "run"
    assert args_run.sample == 0.1
    assert args_run.iterations == 3
    assert args_run.workers == 4

    # Test calibrate
    args_calib = parser.parse_args(["calibrate", "--iterations", "5"])
    assert args_calib.command == "calibrate"
    assert args_calib.iterations == 5

    # Test dashboard
    args_dash = parser.parse_args(["dashboard", "--output-dir", "custom_outputs"])
    assert args_dash.command == "dashboard"
    assert args_dash.output_dir == "custom_outputs"
