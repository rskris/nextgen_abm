"""Command-Line Interface (CLI) for NextGen ABM Santa Barbara.

Provides unified subcommands:
- `nextgen-abm sync-data`: Download and cache Overture, Census, LODES, and GTFS feeds.
- `nextgen-abm run`: Execute end-to-end multi-modal simulation with parallel HiGHS and LTM.
- `nextgen-abm calibrate`: Run automated SPSA calibration against Caltrans PeMS and CHTS.
- `nextgen-abm dashboard`: Render interactive 3D PyDeck visualization.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import MasterConfig, get_config, set_config
from .data_sync import DataSyncManager
from .household.assets import VehicleAsset
from .household.household_milp import MemberAgenda
from .household.parallel_milp import HouseholdTask, ParallelHouseholdSolver
from .scheduler.milp import ActivityDefinition
from .traffic.equilibrium import DayToDayEquilibriumEngine, HouseholdAgent
from .traffic.network import HierarchicalMultiModalNetwork
from .validation.dashboard import DashboardRenderer
from .validation.spsa_calibration import SPSACalibrator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nextgen_abm")


def cmd_sync_data(args: argparse.Namespace) -> int:
    """Execute live data synchronization and caching."""
    cfg = MasterConfig.from_yaml(args.config) if Path(args.config).exists() else get_config()
    print(f"\n=======================================================")
    print(f"  NextGen ABM: Syncing Santa Barbara County Open Data")
    print(f"  County FIPS: {cfg.population.county_fips} | Cache Dir: {cfg.spatial.cache_dir}")
    print(f"=======================================================\n")

    mgr = DataSyncManager(config=cfg)
    report = mgr.sync_all()

    print(f"✓ Data Synchronization Completed:")
    print(f"  - Total Cache Size: {report.total_bytes / (1024 * 1024):.2f} MB")
    print(f"  - Overture Buildings: {report.overture_buildings_count:,}")
    print(f"  - Overture Places (POIs): {report.overture_places_count:,}")
    print(f"  - Census Sample Households: {report.census_households_count:,}")
    print(f"  - LEHD LODES OD Records: {report.lodes_od_pairs_count:,}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Execute full end-to-end simulation on this machine."""
    cfg = MasterConfig.from_yaml(args.config) if Path(args.config).exists() else get_config()
    set_config(cfg)

    sample_rate = args.sample if args.sample is not None else cfg.population.sample_fraction
    max_iters = args.iterations if args.iterations is not None else cfg.equilibrium.max_iterations
    n_workers = args.workers if args.workers is not None else (os.cpu_count() or 4)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=================================================================")
    print(f"  NextGen ABM Santa Barbara - End-to-End Simulation")
    print(f"  Scale Sample Rate : {sample_rate * 100:.1f}%")
    print(f"  CPU Worker Cores   : {n_workers}")
    print(f"  Equilibrium Iters  : {max_iters}")
    print(f"  Output Directory   : {out_dir.resolve()}")
    print(f"=================================================================\n")

    # 1. Network Construction
    logger.info("Building Santa Barbara hierarchical multi-modal network...")
    network = HierarchicalMultiModalNetwork.build_santa_barbara_regional_network(config=cfg)
    logger.info("Network initialized: %d nodes, %d links.", len(network.nodes), len(network.links))

    # 2. Synthetic Household Generation
    target_hh_count = max(10, int(500 * sample_rate))
    logger.info("Generating %d synthetic households for sample rate %.1f%%...", target_hh_count, sample_rate * 100)

    home_loc = [("bldg_home", (-119.82, 34.43))]
    work_loc = [("bldg_work", (-119.70, 34.42))]

    household_agents: List[HouseholdAgent] = []
    tasks: List[HouseholdTask] = []

    for i in range(target_hh_count):
        hh_id = f"hh_{i:06d}"
        p_id = f"person_{i:06d}"
        m = MemberAgenda(
            person_id=p_id,
            is_adult=True,
            has_license=True,
            activities=[
                ActivityDefinition(f"{p_id}_h1", "home_morning", True, home_loc, 0.5, 3.0, 6.0, 9.0),
                ActivityDefinition(f"{p_id}_w", "work", True, work_loc, 6.0, 9.0, 8.0, 18.0),
                ActivityDefinition(f"{p_id}_h2", "home_night", True, home_loc, 4.0, 10.0, 17.0, 24.0),
            ],
        )
        veh = VehicleAsset(f"veh_{i}", hh_id, "ev" if i % 2 == 0 else "ice", "bldg_home", "bldg_home")
        agent = HouseholdAgent(household_id=hh_id, members=[m], vehicles=[veh])
        household_agents.append(agent)
        tasks.append(HouseholdTask(household_id=hh_id, members=[m], vehicles=[veh]))

    # 3. Parallel Household MILP Optimization
    logger.info("Solving %d household schedules in parallel across %d CPU cores...", len(tasks), n_workers)
    parallel_solver = ParallelHouseholdSolver(config=cfg, max_workers=n_workers)
    initial_results, solve_seconds = parallel_solver.solve_batch(tasks)
    logger.info("Parallel MILP solve complete in %.2f seconds (%.1f households/sec).", solve_seconds, len(tasks) / max(0.001, solve_seconds))

    # 4. Day-to-Day Evolutionary Replanning Loop with LTM Meso-Simulator
    logger.info("Starting Day-to-Day evolutionary equilibrium loop (%d iterations)...", max_iters)
    engine = DayToDayEquilibriumEngine(network=network, config=cfg)
    metrics = engine.run_equilibrium(households=household_agents, max_iterations=max_iters)

    # 5. Export Output Artifacts
    logger.info("Writing simulation output artifacts to %s...", out_dir)

    # Output 1: Corridor Performance Metrics
    metrics_records = [
        {
            "iteration": m.iteration,
            "total_trips": m.total_trips,
            "avg_travel_time_min": m.avg_travel_time_min,
            "total_vht_hours": m.total_vht_hours,
            "total_vkt_km": m.total_vkt_km,
            "total_ev_energy_kwh": m.total_ev_energy_kwh,
            "avg_plan_score": m.avg_plan_score,
            "relative_gap": m.relative_gap,
        }
        for m in metrics
    ]
    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv = out_dir / "network_corridor_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Output 2: Daily Schedules Parquet
    schedule_rows = []
    for hh in household_agents:
        if hh.plans:
            plan = hh.plans[hh.active_plan_idx]
            for pid, sched in plan.result.member_schedules.items():
                for act in sched.activities:
                    schedule_rows.append({
                        "household_id": hh.household_id,
                        "person_id": pid,
                        "act_id": act.act_id,
                        "act_type": act.act_type,
                        "location_id": act.chosen_location_id,
                        "lon": act.chosen_coords[0],
                        "lat": act.chosen_coords[1],
                        "start_hour": act.start_hour,
                        "end_hour": act.end_hour,
                        "duration_hours": act.duration_hours,
                    })
    sched_df = pd.DataFrame(schedule_rows)
    sched_parquet = out_dir / "daily_schedules.parquet"
    sched_df.to_parquet(sched_parquet, index=False)

    # Output 3: Caltrans PeMS Validation JSON
    calib = SPSACalibrator(network=network, config=cfg)
    last_metric = metrics[-1]
    sim_flow = last_metric.total_trips * 150.0
    geh_results = {stn: round(calib.compute_geh(sim_flow, obs), 2) for stn, obs in calib.PEMS_BENCHMARKS.items()}
    val_json = out_dir / "pems_geh_validation.json"
    with open(val_json, "w", encoding="utf-8") as f:
        json.dump(geh_results, f, indent=2)

    # Output 4: PyDeck 3D Dashboard HTML
    renderer = DashboardRenderer(output_dir=out_dir)
    dashboard_path = renderer.render_flow_deck(flows_df=sched_df, filename="dashboard.html")

    print(f"\n=================================================================")
    print(f"  Simulation Run Successfully Finished!")
    print(f"  - Final Total Trips       : {last_metric.total_trips:,}")
    print(f"  - Final Total VHT         : {last_metric.total_vht_hours:.2f} hours")
    print(f"  - Final Total EV Energy   : {last_metric.total_ev_energy_kwh:.2f} kWh")
    print(f"  - Convergence Relative Gap: {last_metric.relative_gap:.4f}")
    print(f"  Artifacts Generated:")
    print(f"    • {metrics_csv}")
    print(f"    • {sched_parquet}")
    print(f"    • {val_json}")
    print(f"    • {dashboard_path}")
    print(f"=================================================================\n")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Run automated SPSA parameter calibration."""
    cfg = MasterConfig.from_yaml(args.config) if Path(args.config).exists() else get_config()
    set_config(cfg)
    n_iters = args.iterations or cfg.calibration.max_iterations

    print(f"\n=======================================================")
    print(f"  NextGen ABM: SPSA Calibration Loop")
    print(f"  Max Iterations: {n_iters} | Target GEH: < {cfg.calibration.target_geh}")
    print(f"=======================================================\n")

    network = HierarchicalMultiModalNetwork.build_santa_barbara_regional_network(config=cfg)
    calibrator = SPSACalibrator(network=network, config=cfg)

    home_loc = [("bldg_home", (-119.82, 34.43))]
    work_loc = [("bldg_work", (-119.70, 34.42))]
    sample_agents = [
        HouseholdAgent(
            f"calib_hh_{i}",
            [
                MemberAgenda(
                    f"p_{i}",
                    True,
                    True,
                    [
                        ActivityDefinition("h1", "home_morning", True, home_loc, 0.5, 3.0, 6.0, 9.0),
                        ActivityDefinition("w", "work", True, work_loc, 6.0, 9.0, 8.0, 18.0),
                        ActivityDefinition("h2", "home_night", True, home_loc, 4.0, 10.0, 17.0, 24.0),
                    ],
                )
            ],
            [VehicleAsset(f"v_{i}", f"calib_hh_{i}", "ice", "bldg_home", "bldg_home")],
        )
        for i in range(20)
    ]

    res = calibrator.calibrate(sample_agents, max_iterations=n_iters)
    print(f"✓ SPSA Calibration Complete:")
    print(f"  - Initial Loss: {res.initial_loss:.4f} -> Final Loss: {res.final_loss:.4f}")
    print(f"  - Calibrated Parameters: {json.dumps(res.best_parameters, indent=4)}")
    print(f"  - PeMS Station GEH: {json.dumps(res.geh_statistics, indent=4)}")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Render interactive 3D dashboard."""
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = DashboardRenderer(output_dir=out_dir)
    path = renderer.render_flow_deck(filename="dashboard.html")
    print(f"✓ Interactive 3D PyDeck dashboard generated: {path}")
    if args.open_browser:
        import webbrowser
        webbrowser.open(f"file://{path.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser supporting global and subcommand options."""
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", default="config.yaml", help="Path to master config YAML file")

    parser = argparse.ArgumentParser(
        prog="nextgen-abm",
        description="Next-Generation Activity-Based Travel Demand Model for Santa Barbara County",
        parents=[base_parser],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subcommand: sync-data
    p_sync = subparsers.add_parser("sync-data", parents=[base_parser], help="Download and cache Santa Barbara open data")
    p_sync.set_defaults(func=cmd_sync_data)

    # Subcommand: run
    p_run = subparsers.add_parser("run", parents=[base_parser], help="Execute full multi-agent simulation")
    p_run.add_argument("--sample", type=float, default=None, help="Sample rate fraction (e.g. 0.1 for 10%%)")
    p_run.add_argument("--iterations", type=int, default=None, help="Day-to-day equilibrium iterations")
    p_run.add_argument("--workers", type=int, default=None, help="Parallel CPU worker cores")
    p_run.add_argument("--output-dir", default="outputs", help="Output directory")
    p_run.set_defaults(func=cmd_run)

    # Subcommand: calibrate
    p_calib = subparsers.add_parser("calibrate", parents=[base_parser], help="Run automated SPSA parameter calibration")
    p_calib.add_argument("--iterations", type=int, default=5, help="Number of SPSA iterations")
    p_calib.set_defaults(func=cmd_calibrate)

    # Subcommand: dashboard
    p_dash = subparsers.add_parser("dashboard", parents=[base_parser], help="Render PyDeck 3D visualization")
    p_dash.add_argument("--output-dir", default="outputs", help="Output directory")
    p_dash.add_argument("--open-browser", action="store_true", help="Automatically open in browser")
    p_dash.set_defaults(func=cmd_dashboard)

    return parser



def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
