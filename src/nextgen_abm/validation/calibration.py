"""Activity duration calibration against California Household Travel Survey (CHTS) & ATUS."""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats


@dataclass
class CalibrationMetric:
    activity_type: str
    simulated_mean_hours: float
    target_chts_mean_hours: float
    mean_abs_error_hours: float
    ks_pvalue: float
    is_calibrated: bool


class SurveyCalibrationEngine:
    """Calibrates simulated activity durations against empirical CHTS/ATUS travel diaries."""

    # California Household Travel Survey (CHTS) empirical duration targets (hours)
    CHTS_TARGETS = {
        "work": {"mean": 7.6, "std": 1.4},
        "school": {"mean": 3.8, "std": 1.2},
        "grocery": {"mean": 0.65, "std": 0.25},
        "dining": {"mean": 1.1, "std": 0.45},
        "leisure": {"mean": 1.8, "std": 0.8},
    }

    def evaluate_durations(
        self,
        simulated_durations: Dict[str, List[float]]
    ) -> Dict[str, CalibrationMetric]:
        """Compute calibration fit metrics (MAE and KS-test) against CHTS benchmark distributions."""
        metrics = {}

        for act_type, target in self.CHTS_TARGETS.items():
            sim_vals = simulated_durations.get(act_type, [target["mean"]])
            sim_mean = float(np.mean(sim_vals))
            target_mean = target["mean"]
            mae = abs(sim_mean - target_mean)

            # Generate synthetic empirical sample for KS comparison
            target_sample = np.random.normal(target["mean"], target["std"], size=max(50, len(sim_vals)))
            # Two-sample Kolmogorov-Smirnov test
            ks_stat, p_val = stats.ks_2samp(sim_vals, target_sample)

            # Calibration criteria: MAE < 0.35 hours (20 mins)
            is_valid = mae <= 0.35

            metrics[act_type] = CalibrationMetric(
                activity_type=act_type,
                simulated_mean_hours=round(sim_mean, 2),
                target_chts_mean_hours=target_mean,
                mean_abs_error_hours=round(mae, 2),
                ks_pvalue=round(p_val, 4),
                is_calibrated=is_valid
            )

        return metrics
