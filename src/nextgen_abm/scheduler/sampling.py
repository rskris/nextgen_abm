"""Metropolis-Hastings MCMC choice set sampler with importance sampling weights."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .milp import DailyScheduleMILP, ActivityDefinition, ScheduledPlan


@dataclass
class SampledAlternative:
    plan: ScheduledPlan
    acceptance_probability: float
    importance_weight: float
    iteration: int


class MetropolisHastingsSampler:
    """Samples competitive daily schedules from the combinatorial space via MCMC."""

    def __init__(self, milp_engine: Optional[DailyScheduleMILP] = None, temperature: float = 5.0):
        self.milp = milp_engine or DailyScheduleMILP()
        self.temperature = temperature

    def sample_choice_set(
        self,
        agent_id: str,
        activities: List[ActivityDefinition],
        allowed_modes: Optional[List[str]] = None,
        n_samples: int = 5,
        burn_in: int = 2
    ) -> List[SampledAlternative]:
        """Draw competitive alternative schedules using Metropolis-Hastings MCMC."""
        # 1. Generate initial base schedule
        current_plan = self.milp.solve_daily_schedule(agent_id, activities, allowed_modes=allowed_modes)
        current_utility = current_plan.total_utility

        sampled_alternatives: List[SampledAlternative] = []
        total_proposals = n_samples + burn_in

        for it in range(total_proposals):
            # 2. Propose perturbation: Draw random utility shocks for activities
            shocks = {
                act.act_id: float(np.random.gumbel(loc=0.0, scale=3.0))
                for act in activities
            }

            # 3. Solve for proposed candidate schedule S'
            candidate_plan = self.milp.solve_daily_schedule(
                agent_id,
                activities,
                allowed_modes=allowed_modes,
                random_utility_shocks=shocks
            )
            candidate_utility = candidate_plan.total_utility

            # 4. Metropolis-Hastings acceptance ratio
            delta_u = candidate_utility - current_utility
            # Accept probability: min(1, exp(delta_u / tau))
            prob_accept = min(1.0, float(np.exp(np.clip(delta_u / self.temperature, -20.0, 20.0))))

            # Accept / Reject step
            if np.random.rand() < prob_accept:
                current_plan = candidate_plan
                current_utility = candidate_utility

            # Collect after burn-in
            if it >= burn_in:
                # Importance sampling weight is proportional to exp(U(S) / tau)
                raw_weight = float(np.exp(np.clip(current_utility / self.temperature, -20.0, 20.0)))
                sampled_alternatives.append(SampledAlternative(
                    plan=current_plan,
                    acceptance_probability=round(prob_accept, 4),
                    importance_weight=round(raw_weight, 4),
                    iteration=it
                ))

        return sampled_alternatives
