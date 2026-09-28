"""Space-time candidate opportunity pruning around fixed spatial anchors."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from ..spatial.places import ActivityOpportunity
from ..spatial.routing import MultiScaleRouter


class SpaceTimePruner:
    """Filters millions of candidate building destinations to feasible space-time ellipses."""

    def __init__(self, router: Optional[MultiScaleRouter] = None):
        self.router = router or MultiScaleRouter()

    def prune_candidates(
        self,
        home_coords: Tuple[float, float],
        anchor_coords: Tuple[float, float],
        candidates: List[ActivityOpportunity],
        max_detour_miles: float = 3.5,
        max_results: int = 8
    ) -> List[ActivityOpportunity]:
        """Prune destinations to those lying within the anchor space-time ellipse.
        
        Condition: dist(home, cand) + dist(cand, anchor) <= dist(home, anchor) + max_detour_miles
        """
        direct_dist = self.router.calculate_haversine_distance(home_coords, anchor_coords)
        max_allowed_dist = direct_dist + max_detour_miles

        feasible = []
        for cand in candidates:
            cand_coords = (cand.lon, cand.lat)
            d1 = self.router.calculate_haversine_distance(home_coords, cand_coords)
            d2 = self.router.calculate_haversine_distance(cand_coords, anchor_coords)
            total_detour = (d1 + d2) - direct_dist

            if (d1 + d2) <= max_allowed_dist:
                feasible.append((total_detour, cand))

        # Sort by smallest detour and truncate to max_results to keep MILP compact
        feasible.sort(key=lambda x: x[0])
        return [cand for _, cand in feasible[:max_results]]
