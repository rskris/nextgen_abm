"""Hierarchical Multi-Modal Network Representation for Link Transmission Model (LTM).

Models freeways, arterials, MTD transit corridors, and UCSB Class-I bike facilities
as discrete topological links and nodes, snapping building footprints via centroid connectors.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from ..config import MasterConfig, get_config


@dataclass
class LTMNode:
    """Network junction, ramp, bike roundabout, or transit station."""
    node_id: str
    x: float  # Longitude (EPSG:4326)
    y: float  # Latitude (EPSG:4326)
    elevation_m: float = 10.0
    node_type: str = "intersection"  # "freeway_ramp", "arterial_intersection", "bike_roundabout", "transit_stop", "centroid"


@dataclass
class LTMLink:
    """Discrete facility cell for kinematic wave traffic transmission."""
    link_id: str
    from_node: str
    to_node: str
    length_km: float
    modes: List[str]  # e.g. ["car", "transit"], ["bike"], ["walk"]
    num_lanes: int = 2
    v_free_kmh: float = 60.0
    w_wave_kmh: float = 20.0
    k_jam_per_km: float = 240.0      # Jam density across all lanes
    q_max_vph: float = 3600.0        # Capacity flow across all lanes
    grade_pct: float = 0.0           # Slope percentage (elevation rise / run)
    name: str = ""

    @property
    def free_flow_time_hours(self) -> float:
        """Free-flow traversal time in hours."""
        return self.length_km / max(self.v_free_kmh, 1.0)

    @property
    def free_flow_time_seconds(self) -> float:
        """Free-flow traversal time in seconds."""
        return self.free_flow_time_hours * 3600.0

    @property
    def wave_travel_time_seconds(self) -> float:
        """Time for backward shockwave to traverse the link upstream in seconds."""
        return (self.length_km / max(self.w_wave_kmh, 1.0)) * 3600.0

    @property
    def storage_capacity_vehicles(self) -> float:
        """Maximum number of vehicles physically storable in jam condition."""
        return self.k_jam_per_km * self.length_km


class HierarchicalMultiModalNetwork:
    """Multi-modal network topology with mode filtering and centroid snapping."""

    def __init__(self, config: Optional[MasterConfig] = None):
        self.config = config or get_config()
        self.nodes: Dict[str, LTMNode] = {}
        self.links: Dict[str, LTMLink] = {}
        self.outgoing_links: Dict[str, List[LTMLink]] = {}
        self.incoming_links: Dict[str, List[LTMLink]] = {}
        self.building_connectors: Dict[str, str] = {}  # bldg_id -> nearest node_id

    def add_node(self, node: LTMNode) -> None:
        """Add node to network graph."""
        self.nodes[node.node_id] = node
        if node.node_id not in self.outgoing_links:
            self.outgoing_links[node.node_id] = []
        if node.node_id not in self.incoming_links:
            self.incoming_links[node.node_id] = []

    def add_link(self, link: LTMLink) -> None:
        """Add directional link to network graph."""
        self.links[link.link_id] = link
        if link.from_node not in self.outgoing_links:
            self.outgoing_links[link.from_node] = []
        self.outgoing_links[link.from_node].append(link)

        if link.to_node not in self.incoming_links:
            self.incoming_links[link.to_node] = []
        self.incoming_links[link.to_node].append(link)

    def snap_building(self, bldg_id: str, coords: Tuple[float, float], allowed_mode: str = "car") -> str:
        """Snap a building coordinate (lon, lat) to the nearest accessible network node."""
        if bldg_id in self.building_connectors:
            return self.building_connectors[bldg_id]

        bx, by = coords
        best_node = None
        best_dist = float("inf")

        for node_id, node in self.nodes.items():
            # Check if this node has links supporting allowed_mode
            out_has_mode = any(allowed_mode in l.modes for l in self.outgoing_links.get(node_id, []))
            in_has_mode = any(allowed_mode in l.modes for l in self.incoming_links.get(node_id, []))
            if not (out_has_mode or in_has_mode):
                continue

            dist = (node.x - bx) ** 2 + (node.y - by) ** 2
            if dist < best_dist:
                best_dist = dist
                best_node = node_id

        if best_node is None:
            best_node = next(iter(self.nodes.keys()))

        self.building_connectors[bldg_id] = best_node
        return best_node

    def find_shortest_path(self, origin_node: str, dest_node: str, mode: str = "car") -> List[str]:
        """Compute shortest link path using Dijkstra's algorithm filtered by mode."""
        if origin_node == dest_node:
            return []

        distances: Dict[str, float] = {origin_node: 0.0}
        previous_link: Dict[str, str] = {}
        previous_node: Dict[str, str] = {}
        pq: List[Tuple[float, str]] = [(0.0, origin_node)]

        while pq:
            curr_dist, curr_node = heapq.heappop(pq)
            if curr_dist > distances.get(curr_node, float("inf")):
                continue
            if curr_node == dest_node:
                break

            for link in self.outgoing_links.get(curr_node, []):
                if mode not in link.modes:
                    continue
                nxt_node = link.to_node
                cost = curr_dist + link.free_flow_time_seconds
                if cost < distances.get(nxt_node, float("inf")):
                    distances[nxt_node] = cost
                    previous_link[nxt_node] = link.link_id
                    previous_node[nxt_node] = curr_node
                    heapq.heappush(pq, (cost, nxt_node))

        if dest_node not in previous_link:
            # Fallback direct virtual link if disconnected
            return []

        # Reconstruct link sequence
        path = []
        curr = dest_node
        while curr != origin_node:
            link_id = previous_link[curr]
            path.append(link_id)
            curr = previous_node[curr]
        path.reverse()
        return path

    @classmethod
    def build_santa_barbara_regional_network(cls, config: Optional[MasterConfig] = None) -> HierarchicalMultiModalNetwork:
        """Construct full Santa Barbara multi-modal corridor and UCSB bike network."""
        net = cls(config=config)
        c = net.config

        # 1. Key Regional Nodes (Corridors: US-101, SR-217, SR-154, Arterials)
        nodes = [
            # US-101 Corridor
            LTMNode("node_us101_carpinteria", -119.52, 34.39, 5.0, "freeway_ramp"),
            LTMNode("node_us101_montecito", -119.64, 34.42, 20.0, "freeway_ramp"),
            LTMNode("node_us101_sb_salinas", -119.67, 34.42, 15.0, "freeway_ramp"),
            LTMNode("node_us101_sb_carrillo", -119.70, 34.42, 12.0, "freeway_ramp"),
            LTMNode("node_us101_sb_state", -119.73, 34.44, 25.0, "freeway_ramp"),
            LTMNode("node_us101_goleta_fairview", -119.82, 34.44, 15.0, "freeway_ramp"),
            LTMNode("node_us101_goleta_storke", -119.87, 34.43, 10.0, "freeway_ramp"),
            LTMNode("node_us101_winchester", -119.92, 34.44, 8.0, "freeway_ramp"),

            # SR-217 UCSB Highway
            LTMNode("node_sr217_junction", -119.83, 34.44, 12.0, "freeway_ramp"),
            LTMNode("node_ucsb_henley_gate", -119.84, 34.41, 10.0, "freeway_ramp"),

            # SR-154 San Marcos Pass
            LTMNode("node_sr154_junction", -119.74, 34.45, 45.0, "arterial_intersection"),
            LTMNode("node_sr154_summit", -119.79, 34.50, 680.0, "arterial_intersection"),

            # South Coast Arterials
            LTMNode("node_sb_transit_center", -119.70, 34.42, 15.0, "transit_stop"),
            LTMNode("node_state_st_midtown", -119.72, 34.43, 20.0, "arterial_intersection"),
            LTMNode("node_hollister_fairview", -119.82, 34.43, 12.0, "arterial_intersection"),
            LTMNode("node_hollister_storke", -119.87, 34.42, 10.0, "arterial_intersection"),
            LTMNode("node_calle_real_goleta", -119.85, 34.45, 30.0, "arterial_intersection"),

            # UCSB Campus & Isla Vista Bike Roundabouts
            LTMNode("node_iv_pardall_tunnel", -119.86, 34.41, 8.0, "bike_roundabout"),
            LTMNode("node_ucsb_library_circle", -119.85, 34.41, 12.0, "bike_roundabout"),
            LTMNode("node_ucsb_ocean_rd", -119.85, 34.42, 10.0, "bike_roundabout"),
            LTMNode("node_ucsb_north_hall", -119.84, 34.41, 11.0, "transit_stop"),
        ]

        for n in nodes:
            net.add_node(n)

        # 2. Highway Links (US-101, SR-217, SR-154)
        def add_bidirectional_link(lid, n1, n2, length, lanes, v_free, q_max, modes, grade=0.0, name=""):
            net.add_link(LTMLink(
                link_id=f"{lid}_NB",
                from_node=n1,
                to_node=n2,
                length_km=length,
                modes=modes,
                num_lanes=lanes,
                v_free_kmh=v_free,
                w_wave_kmh=c.ltm.wave_speed_kmh,
                k_jam_per_km=c.ltm.jam_density_veh_km * lanes,
                q_max_vph=q_max * lanes,
                grade_pct=grade,
                name=f"{name} (NB/WB)",
            ))
            net.add_link(LTMLink(
                link_id=f"{lid}_SB",
                from_node=n2,
                to_node=n1,
                length_km=length,
                modes=modes,
                num_lanes=lanes,
                v_free_kmh=v_free,
                w_wave_kmh=c.ltm.wave_speed_kmh,
                k_jam_per_km=c.ltm.jam_density_veh_km * lanes,
                q_max_vph=q_max * lanes,
                grade_pct=-grade,
                name=f"{name} (SB/EB)",
            ))

        # US-101 Segments
        add_bidirectional_link("link_us101_carp_mont", "node_us101_carpinteria", "node_us101_montecito", 11.0, 3, c.ltm.v_free_freeway_kmh, c.ltm.corridor_capacity_freeway_vph, ["car", "transit"], 0.2, "US-101 Montecito Widening")
        add_bidirectional_link("link_us101_mont_salinas", "node_us101_montecito", "node_us101_sb_salinas", 4.0, 2, c.ltm.v_free_freeway_kmh, c.ltm.corridor_capacity_freeway_vph, ["car", "transit"], 0.0, "US-101 Salinas St")
        add_bidirectional_link("link_us101_salinas_carrillo", "node_us101_sb_salinas", "node_us101_sb_carrillo", 3.0, 2, c.ltm.v_free_freeway_kmh, c.ltm.corridor_capacity_freeway_vph, ["car", "transit"], 0.0, "US-101 Carrillo St")
        add_bidirectional_link("link_us101_carrillo_state", "node_us101_sb_carrillo", "node_us101_sb_state", 4.0, 3, c.ltm.v_free_freeway_kmh, c.ltm.corridor_capacity_freeway_vph, ["car", "transit"], 0.3, "US-101 State St")
        add_bidirectional_link("link_us101_state_fairview", "node_us101_sb_state", "node_us101_goleta_fairview", 8.0, 3, c.ltm.v_free_freeway_kmh, c.ltm.corridor_capacity_freeway_vph, ["car", "transit"], 0.0, "US-101 Fairview Ave")
        add_bidirectional_link("link_us101_fairview_storke", "node_us101_goleta_fairview", "node_us101_goleta_storke", 5.0, 2, c.ltm.v_free_freeway_kmh, c.ltm.corridor_capacity_freeway_vph, ["car", "transit"], 0.0, "US-101 Storke Rd")
        add_bidirectional_link("link_us101_storke_winch", "node_us101_goleta_storke", "node_us101_winchester", 5.0, 2, c.ltm.v_free_freeway_kmh, c.ltm.corridor_capacity_freeway_vph, ["car", "transit"], 0.0, "US-101 Winchester")

        # SR-217 Ward Memorial to UCSB
        add_bidirectional_link("link_sr217_highway", "node_sr217_junction", "node_ucsb_henley_gate", 4.0, 2, 90.0, 1800.0, ["car", "transit"], 0.0, "SR-217 Ward Memorial")
        add_bidirectional_link("link_sr217_conn", "node_us101_goleta_fairview", "node_sr217_junction", 1.5, 2, 70.0, 1500.0, ["car", "transit"], 0.0, "SR-217 Junction Ramp")

        # Arterials
        add_bidirectional_link("link_art_hollister", "node_hollister_fairview", "node_hollister_storke", 5.0, 2, c.ltm.v_free_arterial_kmh, c.ltm.corridor_capacity_arterial_vph, ["car", "transit", "bike"], 0.0, "Hollister Ave")
        add_bidirectional_link("link_art_state", "node_sb_transit_center", "node_state_st_midtown", 3.0, 2, 45.0, c.ltm.corridor_capacity_arterial_vph, ["car", "transit", "bike"], 0.5, "State St Downtown")

        # Arterial-Freeway Interchanges & Ramp Connectors
        add_bidirectional_link("link_conn_fairview", "node_hollister_fairview", "node_us101_goleta_fairview", 0.8, 2, 45.0, 1500.0, ["car", "transit", "bike"], 0.0, "Fairview Ave Connector")
        add_bidirectional_link("link_conn_storke", "node_hollister_storke", "node_us101_goleta_storke", 0.8, 2, 45.0, 1500.0, ["car", "transit", "bike"], 0.0, "Storke Rd Connector")
        add_bidirectional_link("link_conn_carrillo", "node_sb_transit_center", "node_us101_sb_carrillo", 0.5, 2, 40.0, 1200.0, ["car", "transit", "bike", "walk"], 0.0, "Carrillo St Connector")
        add_bidirectional_link("link_conn_state_101", "node_state_st_midtown", "node_us101_sb_state", 0.5, 2, 45.0, 1200.0, ["car", "transit", "bike"], 0.0, "State St Freeway Connector")
        add_bidirectional_link("link_conn_salinas", "node_sb_transit_center", "node_us101_sb_salinas", 1.0, 2, 45.0, 1200.0, ["car", "transit", "bike"], 0.0, "Salinas St Connector")


        # UCSB Class-I Dedicated Bicycle Network
        add_bidirectional_link("link_bike_pardall", "node_iv_pardall_tunnel", "node_ucsb_library_circle", 1.0, 2, c.ltm.v_free_bike_kmh, c.ltm.bike_path_capacity_bph, ["bike", "walk"], 0.0, "Pardall Bike Path")
        add_bidirectional_link("link_bike_library_north", "node_ucsb_library_circle", "node_ucsb_north_hall", 0.6, 2, c.ltm.v_free_bike_kmh, c.ltm.bike_path_capacity_bph, ["bike", "walk"], 0.0, "Library Circle to North Hall")
        add_bidirectional_link("link_bike_ocean_rd", "node_ucsb_ocean_rd", "node_ucsb_library_circle", 0.8, 2, c.ltm.v_free_bike_kmh, c.ltm.bike_path_capacity_bph, ["bike", "walk"], 0.0, "Ocean Road Bikeway")
        add_bidirectional_link("link_bike_henley_north", "node_ucsb_henley_gate", "node_ucsb_north_hall", 0.7, 2, c.ltm.v_free_bike_kmh, c.ltm.bike_path_capacity_bph, ["bike", "walk"], 0.0, "Henley Gate Cycleway")

        # MTD Bus Lines Connectors (Line 24x Express between SB Transit Center and UCSB North Hall)
        add_bidirectional_link("link_mtd_24x_express", "node_sb_transit_center", "node_ucsb_north_hall", 16.0, 1, 80.0, 1000.0, ["transit"], 0.0, "MTD Line 24x UCSB Express")

        return net
