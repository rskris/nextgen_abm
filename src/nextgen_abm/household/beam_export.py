"""BEAM (LBNL) / MATSim integration pipeline exporting activity plans and EV fleet profiles."""

from pathlib import Path
from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pandas as pd
from ..scheduler.milp import ScheduledPlan


class BEAMExporter:
    """Exports activity schedules and vehicle fleet profiles to BEAM / MATSim standard formats."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("data/beam_inputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_plans_xml(self, plans: List[ScheduledPlan], filename: str = "plans.xml") -> Path:
        """Export a list of scheduled plans to MATSim/BEAM standard plans.xml format."""
        root = ET.Element("plans")

        for plan in plans:
            person_elem = ET.SubElement(root, "person", id=str(plan.agent_id))
            plan_elem = ET.SubElement(person_elem, "plan", selected="yes")

            for i, act in enumerate(plan.activities):
                # Activity node
                h = int(act.end_hour)
                m = int((act.end_hour - h) * 60)
                s = int((((act.end_hour - h) * 60) - m) * 60)
                end_time_str = f"{h:02d}:{m:02d}:{s:02d}"

                ET.SubElement(
                    plan_elem,
                    "activity",
                    type=act.act_type,
                    x=f"{act.chosen_coords[0]:.6f}",
                    y=f"{act.chosen_coords[1]:.6f}",
                    end_time=end_time_str
                )

                # Leg node connecting to next activity
                if i < len(plan.legs):
                    leg = plan.legs[i]
                    ET.SubElement(
                        plan_elem,
                        "leg",
                        mode=leg.chosen_mode,
                        dep_time=f"{int(leg.departure_hour):02d}:{int((leg.departure_hour % 1)*60):02d}:00"
                    )

        # Pretty print XML
        raw_xml = ET.tostring(root, encoding="utf-8")
        parsed = minidom.parseString(raw_xml)
        pretty_xml = parsed.toprettyxml(indent="  ")

        out_path = self.output_dir / filename
        out_path.write_text(pretty_xml, encoding="utf-8")
        return out_path

    def export_beam_vehicles_csv(
        self,
        vehicles: List[Dict[str, Any]],
        filename: str = "beamVehicles.csv"
    ) -> Path:
        """Export vehicle fleet definitions including EV battery parameters for BEAM simulation."""
        rows = []
        for v in vehicles:
            is_ev = v.get("vehicle_type") == "ev"
            rows.append({
                "vehicleId": v["vehicle_id"],
                "vehicleTypeId": "BEV_Standard" if is_ev else "ICE_Midsize",
                "nominalBatteryCapacityInKWh": 65.0 if is_ev else 0.0,
                "initialSoc": v.get("initial_soc_pct", 85.0) / 100.0 if is_ev else 0.0,
                "householdId": v.get("household_id", "")
            })

        df = pd.DataFrame(rows)
        out_path = self.output_dir / filename
        df.to_csv(out_path, index=False)
        return out_path
