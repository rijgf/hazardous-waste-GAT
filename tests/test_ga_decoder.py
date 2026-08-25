from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from itertools import product
from pathlib import Path

from sample_params import build_sample_params
from src.solution_utils import evaluate_solution


ROOT = Path(__file__).resolve().parents[1]
GA_DIR = ROOT / "遗传算法"
if str(GA_DIR) not in sys.path:
    sys.path.insert(0, str(GA_DIR))

from genetic_algorithm import ClassicGeneticAlgorithm  # noqa: E402


class GADecoderTests(unittest.TestCase):
    def test_decoder_uses_multiple_facilities_when_technology_is_split(self) -> None:
        base = build_sample_params()
        facilities = ["D1", "D2"]
        distance = dict(base.distance)
        accident_probability = dict(base.accident_probability)
        for node in base.pickup_nodes:
            distance["D2", node] = distance["D1", node] + 0.5
            distance[node, "D2"] = distance[node, "D1"] + 0.5
            accident_probability["D2", node] = accident_probability["D1", node]
            accident_probability[node, "D2"] = accident_probability[node, "D1"]

        params = replace(
            base,
            facilities=facilities,
            distance=distance,
            accident_probability=accident_probability,
            facility_capacity={"D1": 30.0, "D2": 30.0},
            initial_facility_inventory={
                (facility, waste): 0.0
                for facility, waste in product(facilities, base.waste_types)
            },
            processing_cost={
                (facility, waste): 8.0
                for facility, waste in product(facilities, base.waste_types)
            },
            technology={
                ("D1", "S1"): 1,
                ("D1", "S2"): 0,
                ("D2", "S1"): 0,
                ("D2", "S2"): 1,
            },
            processing_capacity={
                (facility, waste, period): 20.0
                for facility, waste, period in product(
                    facilities,
                    base.waste_types,
                    base.periods,
                )
            },
            facility_inventory_risk={
                (facility, waste): 0.2
                for facility, waste in product(facilities, base.waste_types)
            },
        )

        solution = ClassicGeneticAlgorithm(params).decode(list(params.pickup_nodes))
        metrics = evaluate_solution(params, solution, (0.5, 0.5))

        self.assertTrue(metrics["feasible"], metrics["violations"])
        used_facilities = {
            route[0]
            for route in solution["plan"].values()
        }
        self.assertEqual({"D1", "D2"}, used_facilities)


if __name__ == "__main__":
    unittest.main()
