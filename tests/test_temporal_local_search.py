from __future__ import annotations

import unittest

from hazardous_waste_model import pickup_node
from sample_params import build_sample_params
from src.heuristics import build_greedy_initial_plan
from src.operators import OPERATORS, OperatorAction, apply_operator, repair_plan
from src.solution_utils import evaluate_solution, make_route, route_plan_to_solution


class TemporalLocalSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = build_sample_params()
        self.g1s1 = pickup_node("G1", "S1")
        self.g1s2 = pickup_node("G1", "S2")
        self.g2s1 = pickup_node("G2", "S1")
        self.g2s2 = pickup_node("G2", "S2")

    def test_repair_preserves_repeat_service_across_periods(self) -> None:
        plan = {
            ("K1", 1): make_route("D1", [self.g1s1]),
            ("K1", 2): make_route("D1", [self.g1s1, self.g2s1]),
            ("K2", 2): make_route("D1", [self.g1s2, self.g2s2]),
        }

        repaired, ok = repair_plan(self.params, plan, seed=7)

        self.assertTrue(ok)
        self.assertIn(self.g1s1, repaired[("K1", 1)])
        self.assertTrue(
            any(self.g1s1 in route for (vehicle, period), route in repaired.items() if period == 2)
        )
        metrics = evaluate_solution(
            self.params,
            route_plan_to_solution(self.params, repaired),
            (0.0, 1.0),
        )
        self.assertTrue(metrics["feasible"], metrics["violations"])

    def test_add_early_service_keeps_terminal_visit_and_reduces_inventory_risk(self) -> None:
        baseline = build_greedy_initial_plan(self.params, seed=11)
        before = evaluate_solution(
            self.params,
            route_plan_to_solution(self.params, baseline),
            (0.0, 1.0),
        )
        action = OperatorAction(
            operator_id=OPERATORS.index("add_early_service"),
            object_1=0,
            object_2=0,
            object_3=0,
        )

        candidate, ok, reason = apply_operator(self.params, baseline, action, seed=11)

        self.assertTrue(ok, reason)
        task = sorted(self.params.pickup_nodes)[0]
        self.assertTrue(any(task in route for (_, period), route in candidate.items() if period == 1))
        self.assertTrue(any(task in route for (_, period), route in candidate.items() if period == 2))
        after = evaluate_solution(
            self.params,
            route_plan_to_solution(self.params, candidate),
            (0.0, 1.0),
        )
        self.assertTrue(after["feasible"], after["violations"])
        self.assertLess(after["producer_inventory_risk"], before["producer_inventory_risk"])


if __name__ == "__main__":
    unittest.main()
