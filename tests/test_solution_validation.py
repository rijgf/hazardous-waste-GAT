from __future__ import annotations

import copy
import unittest

from hazardous_waste_model import solve_with_milp
from sample_params import build_sample_params
from src.heuristics import build_greedy_initial_solution
from src.solution_utils import evaluate_solution, route_plan_to_solution


class SolutionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.params = build_sample_params()

    def test_empty_solution_is_not_feasible(self) -> None:
        metrics = evaluate_solution(self.params, {}, (0.5, 0.5))

        self.assertFalse(metrics["feasible"])
        self.assertTrue(any("missing" in item for item in metrics["violations"]))

    def test_inventory_conservation_corruption_is_reported(self) -> None:
        solution = build_greedy_initial_solution(self.params, seed=3)
        corrupted = copy.deepcopy(solution)
        node = self.params.pickup_nodes[0]
        corrupted["raw"]["IG", node, self.params.periods[0]] += 1.0

        metrics = evaluate_solution(self.params, corrupted, (0.5, 0.5))

        self.assertFalse(metrics["feasible"])
        self.assertTrue(any("producer inventory balance" in item for item in metrics["violations"]))

    def test_generated_plan_remains_feasible(self) -> None:
        solution = build_greedy_initial_solution(self.params, seed=3)

        metrics = evaluate_solution(self.params, solution, (0.5, 0.5))

        self.assertTrue(metrics["feasible"], metrics["violations"])

    def test_broken_route_flow_is_reported(self) -> None:
        solution = build_greedy_initial_solution(self.params, seed=3)
        corrupted = copy.deepcopy(solution)
        return_arc = next(
            key
            for key, value in corrupted["raw"].items()
            if key[0] == "x"
            and value > 0.5
            and key[2] in self.params.facilities
        )
        del corrupted["raw"][return_arc]

        metrics = evaluate_solution(self.params, corrupted, (0.5, 0.5))

        self.assertFalse(metrics["feasible"])
        self.assertTrue(
            any("route flow conservation" in item for item in metrics["violations"]),
            metrics["violations"],
        )

    def test_disconnected_pickup_cycle_is_reported(self) -> None:
        solution = build_greedy_initial_solution(self.params, seed=3)
        corrupted = copy.deepcopy(solution)
        (vehicle, period), route = next(iter(solution["plan"].items()))
        facility = route[0]
        first_pickup = route[1]
        last_pickup = route[-2]
        del corrupted["raw"]["x", facility, first_pickup, vehicle, period]
        del corrupted["raw"]["x", last_pickup, facility, vehicle, period]
        corrupted["raw"]["x", last_pickup, first_pickup, vehicle, period] = 1.0

        metrics = evaluate_solution(self.params, corrupted, (0.5, 0.5))

        self.assertFalse(metrics["feasible"])
        self.assertTrue(
            any("disconnected" in item for item in metrics["violations"]),
            metrics["violations"],
        )

    def test_missing_arc_load_flow_is_reported(self) -> None:
        solution = build_greedy_initial_solution(self.params, seed=3)
        corrupted = copy.deepcopy(solution)
        for key in [key for key in corrupted["raw"] if key[0] == "F"]:
            del corrupted["raw"][key]

        metrics = evaluate_solution(self.params, corrupted, (0.5, 0.5))

        self.assertFalse(metrics["feasible"])
        self.assertTrue(
            any("facility receipt flow" in item for item in metrics["violations"]),
            metrics["violations"],
        )

    def test_milp_solution_exposes_a_replayable_plan(self) -> None:
        result = solve_with_milp(self.params, time_limit=10.0)

        self.assertTrue(result.solution, result.diagnostics)
        self.assertIn("plan", result.solution)
        replayed = evaluate_solution(
            self.params,
            route_plan_to_solution(self.params, result.solution["plan"]),
            (self.params.cost_weight, self.params.risk_weight),
        )
        original = evaluate_solution(
            self.params,
            result.solution,
            (self.params.cost_weight, self.params.risk_weight),
        )
        self.assertTrue(replayed["feasible"], replayed["violations"])
        self.assertAlmostEqual(original["cost"], replayed["cost"], places=6)
        self.assertAlmostEqual(original["risk"], replayed["risk"], places=6)


if __name__ == "__main__":
    unittest.main()
