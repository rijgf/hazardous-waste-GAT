"""Independent micro-fixture contracts for the v4 parameter audit."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import audit_parameter_revision as audit


def capacity_fixture():
    return SimpleNamespace(
        producers=["G"], facilities=["D1", "D2"], waste_types=["S1", "S2"], periods=[1, 2, 3],
        generation={("G", s, t): q for s, q in [("S1", 10.), ("S2", 6.)] for t in [1, 2, 3]},
        technology={(j, s): int(s == "S1" or j == "D1") for j in ["D1", "D2"] for s in ["S1", "S2"]},
        processing_capacity={(j, s, t): (10. if s == "S1" else 12.)
                             for j in ["D1", "D2"] for s in ["S1", "S2"] for t in [1, 2, 3]},
        initial_producer_inventory={("G", s): 0. for s in ["S1", "S2"]},
        initial_facility_inventory={(j, s): 0. for j in ["D1", "D2"] for s in ["S1", "S2"]})


class ParameterRevisionAuditTests(unittest.TestCase):
    def test_risk_prose_requires_all_four_components_of_the_same_endpoint(self):
        row = {name + "_risk_at_minimum_risk": value for name, value in
               [("transport", 1.), ("coload", 2.), ("producer_inventory", 3.), ("facility_inventory", 4.)]}
        row["minimum_risk"] = 10.
        prose = "运输、共载、产废端库存、设施库存风险分别为1.0000、2.0000、3.0000、4.0000"
        audit.assert_risk_endpoint_prose(prose, row)
        with self.assertRaises(AssertionError):
            audit.assert_risk_endpoint_prose(prose.replace("、4.0000", ""), row)
        bad = dict(row, minimum_risk=9.)
        with self.assertRaises(AssertionError):
            audit.assert_risk_endpoint_prose(prose, bad)

    def test_nsga_diagnostics_separates_attempts_evaluations_and_reference_archives(self):
        records = []
        for repeat, invalid, evaluations, invalid_evaluations, identifiers in (
                (0, 10, 10, 10, ["reference"]), (1, 7, 8, 6, ["reference", "new-a", "new-b"])):
            records.append({"archive_id": f"run-{repeat}", "budget": 10, "instance_id": "instance",
                            "task": {"kind": "NSGA", "instance": "case", "repeat": repeat},
                            "reference_plan_sha256": "reference",
                            "counts": {"candidate_attempts": 10, "invalid_candidate_attempts": invalid,
                                       "candidate_objective_evaluations": evaluations,
                                       "invalid_candidates": invalid_evaluations, "operator_failures": invalid},
                            "points": [{"solution_id": value} for value in identifiers]})
        data = {"case": {"instance_id": "instance", "reference_plan_sha256": "reference"}}
        row = audit.inspect_nsga_diagnostics(records, data)[0]
        self.assertEqual(row["candidate_attempts"], 20)
        self.assertEqual(row["invalid_attempt_rate"], .85)
        self.assertEqual(row["feasible_candidate_attempts"], 3)
        self.assertEqual(row["feasible_objective_evaluations_excluding_preparation"], 2)
        self.assertEqual(row["reference_only_archive_ids"], ["run-0"])
        self.assertEqual(row["nonreference_unique_front_solutions"], 2)
        bad = deepcopy(records)
        bad[0]["points"].append({"solution_id": "unexplained-new-point"})
        with self.assertRaises(AssertionError):
            audit.inspect_nsga_diagnostics(bad, data)

    def test_entry_accepts_direct_and_locked_runner_audit_invocations(self):
        self.assertTrue(audit.parse_arguments(["--require-report"]).require_report)
        self.assertFalse(audit.parse_arguments(["audit"]).require_report)
        self.assertTrue(audit.parse_arguments(["audit", "--allow-partial"]).allow_partial)
        self.assertTrue(audit.parse_arguments(["--preflight-only"]).preflight_only)

    def test_effective_capacity_excludes_technology_forbidden_nominal_cells(self):
        rows = audit.inspect_effective_capacity(capacity_fixture(), expected_factor=2.)
        s2 = next(row for row in rows if row["waste_type"] == "S2")
        self.assertEqual(s2["effective_capacity_each_period"], [12., 12., 12.])
        self.assertEqual(s2["eligible_facilities"], ["D1"])
        self.assertEqual(s2["average_period_generation"], 6.)

    def test_suffix_check_detects_late_waste_that_total_horizon_capacity_hides(self):
        params = capacity_fixture()
        for period, amount in zip(params.periods, [1., 1., 28.]):
            params.generation["G", "S1", period] = amount
        s1 = audit.inspect_effective_capacity(params, 2.)[0]
        self.assertFalse(s1["all_suffix_necessary_conditions_pass"])
        self.assertEqual(s1["suffix_necessary_conditions"][-1]["slack"], -8.)

    def test_seeded_pair_relationship_replays_random_strength_not_fixed_maximum(self):
        original = SimpleNamespace(waste_types=["S1", "S2"], compatibility={("S1", "S2"): 1},
                                   coload_risk={("S1", "S2"): .2}, waste_consequence={"S1": 2., "S2": 4.})
        revised = deepcopy(original)
        revised.coload_risk["S1", "S2"] = 13.599796663725433
        rows = audit.inspect_pair_calibration(original, revised, seed=0)
        self.assertAlmostEqual(rows[0]["equal_load_total_transport_risk_ratio"], 5.533265554575144)
        with self.assertRaises(AssertionError):
            revised.coload_risk["S1", "S2"] = 12.  # exactly 5x would be an unauthorized calibration
            audit.inspect_pair_calibration(original, revised, seed=0)

    def test_observed_capacity_usage_distinguishes_cell_and_system_utilization(self):
        params = capacity_fixture()
        usage = audit.inspect_capacity_usage(params, {("p", "D1", "S1", 1): 10.})
        self.assertEqual(usage["maximum_eligible_cell_utilization"], 1.)
        self.assertEqual(usage["maximum_effective_system_utilization"], .5)
        self.assertEqual(usage["binding_eligible_cells"], 1)
        with self.assertRaises(AssertionError):
            audit.inspect_capacity_usage(params, {("p", "D2", "S2", 1): .1})

    def test_report_evidence_cannot_count_forbidden_capacity_or_force_risk_maximum(self):
        capacities = {"fixture": audit.inspect_effective_capacity(capacity_fixture(), 2.)}
        pairs = {"fixture": [{"equal_load_total_transport_risk_ratio": 5.533265554575144}]}
        evidence = {
            "capacity_by_instance": {"fixture": [
                {"waste": "S1", "average_period_generation": 10., "eligible_facilities": ["D1", "D2"],
                 "effective_capacity_each_period": [20., 20., 20.]},
                {"waste": "S2", "average_period_generation": 6., "eligible_facilities": ["D1"],
                 "effective_capacity_each_period": [12., 12., 12.]}]},
            "risk_by_instance": {"fixture": {"compatible_pairs": 1,
                "observed_total_ratio_range": [5.533265554575144, 5.533265554575144]}},
            "capacity_scenario_factors": [.7, .8, 1., 1.2, 1.3],
            "risk_total_ratio_range": [3., 6.], "risk_strength_range": [2., 5.],
            "risk_scope": "same-arc equal positive two-type loads, transport only; not whole-system risk",
            "models_retrained": True, "training_hyperparameters_changed": False}
        audit.assert_revision_report_evidence(evidence, capacities, pairs)
        changed = deepcopy(evidence)
        changed["capacity_by_instance"]["fixture"][1]["effective_capacity_each_period"] = [24.] * 3
        with self.assertRaises(AssertionError):
            audit.assert_revision_report_evidence(changed, capacities, pairs)
        changed = deepcopy(evidence)
        changed["risk_by_instance"]["fixture"]["observed_total_ratio_range"][1] = 5.
        with self.assertRaises(AssertionError):
            audit.assert_revision_report_evidence(changed, capacities, pairs)


if __name__ == "__main__":
    unittest.main()
