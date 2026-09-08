"""Independent, read-only numerical replay of the single-instance experiment.

No search, training, repair, production-artifact edits, or automatic recovery is
performed. Only the audit JSON is written. --allow-partial reports INCOMPLETE;
it never turns missing delivery evidence into PASS.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import xml.etree.ElementTree as ET

import numpy as np

from hazardous_waste_model import HazardousWasteMILP, pickup_type
from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan, json_sha256, plan_sha256
from src.solution_utils import evaluate_solution, route_plan_to_solution
from src.supplementary_experiment import deserialize_solution

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/pareto-single-instance-v3"
OBJECTIVE_FIELDS = ("cost", "risk", "normalized_cost", "normalized_risk",
                    "weighted_objective_raw", "weighted_objective_normalized",
                    "weighted_objective", "fixed_cost", "distance_cost",
                    "processing_cost", "transport_risk", "coload_risk",
                    "producer_inventory_risk", "facility_inventory_risk")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b, label, atol=1e-8, rtol=1e-10):
    if not math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol):
        raise AssertionError(f"{label}: recomputed={a!r}, recorded={b!r}")


def assert_manuscript_row(text, values, label, *, first_cell_aliases=()):
    """Require every expected cell, in order, in an actual manuscript table."""
    expected = [str(value).strip() for value in values]
    rows = [line.strip().split("|")[1:-1] for line in text.splitlines()
            if line.strip().startswith("|") and line.strip().endswith("|")]
    rows += [re.findall(r"<td(?:\s[^>]*)?>(.*?)</td>", row, re.DOTALL)
             for row in re.findall(r"<tr(?:\s[^>]*)?>(.*?)</tr>", text, re.DOTALL)]
    for row in rows:
        cells = [cell.strip() for cell in row]
        if cells and cells[0] in first_cell_aliases:
            cells[0] = expected[0]
        if cells == expected:
            return
    raise AssertionError(label)


def assert_checkpoint_row(text, values, model_label, checkpoint_sha, *, checkpoint_alias=None):
    """Accept a relocated full SHA only when its model label stays explicit."""
    try:
        assert_manuscript_row(text, [*values, checkpoint_sha], "legacy checkpoint " + model_label)
        return
    except AssertionError:
        assert_manuscript_row(text, [*values, checkpoint_alias] if checkpoint_alias else values,
                              "checkpoint training fields " + model_label)
    associated = re.search(re.escape(model_label) + r"(?:（New-[SL]）)?\s*(?:完整)?SHA256\s*[:：]\s*`?"
                           + re.escape(checkpoint_sha) + r"`?(?![0-9a-f])", text)
    assert associated, (
        "checkpoint SHA association " + model_label)


def terminal_max(params, solution):
    raw, last = solution["raw"], params.periods[-1]
    amounts = [raw.get(("IG", node, last), 0.) for node in params.pickup_nodes]
    amounts += [raw.get(("ID", facility, waste, last), 0.)
                for facility in params.facilities for waste in params.waste_types]
    return max(map(abs, amounts), default=0.)


def independent_pair(a, b, refs, tolerance):
    """Direct pairwise weak coverage; no production Pareto helpers are used."""
    av = np.array([[p["cost"] / refs[0], p["risk"] / refs[1]] for p in a])
    bv = np.array([[p["cost"] / refs[0], p["risk"] / refs[1]] for p in b])
    if not len(a) or not len(b):
        return {"n_A": len(a), "n_B": len(b), "status": "failed_empty_front",
                "A_covers_B_count": None, "B_covers_A_count": None,
                "A_covers_B": None, "B_covers_A": None}
    ab = int(np.any(np.all(av[:, None, :] - bv[None, :, :] <= tolerance, axis=2), axis=0).sum())
    ba = int(np.any(np.all(bv[:, None, :] - av[None, :, :] <= tolerance, axis=2), axis=0).sum())
    return {"n_A": len(a), "n_B": len(b), "status": "valid",
            "A_covers_B_count": ab, "B_covers_A_count": ba,
            "A_covers_B": ab / len(b), "B_covers_A": ba / len(a)}


def assert_front(points, refs, tolerance):
    ids = [p["solution_id"] for p in points]
    assert len(set(ids)) == len(ids), "repeated solution identity"
    values = np.array([[p["cost"] / refs[0], p["risk"] / refs[1]] for p in points])
    assert len(values), "empty formal front"
    assert np.isfinite(values).all(), "non-finite objective"
    weak = (values[:, None, :] - values[None, :, :] <= tolerance).all(axis=2)
    equal, strict = weak & weak.T, weak & ~weak.T
    np.fill_diagonal(equal, False)
    assert not equal.any(), "duplicate objectives under locked tolerance"
    assert not strict.any(), "strictly dominated delivered point"


class Auditor:
    def __init__(self):
        self.errors, self.missing, self.verified = [], [], Counter()
        self.protocol = read(OUT / "protocol.json")
        self.protocol_sha = sha(OUT / "protocol.json")
        self.instances, self.fronts = {}, {}
        self.pair_rows, self.milp_rows, self.small_rows = [], [], []
        self.plan_cache = {}
        self.capacity_rows, self.table_checks = {}, []
        self.global_capacity_certificate = None

    def item(self, label, function):
        try:
            function()
        except FileNotFoundError as exc:
            self.missing.append({"item": label, "detail": str(exc)})
        except Exception as exc:
            self.errors.append({"item": label, "detail": f"{type(exc).__name__}: {exc}"})

    def identities(self):
        registry = read(ROOT / "configs/frozen_ppo_models.json")
        assert sha(ROOT / "configs/frozen_ppo_models.json") == self.protocol["registry_sha256"]
        assert registry == self.protocol["registry"]
        for name, model in registry["models"].items():
            assert sha(ROOT / model["checkpoint"]) == model["sha256"], name
            training = read(ROOT / model["training_record"])
            assert training["checkpoint_sha256"] == model["sha256"]
            assert len(training["instance_episode_counts"]) == 24
            assert set(training["instance_episode_counts"].values()) == {20}
        manifest_path = ROOT / registry["dataset_manifest"]
        assert sha(manifest_path) == registry["dataset_manifest_sha256"]
        manifest = read(manifest_path)
        train_hashes, test_hashes, train_seeds, test_seeds = set(), set(), set(), set()
        for record in manifest["records"]:
            path = manifest_path.parent / record["path"]
            assert sha(path) == record["sha256"]
            data = read(path)
            assert json_sha256(data["params"]) == data["instance_sha256"]
            hashes, seeds = ((train_hashes, train_seeds) if data["split"] == "train"
                             else (test_hashes, test_seeds))
            hashes.add(data["instance_sha256"])
            seeds.add(data["instance_seed"])
        assert len(train_hashes) == 48 and len(test_hashes) == 40
        assert not train_hashes & test_hashes and not train_seeds & test_seeds
        for file, expected in self.protocol["sources"].items():
            assert sha(ROOT / file) == expected, f"source changed: {file}"
        preflight = read(OUT / "preflight.json")
        assert preflight["passed"] and preflight["protocol_sha256"] == self.protocol_sha
        assert len(self.protocol["tasks"]) == 60
        assert Counter(t["kind"] for t in self.protocol["tasks"]) == {"PPO": 48, "NSGA": 12}
        assert len(self.protocol["preferences"]) == 21
        assert self.protocol["ppo_steps"] == 60 and self.protocol["ppo_candidates_per_step"] == 32
        assert self.protocol["repeats"] == 3 and self.protocol["test_instances_per_scale"] == 1
        assert self.protocol["milp_time_limit"] == 3600
        self.verified["frozen_checkpoints"] = 2
        self.verified["locked_train_and_candidate_test_instances"] = 88

    def instance(self, name):
        path = OUT / f"instances/{name}.json"
        assert sha(path) == self.protocol["instances"][name]
        data = read(path)
        assert json_sha256(data["params"]) == data["instance_sha256"]
        params, plan = params_from_json_data(data["params"]), deserialize_plan(data["reference_plan"])
        assert plan_sha256(plan) == data["reference_plan_sha256"]
        refs = (data["b_C"], data["b_R"])
        assert all(math.isfinite(x) and x > 0 for x in refs)
        solution = route_plan_to_solution(params, plan)
        metrics = evaluate_solution(params, solution, (.5, .5), objective_refs=refs)
        assert metrics["feasible"] and terminal_max(params, solution) <= 1e-5
        close(metrics["cost"], refs[0], name + " b_C")
        close(metrics["risk"], refs[1], name + " b_R")
        if name.startswith(("capacity-", "coload-")):
            base = read(OUT / "instances/Test-4.json")
            field = data["scenario_parameter"]
            differences = [k for k, value in base["params"].items() if value != data["params"][k]]
            assert differences == [field], differences
            for before, after in zip(base["params"][field], data["params"][field]):
                assert before["key"] == after["key"]
                close(after["value"], before["value"] * data["scenario_multiplier"], name + " multiplier")
        self.instances[name] = (data, params, refs)
        self.verified["bound_instance_versions"] += 1

    def plan(self, name, point):
        data, params, refs = self.instances[name]
        key = (name, point["solution_id"])
        if key not in self.plan_cache:
            record = read(OUT / point["solution_path"])
            assert record["instance_id"] == data["instance_id"]
            assert record["instance_sha256"] == data["instance_sha256"]
            assert (record["b_C"], record["b_R"]) == refs
            plan = deserialize_plan(record["plan"])
            assert plan_sha256(plan) == record["solution_id"] == point["solution_id"]
            solution = route_plan_to_solution(params, plan)
            metrics = evaluate_solution(params, solution, (.5, .5), objective_refs=refs)
            assert metrics["feasible"], metrics["violations"]
            assert terminal_max(params, solution) <= 1e-5
            self.plan_cache[key] = metrics
            capacity_cells = [(solution["raw"].get(("p", j, s, t), 0.), cap)
                              for (j, s, t), cap in params.processing_capacity.items()
                              if params.technology[j, s] and cap > 0]
            utilization = max((used / cap for used, cap in capacity_cells), default=0.)
            slack = min((cap - used for used, cap in capacity_cells), default=math.inf)
            summary = self.capacity_rows.setdefault(name, {"instance": name, "unique_plans": 0,
                         "maximum_processing_utilization": 0., "minimum_processing_slack": math.inf,
                         "plans_with_processing_constraint_binding_at_1e_5": 0})
            summary["unique_plans"] += 1
            summary["maximum_processing_utilization"] = max(summary["maximum_processing_utilization"], utilization)
            summary["minimum_processing_slack"] = min(summary["minimum_processing_slack"], slack)
            summary["plans_with_processing_constraint_binding_at_1e_5"] += int(slack <= 1e-5)
            self.verified["unique_front_solutions_replayed"] += 1
        metrics = self.plan_cache[key]
        assert point["feasible"] and not point["violations"]
        for field in OBJECTIVE_FIELDS:
            close(metrics[field], point[field], f"{key} {field}")
        self.verified["front_point_occurrences"] += 1

    def capacity_certificate(self):
        """Prove redundancy from data-wide mass bounds, not observed solutions."""
        base_data, base, refs = self.instances["Test-4"]
        assert all(value >= 0 for value in base.generation.values())
        assert all(value == 0 for value in base.initial_producer_inventory.values())
        assert all(value == 0 for value in base.initial_facility_inventory.values())
        rows = []
        for waste in base.waste_types:
            total = sum(value for (producer, waste_type, period), value in base.generation.items()
                        if waste_type == waste)
            assert total > 0
            capacities = [value for (facility, waste_type, period), value in base.processing_capacity.items()
                          if waste_type == waste]
            assert capacities
            for capacity in capacities:
                close(capacity, 2 * total, "baseline processing capacity is twice global generation", atol=1e-9)
            lowest = self.instances["capacity-70"][1]
            minimum = min(capacity for (facility, waste_type, period), capacity in lowest.processing_capacity.items()
                          if waste_type == waste)
            assert minimum > total
            close(minimum / total, 1.4, "lowest capacity / mass upper bound", atol=1e-12)
            rows.append({"waste_type": waste, "global_generation_all_producers_and_periods": total,
                         "initial_global_inventory": 0., "any_single_facility_period_processing_upper_bound": total,
                         "baseline_capacity_each_facility_period": capacities[0],
                         "minimum_capacity_in_minus30percent_scenario": minimum,
                         "minimum_capacity_to_processing_upper_bound_ratio": minimum / total,
                         "minimum_unused_capacity_guaranteed_by_global_mass_bound": minimum - total})
        for scene in ("capacity-70", "capacity-80", "capacity-120", "capacity-130"):
            data, params, scene_refs = self.instances[scene]
            assert scene_refs == refs
            assert data["reference_plan_sha256"] == base_data["reference_plan_sha256"]
            assert [field for field, value in base_data["params"].items()
                    if value != data["params"][field]] == ["processing_capacity"]
            for (facility, waste, period), capacity in params.processing_capacity.items():
                bound = next(row["any_single_facility_period_processing_upper_bound"]
                             for row in rows if row["waste_type"] == waste)
                assert capacity >= bound
        self.global_capacity_certificate = {
            "status": "PASS", "base_instance": base_data["instance_id"],
            "base_instance_sha256": base_data["instance_sha256"],
            "scenario_multipliers": [.7, .8, 1., 1.2, 1.3],
            "zero_initial_producer_and_facility_inventories": True,
            "nonnegative_generation": True,
            "bound_argument": "Mass conservation and nonnegative inventories imply total processing over all facilities and periods is at most total generated mass (initial inventory is zero). Each nonnegative single-facility-period processing variable is therefore at most that same global total. Even the smallest scenario capacity is 1.4 times this bound; technology-forbidden cells remain at zero under every scenario.",
            "conclusion": "All five processing-capacity versions have identical feasible raw decision sets and identical original cost/risk objective functions. Capacity constraints are globally redundant, not merely inactive in observed front points. Differences in PPO approximations cannot be interpreted as gains from relaxed true capacity constraints; the capacity-dependent encoder input and finite search may differ.",
            "bound_references_and_reference_plan_identical_across_capacity_versions": True,
            "waste_type_certificates": rows}
        self.verified["global_capacity_redundancy_certificate"] = 1

    def front(self, task, budget):
        archive_id = f'{task["id"]}-B{budget}'
        front = read(OUT / f"fronts/{archive_id}.json")
        assert front["protocol_sha256"] == self.protocol_sha
        assert front["task"] == task and front["archive_id"] == archive_id
        data, params, refs = self.instances[task["instance"]]
        assert front["instance_sha256"] == data["instance_sha256"]
        assert (front["b_C"], front["b_R"]) == refs
        assert front["reference_plan_sha256"] == front["search_initial_plan_sha256"] == data["reference_plan_sha256"]
        assert front["budget"] == budget
        counts = front["counts"]
        assert counts["candidate_attempts"] == budget
        assert counts["candidate_objective_evaluations"] + counts["cache_hits"] == budget
        assert 0 <= counts["invalid_candidates"] <= counts["candidate_objective_evaluations"]
        assert counts["invalid_candidates"] <= counts["invalid_candidate_attempts"] <= budget
        assert 0 <= counts["operator_failures"] <= budget
        assert math.isfinite(front["seconds"]) and front["seconds"] > 0
        assert front["status"] == "success" and front["strict_feasible_points"] == len(front["points"])
        assert_front(front["points"], refs, self.protocol["tolerance_normalized"])
        for point in front["points"]:
            self.plan(task["instance"], point)
        if task["kind"] == "PPO":
            expected_internal_evaluations = budget - counts["operator_failures"] + 21 * (self.protocol["ppo_steps"] + 1)
            assert counts["ppo_internal_objective_evaluations"] == expected_internal_evaluations
            assert front["checkpoint"] == self.protocol["registry"]["models"][task["model"]]
            finals = front["final_solutions"]
            assert len(finals) == 21
            for index, final in enumerate(finals):
                preference = tuple(self.protocol["preferences"][index])
                assert tuple(final["preference"]) == preference
                plan = deserialize_plan(final["plan"])
                assert plan_sha256(plan) == final["solution_id"]
                solution = route_plan_to_solution(params, plan)
                metrics = evaluate_solution(params, solution, preference, objective_refs=refs)
                assert metrics["feasible"] and terminal_max(params, solution) <= 1e-5
                for field in OBJECTIVE_FIELDS:
                    close(metrics[field], final["metrics"][field], f"{archive_id} final {index} {field}")
                self.verified["ppo_final_weight_solutions_replayed"] += 1
            assert front["seconds"] + 1e-6 >= sum(f["seconds"] for f in finals)
        else:
            assert counts["ppo_internal_objective_evaluations"] == 0
        self.fronts[archive_id] = front
        self.verified["formal_fronts"] += 1

    def completion(self, task):
        record = read(OUT / f'completed/{task["id"]}.json')
        assert record["protocol_sha256"] == self.protocol_sha and record["task"] == task
        assert record["seconds_total_including_load_and_export"] > 0
        self.verified["completed_tasks"] += 1

    def pairs(self):
        for scale in ("Test-1", "Test-2", "Test-3", "Test-4"):
            refs = self.instances[scale][2]
            for model in ("small", "large"):
                for repeat in range(3):
                    budgets = (40320, 120960) if scale == "Test-4" and model == "large" else (40320,)
                    for budget in budgets:
                        p_id = f"PPO-{scale}-{model}-r{repeat}-B40320"
                        n_id = f"NSGA-{scale}-r{repeat}-B{budget}"
                        if p_id not in self.fronts or n_id not in self.fronts:
                            continue
                        p, n = self.fronts[p_id], self.fronts[n_id]
                        values = independent_pair(n["points"], p["points"], refs,
                                                  self.protocol["tolerance_normalized"])
                        self.pair_rows.append({"scale": scale, "model": model, "repeat": repeat,
                                               "nsga_budget": budget, "A_archive": n_id,
                                               "B_archive": p_id, **values})
        self.verified["coverage_pairs_recomputed"] = len(self.pair_rows)

    def small(self, index, repeat):
        record = read(OUT / f"small/ppo-p{index}-r{repeat}.json")
        data, params, refs = self.instances["small-fixed"]
        assert record["protocol_sha256"] == self.protocol_sha
        assert record["instance_sha256"] == data["instance_sha256"]
        assert (record["b_C"], record["b_R"]) == refs
        assert record["seed"] == self.protocol["small_seeds"][index][repeat]
        assert record["checkpoint"] == self.protocol["registry"]["models"]["small"]
        preference = tuple(self.protocol["small_preferences"][index])
        assert tuple(record["preference"]) == preference
        plan = deserialize_plan(record["plan"])
        assert plan_sha256(plan) == record["solution_id"]
        solution = route_plan_to_solution(params, plan)
        metrics = evaluate_solution(params, solution, preference, objective_refs=refs)
        assert metrics["feasible"] and terminal_max(params, solution) <= 1e-5
        for field in OBJECTIVE_FIELDS:
            close(metrics[field], record["metrics"][field], f"small p{index} r{repeat} {field}")
        self.small_rows.append({"preference_index": index, "repeat": repeat,
                                "seconds": record["seconds"], "solution_id": record["solution_id"],
                                "cost": metrics["cost"], "risk": metrics["risk"],
                                "J": metrics["weighted_objective"]})
        self.verified["small_ppo_solutions_replayed"] += 1

    def milp(self, index):
        protocol = read(OUT / "milp/protocol.json")
        record = read(OUT / f"milp/p{index}.json")
        assert protocol["solver_options"]["time_limit"] == 3600
        assert protocol["solver_options"]["mip_rel_gap"] == 0
        assert record["protocol_sha256"] == json_sha256(protocol)
        for file, expected in protocol["sources"].items():
            assert sha(ROOT / file) == expected
        data, params, refs = self.instances["small-fixed"]
        assert record["instance_sha256"] == data["instance_sha256"]
        assert tuple(record["objective_refs"]) == refs
        preference = tuple(self.protocol["small_preferences"][index])
        assert tuple(record["preference"]) == preference
        model = HazardousWasteMILP(replace(params, cost_weight=preference[0] / refs[0],
                                          risk_weight=preference[1] / refs[1]))
        model._build()
        x = np.asarray(record["full_solver_vector"], dtype=float)
        assert x.shape == (len(model.names),) and np.isfinite(x).all()
        row_values = np.array([sum(value * x[column] for column, value in row.items()) for row in model.rows])
        integer = x[np.array(model.integrality, dtype=bool)]
        residual = max(0., float(np.max(np.array(model.row_lbs) - row_values)),
                       float(np.max(row_values - np.array(model.row_ubs))),
                       float(np.max(np.array(model.lbs) - x)),
                       float(np.max(x - np.array(model.ubs))),
                       float(np.max(abs(integer - np.rint(integer)))))
        assert residual <= 1e-5, f"MILP matrix/bound/integer residual {residual}"
        solution = deserialize_solution(record["solution"])
        metrics = evaluate_solution(params, solution, preference, objective_refs=refs)
        assert metrics["feasible"] and terminal_max(params, solution) <= 1e-5
        for field in OBJECTIVE_FIELDS:
            close(metrics[field], record["metrics"][field], f"MILP p{index} {field}")
        close(np.dot(x, model.costs), record["solver"]["fun"], "MILP solver vector objective", atol=1e-7)
        original_decoded_objective = metrics["weighted_objective"]
        original_consistent = math.isclose(original_decoded_objective, record["solver"]["fun"], abs_tol=1e-7, rel_tol=1e-7)
        assert original_consistent == record["objective_consistent"]
        # Every preference uses the same explicit, source-linked numerical
        # normalization. Never bypass a mismatch or overwrite the original.
        validated_path = OUT / f"milp/validated/p{index}.json"
        normalized = read(validated_path)
        assert normalized["schema"] == "pareto-tolerance-normalized-milp-v1"
        assert normalized["original_result_sha256"] == sha(OUT / f"milp/p{index}.json")
        assert ROOT / normalized["original_result_path"] == OUT / f"milp/p{index}.json"
        assert normalized["cleanup_source_sha256"] == sha(ROOT / "normalize_pareto_milp.py")
        assert normalized["original_protocol_file_sha256"] == sha(OUT / "milp/protocol.json")
        assert normalized["feasibility_tolerance"] == 1e-5 and normalized["objective_tolerance"] == 1e-7
        assert normalized["additional_solver_calls"] == 0
        assert normalized["original_full_solver_vector"] == record["full_solver_vector"]
        assert normalized["original_metrics"] == record["metrics"]
        assert normalized["original_objective_consistent"] == original_consistent
        for field in ("status", "proven_optimal", "elapsed_seconds", "build_seconds", "solver_seconds", "solver",
                      "instance_sha256", "reference_sha256", "objective_refs", "preference", "protocol_sha256"):
            assert normalized[field] == record[field], "normalization changed immutable field " + field
        integer_mask = np.array(model.integrality, dtype=bool)
        assert np.max(abs(x[integer_mask] - np.rint(x[integer_mask]))) <= 1e-5
        independently_cleaned = x.copy()
        independently_cleaned[integer_mask] = np.rint(independently_cleaned[integer_mask])
        independently_cleaned[(~integer_mask) & (abs(independently_cleaned) < 1e-5)] = 0.
        assert np.array_equal(independently_cleaned, np.asarray(normalized["full_cleaned_solver_vector"]))
        assert normalized["full_solver_vector"] == normalized["full_cleaned_solver_vector"]
        changes = [{"index": offset, "name": list(model.names[offset]), "before": float(before), "after": float(after)}
                   for offset, (before, after) in enumerate(zip(x, independently_cleaned)) if before != after]
        assert normalized["changed_variables"] == changes
        clean_row_values = np.array([sum(value * independently_cleaned[column] for column, value in row.items())
                                     for row in model.rows])
        clean_matrix = max(0., float(np.max(np.array(model.row_lbs) - clean_row_values)),
                           float(np.max(clean_row_values - np.array(model.row_ubs))))
        clean_bound = max(0., float(np.max(np.array(model.lbs) - independently_cleaned)),
                          float(np.max(independently_cleaned - np.array(model.ubs))))
        clean_integer = float(np.max(abs(independently_cleaned[integer_mask] - np.rint(independently_cleaned[integer_mask]))))
        clean_residual = max(clean_matrix, clean_bound, clean_integer)
        assert clean_residual <= 1e-5
        for field, value in (("max_matrix_residual", clean_matrix), ("max_bound_residual", clean_bound),
                             ("max_integer_residual", clean_integer)):
            close(normalized[field], value, "independent normalized residual", atol=1e-12)
        normalized_solution = deserialize_solution(normalized["solution"])
        assert normalized_solution["raw"] == model._decode(independently_cleaned)["raw"]
        metrics = evaluate_solution(params, normalized_solution, preference, objective_refs=refs)
        assert metrics["feasible"] and terminal_max(params, normalized_solution) <= 1e-5
        assert normalized["strict_feasible"] and normalized["objective_consistent"]
        for field in OBJECTIVE_FIELDS:
            close(metrics[field], normalized["metrics"][field], "normalized MILP " + field)
        close(metrics["weighted_objective"], record["solver"]["fun"], "MILP normalized decoded objective", atol=1e-7)
        close(metrics["weighted_objective"], np.dot(independently_cleaned, model.costs), "normalized MILP vector objective", atol=1e-7)
        close(normalized["solver_objective_difference"], metrics["weighted_objective"] - record["solver"]["fun"],
              "normalized objective difference", atol=1e-12)
        if record["proven_optimal"]:
            assert record["solver"]["success"] and record["solver"]["status"] == 0
            assert record["solver"]["mip_gap"] <= 1e-8
        self.milp_rows.append({"preference_index": index, "cost": metrics["cost"], "risk": metrics["risk"],
                               "J": metrics["weighted_objective"], "seconds": record["elapsed_seconds"],
                               "max_matrix_bound_integrality_residual": residual,
                               "normalized_max_matrix_bound_integrality_residual": clean_residual,
                               "normalized_solver_objective_difference": normalized["solver_objective_difference"],
                               "normalization_changed_variables": len(changes),
                               "original_objective_consistent": original_consistent,
                               "original_decoded_objective": original_decoded_objective,
                               "validated_record_path": str(validated_path.relative_to(ROOT)),
                               "validated_record_sha256": sha(validated_path),
                               "proven_optimal": record["proven_optimal"], "status": record["status"]})
        self.verified["milp_full_vectors_replayed"] += 1
        self.verified["milp_independent_uniform_normalizations"] += 1

    def csv_rows(self, name):
        with (OUT / "tables" / name).open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def tables(self):
        """Check every source table cell against independently replayed runs."""
        pairs = self.csv_rows("coverage_pairs.csv")
        assert len(pairs) == len(self.pair_rows) == 27
        pair_lookup = {(r["scale"], r["model"], r["repeat"], r["nsga_budget"]): r for r in self.pair_rows}
        for row in pairs:
            key = (row["instance"], row["model"], int(row["repeat"]), int(row["nsga_budget"]))
            expected = pair_lookup[key]
            assert row["P_archive"] == expected["B_archive"] and row["N_archive"] == expected["A_archive"]
            for column, source in (("n_P", "n_B"), ("n_N", "n_A"),
                                   ("N_covers_P_count", "A_covers_B_count"), ("P_covers_N_count", "B_covers_A_count"),
                                   ("N_covers_P", "A_covers_B"), ("P_covers_N", "B_covers_A")):
                close(row[column], expected[source], "coverage CSV " + str(key) + " " + column, atol=1e-12)
            close(row["tolerance"], self.protocol["tolerance_normalized"], "coverage tolerance", atol=0)
            assert row["status"] == expected["status"]
        point_rows = self.csv_rows("frontier_points.csv")
        expected_points = {(aid, p["solution_id"]): (f, p)
                           for aid, f in self.fronts.items() for p in f["points"]}
        assert len(point_rows) == len(expected_points)
        seen = set()
        for row in point_rows:
            key = (row["archive_id"], row["solution_id"])
            assert key not in seen
            seen.add(key)
            front, point = expected_points[key]
            for column in ("cost", "risk"):
                close(row[column], point[column], "front points CSV " + str(key))
            for column in ("b_C", "b_R"):
                close(row[column], front[column], "front points refs " + str(key))
            assert row["solution_path"] == point["solution_path"]
            assert row["instance"] == front["task"]["instance"]
            assert row["instance_id"] == front["instance_id"] and row["algorithm"] == front["task"]["kind"]
            assert row["model"] == front["task"].get("model", "")
            assert int(row["repeat"]) == front["task"]["repeat"] and int(row["budget"]) == front["budget"]
            assert json.loads(row["source"]) == point["provenance"] and row["feasible"] == "True"
        run_rows = self.csv_rows("frontier_runs.csv")
        assert len(run_rows) == len(self.fronts) == 63
        for row in run_rows:
            front = self.fronts[row["archive_id"]]
            close(row["seconds"], front["seconds"], "front runtime CSV")
            close(row["preparation_seconds"], front["preparation_seconds"], "front preparation CSV")
            assert int(row["points"]) == len(front["points"]) and row["status"] == front["status"]
            for key, value in front["counts"].items():
                assert int(row[key]) == value, key
            return_validations = len(front.get("final_solutions", []))
            assert int(row["return_validation_evaluations"]) == return_validations
            assert int(row["objective_evaluations_total_excluding_preparation"]) == (
                front["counts"]["candidate_objective_evaluations"] + front["counts"]["ppo_internal_objective_evaluations"]
                + return_validations)
        selected = self.csv_rows("table4_selected_solutions.csv")
        independent = self.report()["small_selected"]
        assert len(selected) == len(independent) == 5
        for index, row in enumerate(selected):
            expected = next(r for r in independent if r["preference_index"] == index)
            milp = next(r for r in self.milp_rows if r["preference_index"] == index)
            assert tuple(ast.literal_eval(row["preference"])) == tuple(self.protocol["small_preferences"][index])
            assert int(row["selected_repeat"]) == expected["repeat"] and row["solution_id"] == expected["solution_id"]
            for column, key in (("ppo_cost", "cost"), ("ppo_risk", "risk"), ("ppo_J", "J"),
                                ("ppo_total_seconds", "three_restart_seconds"), ("relative_difference_percent", "relative_milp_percent")):
                close(row[column], expected[key], "table4 CSV " + column)
            close(row["milp_J"], milp["J"], "table4 MILP J")
            close(row["milp_seconds"], milp["seconds"], "table4 MILP time")
            assert row["milp_proven_optimal"] == str(milp["proven_optimal"])
            close(row["milp_cost"], milp["cost"], "table4 MILP cost")
            close(row["milp_risk"], milp["risk"], "table4 MILP risk")
            normalized = read(OUT / "milp" / "validated" / f"p{index}.json")
            for field in ("original_result_path", "original_result_sha256", "cleanup_source_sha256"):
                assert row["milp_" + field] == normalized[field], "table4 MILP lineage " + field
            for field in ("cleanup_seconds", "solver_objective_difference"):
                close(row["milp_" + field], normalized[field], "table4 MILP normalization " + field, atol=1e-12)
        scenes = self.csv_rows("sensitivity_points.csv")
        scenario_summary = self.csv_rows("sensitivity_scenarios.csv")
        expected_scenes = {"baseline", "capacity-70", "capacity-80", "capacity-120", "capacity-130",
                           "coload-70", "coload-80", "coload-120", "coload-130"}
        assert {r["scenario"] for r in scenes} == expected_scenes
        assert {r["scenario"] for r in scenario_summary} == expected_scenes and len(scenario_summary) == 9
        for scene in expected_scenes:
            source = "Test-4" if scene == "baseline" else scene
            refs = self.instances[source][2]
            rows = [r for r in scenes if r["scenario"] == scene]
            values = [{**r, "cost": float(r["cost"]), "risk": float(r["risk"])} for r in rows]
            assert_front(values, refs, self.protocol["tolerance_normalized"])
            union = [p for repeat in range(3)
                     for p in self.fronts[f"PPO-{source}-large-r{repeat}-B40320"]["points"]]
            pair = independent_pair(values, union, refs, self.protocol["tolerance_normalized"])
            assert pair["A_covers_B"] == 1, "union omitted an uncovered candidate"
            for row in rows:
                key = (row["archive_id"], row["solution_id"])
                front, point = expected_points[key]
                assert front["task"]["instance"] == source and front["task"]["model"] == "large"
                for column in ("cost", "risk"):
                    close(row[column], point[column], "sensitivity point " + scene)
                assert row["solution_path"] == point["solution_path"]
            summary = next(r for r in scenario_summary if r["scenario"] == scene)
            lowest_cost = min(values, key=lambda p: (p["cost"], p["risk"], p["solution_id"]))
            lowest_risk = min(values, key=lambda p: (p["risk"], p["cost"], p["solution_id"]))
            assert summary["minimum_cost_solution"] == lowest_cost["solution_id"]
            assert summary["minimum_risk_solution"] == lowest_risk["solution_id"]
            assert int(summary["points"]) == len(values)
            for column, value in (("minimum_cost", lowest_cost["cost"]), ("risk_at_minimum_cost", lowest_cost["risk"]),
                                  ("minimum_risk", lowest_risk["risk"]), ("cost_at_minimum_risk", lowest_risk["cost"]),
                                  ("b_C", refs[0]), ("b_R", refs[1])):
                close(summary[column], value, "sensitivity summary " + column)
            close(summary["time_mean"], statistics.mean(self.fronts[f"PPO-{source}-large-r{repeat}-B40320"]["seconds"]
                                                        for repeat in range(3)), "scenario mean whole-front time")
            params = self.instances[source][1]
            endpoint_plan = deserialize_plan(read(OUT / lowest_risk["solution_path"])["plan"])
            endpoint_solution = route_plan_to_solution(params, endpoint_plan)
            endpoint_metrics = evaluate_solution(params, endpoint_solution, (.5, .5), objective_refs=refs)
            assert int(summary["vehicle_period_routes_at_minimum_risk"]) == len(endpoint_plan)
            assert int(summary["service_visits_at_minimum_risk"]) == sum(len(route) - 2 for route in endpoint_plan.values())
            visits = {str(period): sum(len(route) - 2 for (vehicle, time), route in endpoint_plan.items() if time == period)
                      for period in params.periods}
            assert json.loads(summary["visits_by_period_at_minimum_risk"]) == visits
            for field in ("transport_risk", "coload_risk", "producer_inventory_risk", "facility_inventory_risk"):
                close(summary[field + "_at_minimum_risk"], endpoint_metrics[field], "endpoint risk component " + field)
            ordered_routes = sorted(endpoint_plan.items(), key=lambda item: (item[0][1], item[0][0]))
            mixed_routes = [item for item in ordered_routes if len({pickup_type(node) for node in item[1][1:-1]}) > 1]
            (vehicle, period), route = (mixed_routes or ordered_routes)[0]
            assert summary["example_route_vehicle"] == vehicle and int(summary["example_route_period"]) == period
            assert json.loads(summary["example_route"]) == route and summary["example_route_mixed"] == str(bool(mixed_routes))
            loads = {waste: endpoint_solution["raw"].get(("F", route[-2], route[-1], waste, vehicle, period), 0.)
                     for waste in params.waste_types}
            loads = {waste: quantity for waste, quantity in loads.items() if quantity > 1e-9}
            recorded_loads = json.loads(summary["example_return_loads"])
            assert set(loads) == set(recorded_loads)
            for waste, quantity in loads.items():
                close(quantity, recorded_loads[waste], "example actual return load")
            peak = 0.
            for value in values:
                saved_plan = deserialize_plan(read(OUT / value["solution_path"])["plan"])
                solution = route_plan_to_solution(params, saved_plan)
                peak = max(peak, max(solution["raw"].get(("p", j, s, t), 0.) / cap
                                     for (j, s, t), cap in params.processing_capacity.items()
                                     if params.technology[j, s]))
            close(summary["maximum_processing_utilization_in_union"], peak, "sensitivity union utilization")
        self.verified["csv_tables_cell_checked"] = 6

    def manuscript(self):
        text = (ROOT / "供应链管理写作/数值实验与结果分析_论文稿.md").read_text(encoding="utf-8")
        report = self.report()
        for index, preference in enumerate(self.protocol["small_preferences"]):
            selected = next(r for r in report["small_selected"] if r["preference_index"] == index)
            mip = next(r for r in self.milp_rows if r["preference_index"] == index)
            diff = f'{selected["relative_milp_percent"]:.3f}'
            if diff == "-0.000":
                diff = "0.000"
            diff += "†" if not mip["proven_optimal"] else ""
            row = ["PPO-Transformer", str(tuple(preference)), f'{selected["cost"]:.3f}', f'{selected["risk"]:.4f}',
                   f'{selected["J"]:.6f}', diff, f'{selected["three_restart_seconds"]:.2f}', "严格可行"]
            assert_manuscript_row(text, row, "table4 PPO manuscript row " + str(index), first_cell_aliases=("PPO",))
            row = ["MILP", str(tuple(preference)), f'{mip["cost"]:.3f}', f'{mip["risk"]:.4f}', f'{mip["J"]:.6f}',
                   "—", f'{mip["seconds"]:.2f}', "已证最优" if mip["proven_optimal"] else "限时可行（未证最优）"]
            assert_manuscript_row(text, row, "table4 MILP manuscript row " + str(index))
        summaries = {(r["scale"], r["model"], r["nsga_budget"]): r for r in report["coverage_summaries"]}

        def formatted(scale, model, budget):
            summary = summaries[scale, model, budget]
            return [f'{summary[direction + "_mean"]:.4f} ± {summary[direction + "_sample_sd"]:.4f}'
                    for direction in ("A_covers_B", "B_covers_A")]

        for model, label in (("small", "Train-S"), ("large", "Train-L")):
            row = [label] + [value for scale in ("Test-1", "Test-2", "Test-3", "Test-4")
                              for value in formatted(scale, model, 40320)]
            assert_manuscript_row(text, row, "table6 " + label)
        row = ["PPO（Train-L）与NSGA-II；L20，1实例", *formatted("Test-4", "large", 40320), *formatted("Test-4", "large", 120960)]
        assert_manuscript_row(text, row, "table5", first_cell_aliases=("PPO-L vs NSGA-II",))
        for forbidden in ("待填", "待核验", "两份模型各由一个训练实例", "每种50个未见实例"):
            assert forbidden not in text, "stale or unfinished manuscript claim: " + forbidden
        assert self.protocol_sha in text
        self.verified["main_manuscript_table_rows_verified"] = 13
        annex_count = 0

        def has_markdown_row(values, label, *, first_cell_aliases=()):
            nonlocal annex_count
            assert_manuscript_row(text, values, label, first_cell_aliases=first_cell_aliases)
            annex_count += 1

        for model, label in (("small", "S"), ("large", "L")):
            entry = self.protocol["registry"]["models"][model]
            training = read(ROOT / entry["training_record"])
            assert_checkpoint_row(text, [f"PPO（Train-{label}）", 24, training["training_seed"],
                                  f'{training["training_seconds"]:.3f}'], "Train-" + label, entry["sha256"],
                                  checkpoint_alias=entry.get("experiment_alias", "New-" + label))
            annex_count += 1

        def mean_sd(values):
            return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"

        for model, label in (("small", "Train-S"), ("large", "Train-L")):
            for scale in ("Test-1", "Test-2", "Test-3", "Test-4"):
                fronts = [self.fronts[f"PPO-{scale}-{model}-r{repeat}-B40320"] for repeat in range(3)]
                has_markdown_row([f"PPO（{label}）", scale, mean_sd([f["seconds"] for f in fronts]),
                                  mean_sd([len(f["points"]) for f in fronts]), "3/3", "100%", "100%"], "annex A2 PPO")
        for scale in ("Test-1", "Test-2", "Test-3", "Test-4"):
            for budget in ((40320, 120960) if scale == "Test-4" else (40320,)):
                fronts = [self.fronts[f"NSGA-{scale}-r{repeat}-B{budget}"] for repeat in range(3)]
                has_markdown_row([f'NSGA-II（B{1 if budget == 40320 else 2}）', scale,
                                  mean_sd([f["seconds"] for f in fronts]), mean_sd([len(f["points"]) for f in fronts]),
                                  "3/3", "100%", "100%"], "annex A2 NSGA")
        for row in self.csv_rows("sensitivity_scenarios.csv"):
            values = ["PPO（Train-L）", row["scenario"], f'{float(row["b_C"]):.3f}', f'{float(row["b_R"]):.4f}', row["points"],
                      f'{float(row["minimum_cost"]):.3f}', f'{float(row["risk_at_minimum_cost"]):.4f}',
                      f'{float(row["minimum_risk"]):.4f}', f'{float(row["cost_at_minimum_risk"]):.3f}',
                      f'{float(row["time_mean"]):.2f}']
            has_markdown_row(values, "annex A3 sensitivity endpoint", first_cell_aliases=("PPO-L",))
        self.verified["annex_manuscript_table_rows_verified"] = annex_count

    def figures(self):
        from PIL import Image
        png = OUT / "figures/sensitivity.png"
        with Image.open(png) as image:
            dpi = image.info.get("dpi")
            assert dpi and min(dpi) >= 300, f"PNG dpi={dpi}"
            assert image.width > image.height and image.width >= 3000
        svg = ET.parse(OUT / "figures/sensitivity.svg").getroot()
        ns = {"s": "http://www.w3.org/2000/svg"}
        rows = self.csv_rows("sensitivity_points.csv")
        svg_points = 0

        def tick_affine(axes, prefix, coordinate):
            anchors = []
            for group in axes.findall(".//s:g", ns):
                if not group.attrib.get("id", "").startswith(prefix):
                    continue
                labels = group.findall(".//s:text", ns)
                markers = [use for use in group.findall(".//s:use", ns) if coordinate in use.attrib]
                if not labels or not markers:
                    continue
                try:
                    value = float("".join(labels[0].itertext()).replace("−", "-"))
                except ValueError:
                    continue
                anchors.append((value, float(markers[0].attrib[coordinate])))
            assert len(anchors) >= 2, f"Cannot independently recover SVG {prefix} scale"
            data, pixels = np.array(anchors).T
            slope, intercept = np.polyfit(data, pixels, 1)
            assert np.max(abs(slope * data + intercept - pixels)) < 1e-4
            return slope, intercept

        first_axes = svg.find('.//s:g[@id="axes_1"]', ns)
        assert first_axes is not None
        y_slope, y_intercept = tick_affine(first_axes, "ytick_", "y")
        for panel, parameter in enumerate(("capacity", "coload"), 1):
            axes = svg.find(f'.//s:g[@id="axes_{panel}"]', ns)
            assert axes is not None
            x_slope, x_intercept = tick_affine(axes, "xtick_", "x")
            collections = [child for child in axes if child.attrib.get("id", "").startswith("PathCollection_")]
            assert len(collections) == 5, "Each SVG panel must have five real scatter series"
            for level, collection in zip((70, 80, 100, 120, 130), collections):
                scene = "baseline" if level == 100 else f"{parameter}-{level}"
                source_rows = [row for row in rows if row["scenario"] == scene]
                expected = len(source_rows)
                uses = collection.findall(".//s:use", ns)
                positions = [(float(use.attrib["x"]), float(use.attrib["y"])) for use in uses]
                # Matplotlib embeds triangle markers directly as translated
                # paths instead of <use> references. Recover their actual
                # centers from the symmetric marker bounding box.
                for path in collection.findall("s:path", ns):
                    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", path.attrib["d"])
                    coords = np.array([float(number) for number in numbers]).reshape(-1, 2)
                    center = (coords.min(axis=0) + coords.max(axis=0)) / 2
                    positions.append((float(center[0]), float(center[1])))
                actual = len(positions)
                assert actual == expected, f"SVG {scene}: {actual} markers, {expected} real points"
                for row, (pixel_x, pixel_y) in zip(source_rows, positions):
                    close(pixel_x, x_slope * float(row["cost"]) + x_intercept,
                          "SVG cost coordinate " + scene, atol=1e-4, rtol=0)
                    close(pixel_y, y_slope * float(row["risk"]) + y_intercept,
                          "SVG risk coordinate " + scene, atol=1e-4, rtol=0)
                svg_points += actual
        self.verified["svg_real_point_markers_checked"] = svg_points
        self.verified["two_panel_350dpi_and_vector_figure"] = 1

    def report(self):
        coverage_summaries = []
        keys = sorted({(r["scale"], r["model"], r["nsga_budget"]) for r in self.pair_rows})
        for scale, model, budget in keys:
            rows = [r for r in self.pair_rows if (r["scale"], r["model"], r["nsga_budget"]) == (scale, model, budget)]
            row = {"scale": scale, "model": model, "nsga_budget": budget, "paired_runs": len(rows)}
            for direction in ("A_covers_B", "B_covers_A"):
                values = [r[direction] for r in rows if r[direction] is not None]
                row[direction + "_mean"] = statistics.mean(values) if values else None
                row[direction + "_sample_sd"] = statistics.stdev(values) if len(values) > 1 else None
            coverage_summaries.append(row)
        selected_small = []
        for index in range(5):
            rows = [r for r in self.small_rows if r["preference_index"] == index]
            if len(rows) == 3:
                # Exact minimum; stable repeat id is the predeclared tie breaker.
                selected = min(rows, key=lambda r: (r["J"], r["repeat"]))
                row = {**selected, "three_restart_seconds": sum(r["seconds"] for r in rows)}
                milp = next((r for r in self.milp_rows if r["preference_index"] == index), None)
                if milp:
                    row["relative_milp_percent"] = 100 * (selected["J"] - milp["J"]) / milp["J"] if milp["J"] else None
                    row["absolute_milp_difference"] = selected["J"] - milp["J"]
                selected_small.append(row)
        delivery_paths = [ROOT / "供应链管理写作/数值实验与结果分析_论文稿.md",
                          ROOT / "report_pareto_experiments.py", ROOT / "plot_pareto_sensitivity.py",
                          ROOT / "normalize_pareto_milp.py", *sorted((OUT / "tables").glob("*.csv")),
                          OUT / "figures/sensitivity.svg", OUT / "figures/sensitivity.png"]
        delivery_hashes = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
                           for path in delivery_paths if path.exists()}
        return {"schema": "independent-pareto-numerics-audit-v1", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
                "auditor_sha256": sha(Path(__file__)), "protocol_sha256": self.protocol_sha,
                "scope": "Independent saved-solution/matrix replay, raw pair statistics, and optional complete report checks; narrative interpretation and visual layout still receive separate human/agent review.",
                "status": "FAIL" if self.errors else "INCOMPLETE" if self.missing else "PASS",
                "verified": dict(self.verified), "errors": self.errors, "missing": self.missing,
                "delivery_evidence_sha256": delivery_hashes,
                "coverage_pairs": self.pair_rows, "coverage_summaries": coverage_summaries,
                "small_ppo_runs": self.small_rows, "small_selected": selected_small,
                "milp": self.milp_rows, "processing_capacity_on_delivered_front_solutions": list(self.capacity_rows.values()),
                "global_processing_capacity_redundancy_certificate": self.global_capacity_certificate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-report", action="store_true", help="Also require final CSV tables, manuscript table rows and vector/PNG figure checks.")
    args = parser.parse_args()
    audit = Auditor()
    audit.item("protocol/models/data identities", audit.identities)
    for name in audit.protocol["instances"]:
        audit.item("instance " + name, lambda name=name: audit.instance(name))
    audit.item("global mass bound processing-capacity certificate", audit.capacity_certificate)
    for task in audit.protocol["tasks"]:
        for budget in task.get("budgets", (40320,)):
            audit.item(f'{task["id"]} B{budget}', lambda task=task, budget=budget: audit.front(task, budget))
        audit.item(task["id"] + " completion", lambda task=task: audit.completion(task))
    audit.item("independent paired coverage", audit.pairs)
    for index in range(5):
        for repeat in range(3):
            audit.item(f"small p{index} r{repeat}", lambda index=index, repeat=repeat: audit.small(index, repeat))
        audit.item(f"MILP p{index}", lambda index=index: audit.milp(index))
    if args.require_report:
        audit.item("CSV table cells and sensitivity union", audit.tables)
        audit.item("main manuscript numerical table rows", audit.manuscript)
        audit.item("SVG scatter markers and PNG resolution", audit.figures)
    expected = {"bound_instance_versions": 13, "formal_fronts": 63, "completed_tasks": 60,
                "coverage_pairs_recomputed": 27, "small_ppo_solutions_replayed": 15,
                "milp_full_vectors_replayed": 5, "ppo_final_weight_solutions_replayed": 1008}
    for name, count in expected.items():
        if audit.verified[name] != count:
            audit.missing.append({"item": name, "detail": f"Expected {count}; verified {audit.verified[name]}"})
    report = audit.report()
    destination = OUT / "audit/independent_numerics.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        history = destination.parent / "history"
        history.mkdir(exist_ok=True)
        previous = history / ("independent_numerics-" + sha(destination) + ".json")
        if not previous.exists():
            shutil.copy2(destination, previous)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "verified", "errors")}, ensure_ascii=False, indent=2))
    print(f"Missing evidence items: {len(audit.missing)}; audit: {destination.relative_to(ROOT)}")
    if audit.errors or (audit.missing and not args.allow_partial):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
