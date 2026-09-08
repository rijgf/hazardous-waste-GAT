"""Independent v4 audit. No training, solving, repair or v3 artifact writes.

The shared v3 replay checks retain their numeric tolerances and complete saved
solution checks. Only version identity, data lineage and the capacity certificate
are specialized here. Importing this module performs no artifact audit or write.
"""
from __future__ import annotations

import math
from itertools import combinations
import random
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import sys

import audit_pareto_delivery as shared
from sample_params import params_from_json_data
from src.reproducibility import deserialize_plan, json_sha256, plan_sha256
from src.solution_utils import evaluate_solution, route_plan_to_solution

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output/pareto-parameter-revision-v4"
OLD = ROOT / "output/pareto-single-instance-v3"
DATA = ROOT / "datasets/parameter_revision_v4"
REGISTRY = ROOT / "configs/ppo_models_parameter_revision_v4.json"
read, sha = shared.read, shared.sha


def _close(actual, expected, label, tolerance=1e-9):
    assert math.isclose(actual, expected, abs_tol=tolerance, rel_tol=1e-10), (
        f"{label}: actual={actual!r}, expected={expected!r}")


def inspect_effective_capacity(params, expected_factor):
    """Check sums over eligible facilities, not nominal forbidden cells."""
    rows = []
    for waste in params.waste_types:
        generation = [sum(params.generation[i, waste, t] for i in params.producers)
                      for t in params.periods]
        assert all(math.isfinite(value) and value >= 0 for value in generation)
        average = sum(generation) / len(generation)
        eligible = [j for j in params.facilities if params.technology[j, waste]]
        assert eligible and average > 0
        share = expected_factor * average / len(eligible)
        effective = []
        for period in params.periods:
            for facility in params.facilities:
                _close(params.processing_capacity[facility, waste, period], share,
                       "uniform nominal allocation")
            effective.append(sum(params.processing_capacity[j, waste, period] for j in eligible))
            _close(effective[-1], expected_factor * average, "effective system capacity")
        initial = (sum(params.initial_producer_inventory[i, waste] for i in params.producers)
                   + sum(params.initial_facility_inventory[j, waste] for j in params.facilities))
        assert math.isfinite(initial) and initial >= 0
        suffixes = []
        for start, period in enumerate(params.periods):
            required = sum(generation[start:]) + (initial if start == 0 else 0.)
            available = sum(effective[start:])
            suffixes.append({"first_period": period, "generated_mass_requiring_future_processing": required,
                             "future_effective_capacity": available, "slack": available - required,
                             "necessary_condition_pass": required <= available + 1e-5})
        rows.append({"waste_type": waste, "generation_each_period": generation,
                     "average_period_generation": average, "eligible_facilities": eligible,
                     "capacity_per_eligible_facility": share,
                     "effective_capacity_each_period": effective,
                     "effective_capacity_to_average_generation": expected_factor,
                     "initial_inventory": initial, "suffix_necessary_conditions": suffixes,
                     "all_suffix_necessary_conditions_pass": all(x["necessary_condition_pass"] for x in suffixes),
                     "scope": "Necessary only: passing does not prove route, vehicle or inventory feasibility."})
    return rows


def inspect_pair_calibration(original, revised, seed):
    """Replay independent standard-library draws; never call the generator."""
    rng = random.Random(seed)
    calibrated, rows = set(), []
    assert set(original.coload_risk) == set(revised.coload_risk)
    for first, second in combinations(original.waste_types, 2):
        pair = tuple(sorted((first, second)))
        if not original.compatibility[pair] or original.coload_risk[pair] <= 0:
            continue
        strength = 2. + 3. * rng.random()
        expected = strength * (original.waste_consequence[first] + original.waste_consequence[second]) / 2.
        actual = revised.coload_risk[pair]
        _close(actual, expected, "seeded pair gamma", tolerance=1e-12)
        ratio = 1. + 2. * actual / (original.waste_consequence[first] + original.waste_consequence[second])
        assert 3. <= ratio <= 6.
        calibrated.add(pair)
        rows.append({"pair": list(pair), "old_gamma": original.coload_risk[pair], "new_gamma": actual,
                     "independently_replayed_strength": strength,
                     "equal_load_total_transport_risk_ratio": ratio})
    assert rows, "no positive compatible pair"
    for pair, value in original.coload_risk.items():
        if pair not in calibrated:
            assert revised.coload_risk[pair] == value, "uncalibrated pair changed"
    return rows


def inspect_capacity_usage(params, raw):
    """Measure actual processing flows, excluding forbidden nominal capacity."""
    cell_utilization, system_utilization, slacks = [], [], []
    binding = 0
    for waste in params.waste_types:
        for period in params.periods:
            total_processed, effective_capacity = 0., 0.
            for facility in params.facilities:
                used = raw.get(("p", facility, waste, period), 0.)
                assert math.isfinite(used) and used >= -1e-5
                if not params.technology[facility, waste]:
                    assert abs(used) <= 1e-5, "processing in technology-forbidden cell"
                    continue
                capacity = params.processing_capacity[facility, waste, period]
                assert capacity > 0 and used <= capacity + 1e-5
                cell_utilization.append(used / capacity)
                slacks.append(capacity - used)
                binding += abs(capacity - used) <= 1e-5
                total_processed += used
                effective_capacity += capacity
            assert effective_capacity > 0
            system_utilization.append(total_processed / effective_capacity)
    return {"maximum_eligible_cell_utilization": max(cell_utilization),
            "maximum_effective_system_utilization": max(system_utilization),
            "minimum_eligible_cell_slack": min(slacks), "binding_eligible_cells": binding}


def independent_parameter_seed(instance_seed):
    payload = {"schema": "sha256-seed-v1", "master_seed": instance_seed,
               "namespace": ["parameter-revision-v4-coload"]}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 32) - 1)


def inspect_bound_reference(record):
    assert json_sha256(record["params"]) == record["instance_sha256"]
    params = params_from_json_data(record["params"])
    plan = deserialize_plan(record["reference_plan"])
    assert plan_sha256(plan) == record["reference_plan_sha256"] == record["search_initial_plan_sha256"]
    refs = record["b_C"], record["b_R"]
    assert all(math.isfinite(value) and value > 0 for value in refs)
    solution = route_plan_to_solution(params, plan)
    metrics = evaluate_solution(params, solution, (.5, .5), objective_refs=refs)
    assert metrics["feasible"] and shared.terminal_max(params, solution) <= 1e-5
    _close(metrics["cost"], refs[0], "reference cost b")
    _close(metrics["risk"], refs[1], "reference risk b")
    for field in ("cost", "risk", "fixed_cost", "distance_cost", "processing_cost",
                  "transport_risk", "coload_risk", "producer_inventory_risk", "facility_inventory_risk"):
        _close(metrics[field], record["reference_metrics"][field], "reference metric " + field)
    return params, plan, refs, inspect_capacity_usage(params, solution["raw"])


def inspect_revision_record(record):
    """Check one train/test/small baseline against its immutable old source."""
    source = (ROOT / record["source_instance_path"]).resolve()
    assert source.is_relative_to(ROOT)
    assert sha(source) == record["source_instance_sha256"]
    original = read(source)
    assert original["instance_sha256"] == record["original_params_sha256"]
    assert json_sha256(original["params"]) == original["instance_sha256"]
    assert set(original["params"]) == set(record["params"])
    differences = [key for key in original["params"] if original["params"][key] != record["params"][key]]
    assert set(differences) == {"processing_capacity", "coload_risk"}
    assert set(record["changed_fields"]) == set(differences)
    assert record["base_instance_id"] == original["instance_id"]
    assert record["instance_id"] == original["instance_id"] + "--revision-v4"
    assert record["instance_seed"] == original["instance_seed"]
    assert record["reference_seed"] == original["reference_seed"]
    params, plan, refs, usage = inspect_bound_reference(record)
    before = params_from_json_data(original["params"])
    seed = independent_parameter_seed(original["instance_seed"])
    calibration = record["parameter_revision"]
    assert calibration["risk_parameter_seed"] == seed
    assert calibration["capacity_factor"] == 2. and calibration["risk_total_ratio_range"] == [3., 6.]
    pairs = inspect_pair_calibration(before, params, seed)
    assert len(pairs) == len(calibration["pairs"])
    for observed, recorded in zip(pairs, calibration["pairs"]):
        assert observed["pair"] == recorded["pair"]
        for field in ("old_gamma", "new_gamma", "equal_load_total_transport_risk_ratio"):
            _close(observed[field], recorded[field], "calibration metadata " + field)
    capacities = inspect_effective_capacity(params, 2.)
    assert all(row["all_suffix_necessary_conditions_pass"] for row in capacities)
    assert len(calibration["capacity_by_waste"]) == len(capacities)
    for reported, independent in zip(calibration["capacity_by_waste"], capacities):
        assert reported["waste"] == independent["waste_type"]
        assert reported["eligible_facilities"] == independent["eligible_facilities"]
        for field in ("average_period_generation", "capacity_per_eligible_facility"):
            _close(reported[field], independent[field], "capacity calibration metadata " + field)
        for expected in independent["effective_capacity_each_period"]:
            _close(reported["effective_capacity_each_period"], expected, "recorded effective capacity")
    construction = record["reference_construction"]
    assert construction["solver_seed"] == 0 and construction["time_limit_per_period"] == 60.
    assert math.isfinite(construction["seconds"]) and construction["seconds"] > 0
    assert [row["period"] for row in construction["periods"]] == params.periods
    assert all(row["status"] in (0, 1) for row in construction["periods"])
    # The disclosed construction serves every entity every period; independently
    # inspect that claim without invoking its assignment MILP or routing helper.
    for period in params.periods:
        visits = [node for (vehicle, time), route in plan.items() if time == period for node in route[1:-1]]
        assert Counter(visits) == Counter(params.pickup_nodes), "reference is not all-period all-entity service"
    return {"instance_id": record["instance_id"], "instance_sha256": record["instance_sha256"],
            "source_instance_path": record["source_instance_path"], "changed_fields": differences,
            "b_C": refs[0], "b_R": refs[1], "parameter_seed": seed,
            "pair_calibration": pairs, "effective_capacity": capacities, "reference_capacity_usage": usage}


def assert_revision_report_evidence(evidence, capacities, pairs):
    """Match presentation metadata to independently inspected physical data."""
    expected_keys = {"capacity_by_instance", "risk_by_instance", "capacity_scenario_factors",
                     "risk_total_ratio_range", "risk_strength_range", "risk_scope",
                     "models_retrained", "training_hyperparameters_changed"}
    assert set(evidence) == expected_keys
    assert evidence["capacity_scenario_factors"] == [.7, .8, 1., 1.2, 1.3]
    assert evidence["risk_total_ratio_range"] == [3., 6.] and evidence["risk_strength_range"] == [2., 5.]
    assert evidence["models_retrained"] is True and evidence["training_hyperparameters_changed"] is False
    assert evidence["risk_scope"] == "same-arc equal positive two-type loads, transport only; not whole-system risk"
    assert set(evidence["capacity_by_instance"]) == set(evidence["risk_by_instance"]) == set(capacities) == set(pairs)
    for name, capacity_rows in capacities.items():
        observed = evidence["capacity_by_instance"][name]
        assert len(observed) == len(capacity_rows)
        for reported, independent in zip(observed, capacity_rows):
            assert set(reported) == {"waste", "average_period_generation", "eligible_facilities", "effective_capacity_each_period"}
            assert reported["waste"] == independent["waste_type"]
            assert reported["eligible_facilities"] == independent["eligible_facilities"]
            _close(reported["average_period_generation"], independent["average_period_generation"], name + " average")
            assert len(reported["effective_capacity_each_period"]) == len(independent["effective_capacity_each_period"])
            for actual, expected in zip(reported["effective_capacity_each_period"], independent["effective_capacity_each_period"]):
                _close(actual, expected, name + " reported effective capacity")
        risk = evidence["risk_by_instance"][name]
        assert set(risk) == {"compatible_pairs", "observed_total_ratio_range"}
        assert risk["compatible_pairs"] == len(pairs[name])
        ratios = [row["equal_load_total_transport_risk_ratio"] for row in pairs[name]]
        assert len(risk["observed_total_ratio_range"]) == 2
        for actual, expected in zip(risk["observed_total_ratio_range"], (min(ratios), max(ratios))):
            _close(actual, expected, name + " reported pair risk ratio")


def inspect_nsga_diagnostics(records, instances):
    """Independently count feasible work without equating it to discoveries."""
    records = [record for record in records if record["task"]["kind"] == "NSGA"]
    keys = sorted({(record["task"]["instance"], record["budget"]) for record in records})
    result = []
    for name, budget in keys:
        selected = sorted((record for record in records
                           if (record["task"]["instance"], record["budget"]) == (name, budget)),
                          key=lambda record: record["task"]["repeat"])
        reference = instances[name]["reference_plan_sha256"]
        nonreference_by_run, reference_only, unique = [], [], set()
        for record in selected:
            assert record["instance_id"] == instances[name]["instance_id"]
            assert record["reference_plan_sha256"] == reference
            counts = record["counts"]
            assert counts["candidate_attempts"] == budget
            assert 0 <= counts["invalid_candidate_attempts"] <= budget
            assert 0 <= counts["invalid_candidates"] <= counts["candidate_objective_evaluations"] <= budget
            identifiers = [point["solution_id"] for point in record["points"]]
            assert identifiers and len(identifiers) == len(set(identifiers))
            if counts["invalid_candidate_attempts"] == budget:
                assert identifiers == [reference], "no feasible candidate but unexplained non-reference archive"
            if identifiers == [reference]:
                reference_only.append(record["archive_id"])
            nonreference = set(identifiers) - {reference}
            unique.update(nonreference)
            nonreference_by_run.append({"archive_id": record["archive_id"], "repeat": record["task"]["repeat"],
                                        "nonreference_front_points": len(nonreference)})

        def total(field):
            return sum(record["counts"][field] for record in selected)

        attempts, invalid = total("candidate_attempts"), total("invalid_candidate_attempts")
        evaluations, invalid_evaluations = total("candidate_objective_evaluations"), total("invalid_candidates")
        result.append({"instance": name, "instance_id": instances[name]["instance_id"], "budget": budget,
                       "runs": len(selected), "candidate_attempts": attempts, "invalid_candidate_attempts": invalid,
                       "invalid_attempt_rate": invalid / attempts, "feasible_candidate_attempts": attempts - invalid,
                       "candidate_objective_evaluations": evaluations, "invalid_objective_evaluations": invalid_evaluations,
                       "feasible_objective_evaluations_excluding_preparation": evaluations - invalid_evaluations,
                       "operator_failures": total("operator_failures"), "reference_only_run_count": len(reference_only),
                       "reference_only_archive_ids": reference_only,
                       "nonreference_front_point_occurrences": sum(row["nonreference_front_points"] for row in nonreference_by_run),
                       "nonreference_unique_front_solutions": len(unique), "nonreference_front_points_by_run": nonreference_by_run})
    return result


def assert_risk_endpoint_prose(text, row):
    components = [float(row[field + "_risk_at_minimum_risk"]) for field in
                  ("transport", "coload", "producer_inventory", "facility_inventory")]
    assert all(math.isfinite(value) and value >= 0 for value in components)
    _close(sum(components), float(row["minimum_risk"]), "four endpoint risk components")
    description = "运输、共载、产废端库存、设施库存风险分别为" + "、".join(f"{value:.4f}" for value in components)
    assert description in text, "missing or changed fourth risk component in endpoint prose"


@contextmanager
def revision_context():
    """Temporarily route shared read-only checks, restoring v3 on exit."""
    previous = shared.OUT
    shared.OUT = OUT
    try:
        yield
    finally:
        shared.OUT = previous


class RevisionAuditor(shared.Auditor):
    def __init__(self):
        assert shared.OUT == OUT, "Use revision_context; never route a v4 audit to v3"
        super().__init__()
        self.revision_records, self.capacity_versions = {}, {}
        self.actual_capacity_usage, self.actual_usage_seen = {}, set()
        self.training_lineage = []
        self.encoder_capacity_checks = []
        self.nsga_diagnostics = []
        self.sensitivity_narrative_evidence = {}

    def identities(self):
        old_protocol = read(OLD / "protocol.json")
        revision = self.protocol["parameter_revision"]
        assert self.protocol["protocol"] == "pareto-parameter-revision-v4"
        assert revision["source_protocol_sha256"] == sha(OLD / "protocol.json")
        assert revision["models_retrained"] and revision["small_case_revised"]
        assert not revision["training_hyperparameters_changed"]
        assert revision["capacity_factor"] == 2. and revision["risk_strength_range"] == [2., 5.]
        assert revision["risk_total_ratio_range"] == [3., 6.]
        changed_protocol_fields = {"protocol", "sources", "registry", "registry_sha256",
                                   "dataset_manifest_sha256", "instances", "parameter_revision"}
        for key, value in old_protocol.items():
            if key not in changed_protocol_fields:
                assert self.protocol[key] == value, "algorithm/budget/seed changed: " + key
        for path, expected in self.protocol["sources"].items():
            assert sha(ROOT / path) == expected, "v4 source lock " + path
        for path, expected in old_protocol["sources"].items():
            assert sha(ROOT / path) == expected, "core algorithm changed since v3 " + path
        snapshot = OUT / "prior_artifact_hashes.json"
        assert sha(snapshot) == revision["old_artifact_snapshot_sha256"]
        for path, expected in read(snapshot).items():
            assert sha(ROOT / path) == expected, "prior artifact overwritten: " + path
        old_registry_path = ROOT / "configs/frozen_ppo_models.json"
        assert sha(old_registry_path) == old_protocol["registry_sha256"]
        old_registry = read(old_registry_path)
        for entry in old_registry["models"].values():
            assert sha(ROOT / entry["checkpoint"]) == entry["sha256"]
        registry = read(REGISTRY)
        assert registry == self.protocol["registry"] and sha(REGISTRY) == self.protocol["registry_sha256"]
        assert registry["objective_normalization"] == "instance_reference"
        assert ROOT / registry["dataset_manifest"] == DATA / "manifest.json"
        assert sha(DATA / "manifest.json") == registry["dataset_manifest_sha256"] == self.protocol["dataset_manifest_sha256"]
        manifest = read(DATA / "manifest.json")
        old_manifest = read(ROOT / old_registry["dataset_manifest"])
        assert manifest["source_manifest_sha256"] == old_registry["dataset_manifest_sha256"]
        assert manifest["algorithm"] == old_manifest["algorithm"]
        assert manifest["network"] == old_manifest["network"]
        assert manifest["network"]["objective_normalization"] == "instance_reference"
        assert manifest["training_instances_per_scale"] == 24 and manifest["test_instances_per_scale"] == 1
        assert manifest["source_hashes"] == self.protocol["sources"]
        records = manifest["records"]
        assert len(records) == 52
        counts = Counter((r["split"], r["scale"]) for r in records)
        assert counts == {("train", "Test-1"): 24, ("train", "Test-4"): 24,
                          **{("test", f"Test-{scale}"): 1 for scale in range(1, 5)}}
        old_entries = {r["path"]: r for r in old_manifest["records"]}
        hashes, seeds = {"train": set(), "test": set()}, {"train": set(), "test": set()}
        for entry in records:
            path = DATA / entry["path"]
            assert sha(path) == entry["sha256"]
            data = read(path)
            assert data["source_instance_sha256"] == old_entries[entry["path"]]["sha256"]
            assert data["split"] == entry["split"] and data["scale"] == entry["scale"]
            assert data["instance_sha256"] == entry["instance_sha256"]
            if data["split"] == "test":
                assert data["index"] == entry["index"] == 0
            hashes[data["split"]].add(data["instance_sha256"])
            seeds[data["split"]].add(data["instance_seed"])
            self.revision_records[data["instance_id"]] = inspect_revision_record(data)
        assert len(hashes["train"]) == 48 and len(hashes["test"]) == 4
        assert not hashes["train"] & hashes["test"] and not seeds["train"] & seeds["test"]
        import torch
        for name, entry in registry["models"].items():
            assert name in ("small", "large")
            assert sha(ROOT / entry["checkpoint"]) == entry["sha256"]
            assert entry["sha256"] != old_registry["models"][name]["sha256"]
            assert (ROOT / entry["checkpoint"]).is_relative_to(ROOT / "outputs/parameter_revision_v4/models")
            record = read(ROOT / entry["training_record"])
            old_training = read(ROOT / old_registry["models"][name]["training_record"])
            scale = "Test-1" if name == "small" else "Test-4"
            assert entry["training_scale"] == scale
            assert entry["experiment_alias"] == ("New-S-v4" if name == "small" else "New-L-v4")
            ids = {r["instance_id"] for r in records if r["split"] == "train" and r["scale"] == scale}
            assert record["instance_episode_counts"] == {identity: 20 for identity in ids}
            assert record["checkpoint_sha256"] == entry["sha256"]
            assert record["dataset_manifest_sha256"] == registry["dataset_manifest_sha256"]
            assert record["algorithm"] == manifest["algorithm"]["ppo"] == old_training["algorithm"]
            assert record["network"] == manifest["network"]
            assert record["training_seed"] == old_training["training_seed"]
            assert math.isfinite(record["training_seconds"]) and record["training_seconds"] > 0
            checkpoint = torch.load(ROOT / entry["checkpoint"], map_location="cpu", weights_only=False)
            assert checkpoint["seed"] == record["training_seed"]
            assert checkpoint["network_config"] == manifest["network"]
            assert checkpoint["algorithm_config"] == {"ppo": manifest["algorithm"]["ppo"]}
            assert checkpoint["optimizer_state_dict"] and checkpoint["state_dict"]
            assert all(torch.isfinite(value).all().item() for value in checkpoint["state_dict"].values())
            optimizer_states = checkpoint["optimizer_state_dict"]["state"]
            expected_updates = record["algorithm"]["train_iterations"] * record["algorithm"]["update_epochs"]
            assert optimizer_states and all(float(state["step"]) == expected_updates for state in optimizer_states.values())
            assert all(torch.isfinite(state[field]).all().item() for state in optimizer_states.values()
                       for field in ("exp_avg", "exp_avg_sq"))
            # v2 recorded runtime availability in its dataset manifest, not in
            # each model JSON. Do not invent a missing per-model device log.
            old_runtime = old_manifest["environment"]["torch_runtime"]
            assert record["environment"]["torch_runtime"]["cuda_available"] == old_runtime["cuda_available"]
            assert record["environment"]["torch_runtime"]["deterministic_algorithms"]
            assert not record["environment"]["torch_runtime"]["cudnn_benchmark"]
            assert record["environment"]["torch_runtime"]["cudnn_deterministic"]
            history_path = ROOT / f"outputs/parameter_revision_v4/training/{name}_history.json"
            history = read(history_path)
            assert history and all(len(values) == 20 for values in history.values())
            assert all(math.isfinite(float(value)) for values in history.values() for value in values)
            self.training_lineage.append({"model": name, "checkpoint_sha256": entry["sha256"],
                "training_record_sha256": sha(ROOT / entry["training_record"]), "history_sha256": sha(history_path),
                "training_instances": len(ids), "episodes_per_instance": 20, "training_seed": record["training_seed"],
                "training_seconds": record["training_seconds"], "algorithm_and_network_unchanged": True,
                "optimizer_steps_per_parameter": expected_updates, "checkpoint_and_optimizer_tensors_finite": True,
                "historical_runtime_comparison_source": old_registry["dataset_manifest"],
                "historical_per_model_device_record_available": "environment" in old_training,
                "environment": record["environment"]})
        assert set(registry["models"]) == {"small", "large"}
        preflight = read(OUT / "preflight.json")
        assert preflight["passed"] and preflight["protocol_sha256"] == self.protocol_sha
        self.verified.update({"frozen_checkpoints": 2, "preserved_old_frozen_checkpoints": 2,
                              "locked_revised_train_and_test_instances": 52, "retrained_model_lineages": 2})

    def instance(self, name):
        super().instance(name)
        data, params, refs = self.instances[name]
        if not name.startswith(("capacity-", "coload-")):
            self.revision_records[data["instance_id"]] = inspect_revision_record(data)
            if name.startswith("Test-"):
                assert data == read(DATA / f"test/{name}/000.json")
        else:
            assert data["base_instance_id"] == self.instances["Test-4"][0]["instance_id"]
            assert data["instance_seed"] == self.instances["Test-4"][0]["instance_seed"]
        # Each scenario is rebound and checked independently. Do not require
        # references or reference routes to equal the baseline when infeasible.
        inspect_bound_reference(data)
        from src.ppo_improver import StateEncoder
        network = read(DATA / "manifest.json")["network"]
        periods, pickups = len(params.periods), len(params.pickup_nodes)
        vehicle_periods, visits = len(params.vehicles) * periods, pickups * periods
        token_bound = 1 + int(network.get("use_preference_token", True)) + periods + vehicle_periods + visits
        if network.get("use_facility_tokens", True):
            token_bound += len(params.facilities) * len(params.waste_types) * periods
        if network.get("use_route_arc_tokens", True):
            token_bound += visits + min(vehicle_periods, visits)
        object_bound = max(visits, len(params.facilities))
        assert token_bound <= network["max_tokens"] and object_bound <= network["object_count"]
        encoder = StateEncoder(params, network, refs)
        plan = deserialize_plan(data["reference_plan"])
        for preference in ((1., 0.), (.5, .5), (0., 1.)):
            tokens, mask, global_values = encoder.encode(plan, preference, 0., 0.)
            for index in (3, 4, 5):
                _close(float(tokens[0, index]), 1., "bound-reference normalized encoder feature", 1e-6)
            assert int(mask.sum()) <= token_bound and encoder.objective_refs == refs
        self.encoder_capacity_checks.append({"instance": name, "safe_token_bound": token_bound,
            "safe_object_bound": object_bound, "checkpoint_token_capacity": network["max_tokens"],
            "checkpoint_object_capacity": network["object_count"], "reference_features_normalized_to_one": True})

    def capacity_certificate(self):
        for name, (data, params, refs) in self.instances.items():
            factor = 2. * (data["scenario_multiplier"] if name.startswith("capacity-") else 1.)
            rows = inspect_effective_capacity(params, factor)
            assert all(row["all_suffix_necessary_conditions_pass"] for row in rows)
            _, _, _, usage = inspect_bound_reference(data)
            self.capacity_versions[name] = {"instance_id": data["instance_id"], "b_C": refs[0], "b_R": refs[1],
                "reference_plan_sha256": data["reference_plan_sha256"], "waste_type_conditions": rows,
                "reference_capacity_usage": usage}
        self.verified["effective_capacity_and_suffix_certificates"] = len(self.capacity_versions)

    def plan(self, name, point):
        super().plan(name, point)
        if (name, point["solution_id"]) in self.actual_usage_seen:
            return
        params = self.instances[name][1]
        record = read(OUT / point["solution_path"])
        raw = route_plan_to_solution(params, deserialize_plan(record["plan"]))["raw"]
        self.observe_capacity_usage(name, point["solution_id"], raw)

    def small(self, index, repeat):
        super().small(index, repeat)
        record = read(OUT / f"small/ppo-p{index}-r{repeat}.json")
        params = self.instances["small-fixed"][1]
        raw = route_plan_to_solution(params, deserialize_plan(record["plan"]))["raw"]
        self.observe_capacity_usage("small-fixed", record["solution_id"], raw)

    def milp(self, index):
        super().milp(index)
        protocol = read(OUT / "milp/protocol.json")
        assert protocol["schema"] == "pareto-parameter-revision-milp-v4"
        assert protocol["main_protocol_sha256"] == self.protocol_sha
        assert protocol["parameter_revision"] == self.protocol["parameter_revision"]
        assert protocol["instance_id"] == self.instances["small-fixed"][0]["instance_id"]
        normalized = read(OUT / f"milp/validated/p{index}.json")
        assert normalized["reference_sha256"] == self.instances["small-fixed"][0]["reference_plan_sha256"]
        raw = shared.deserialize_solution(normalized["solution"])["raw"]
        self.observe_capacity_usage("small-fixed", "MILP-p" + str(index), raw)

    def observe_capacity_usage(self, name, identity, raw):
        key = name, identity
        if key in self.actual_usage_seen:
            return
        params = self.instances[name][1]
        usage = inspect_capacity_usage(params, raw)
        summary = self.actual_capacity_usage.setdefault(name, {"unique_solutions": 0,
            "maximum_eligible_cell_utilization": 0., "maximum_effective_system_utilization": 0.,
            "minimum_eligible_cell_slack": None, "solutions_with_binding_eligible_cell": 0})
        summary["unique_solutions"] += 1
        for field in ("maximum_eligible_cell_utilization", "maximum_effective_system_utilization"):
            summary[field] = max(summary[field], usage[field])
        field = "minimum_eligible_cell_slack"
        summary[field] = usage[field] if summary[field] is None else min(summary[field], usage[field])
        summary["solutions_with_binding_eligible_cell"] += bool(usage["binding_eligible_cells"])
        self.actual_usage_seen.add(key)

    def report_metadata(self):
        baselines = ["Test-1", "Test-2", "Test-3", "Test-4", "small-fixed"]
        capacities = {name: self.capacity_versions[name]["waste_type_conditions"] for name in baselines}
        pairs = {name: self.revision_records[self.instances[name][0]["instance_id"]]["pair_calibration"]
                 for name in baselines}
        results, verification = read(OUT / "results.json"), read(OUT / "report_verification.json")
        self.nsga_diagnostics = inspect_nsga_diagnostics(self.fronts.values(), {name: item[0] for name, item in self.instances.items()})
        assert len(self.nsga_diagnostics) == 5
        for metadata in (results, verification):
            assert metadata["protocol_sha256"] == self.protocol_sha
            assert metadata["report_script"] == "report_pareto_experiments.py"
            assert metadata["report_entry_script"] == "report_parameter_revision.py"
            for field in ("report_script", "report_entry_script"):
                assert metadata[field + "_sha256"] == sha(ROOT / metadata[field])
            assert_revision_report_evidence(metadata["parameter_revision_evidence"], capacities, pairs)
            assert metadata["nsga_search_diagnostics"] == self.nsga_diagnostics
        assert verification["passed"] is True
        expected_counts = {"formal_front_records": len(self.fronts), "formal_tasks": 60,
            "front_points_replayed": self.verified["front_point_occurrences"], "coverage_pair_count": len(self.pair_rows),
            "coverage_valid": sum(row["status"] == "valid" for row in self.pair_rows), "small_ppo_runs": 15, "milp_runs": 5}
        for field, expected in expected_counts.items():
            assert verification[field] == expected, "report verification count " + field
        assert verification["manuscript_sha256"] == sha(ROOT / "供应链管理写作/数值实验与结果分析_论文稿.md")
        assert verification["plot_script_sha256"] == sha(ROOT / "plot_pareto_sensitivity.py")
        assert verification["plot_variants_script_sha256"] == sha(ROOT / "plot_pareto_sensitivity_variants.py")
        assert verification["milp_cleanup_source_sha256"] == sha(ROOT / "normalize_pareto_milp.py")
        assert results["coverage_planned"] == 27 and results["coverage_valid"] == expected_counts["coverage_valid"]
        assert results["front_records"] == len(self.fronts)
        assert results["front_point_replays"] == self.verified["front_point_occurrences"]
        summary = self.report()
        assert len(results["small_selected"]) == 5
        for index, recorded in enumerate(results["small_selected"]):
            selected = next(row for row in summary["small_selected"] if row["preference_index"] == index)
            assert recorded["preference"] == self.protocol["small_preferences"][index]
            assert recorded["repeat"] == selected["repeat"]
            _close(recorded["J"], selected["J"], "results selected J")
            _close(recorded["difference"], selected["relative_milp_percent"], "results relative difference")
        summaries = {(row["scale"], row["model"], row["nsga_budget"]): row for row in summary["coverage_summaries"]}

        def formatted(scale, model, budget):
            record = summaries[scale, model, budget]
            return [f'{record[direction + "_mean"]:.4f} ± {record[direction + "_sample_sd"]:.4f}'
                    for direction in ("A_covers_B", "B_covers_A")]

        assert results["table5"] == formatted("Test-4", "large", 40320) + formatted("Test-4", "large", 120960)
        assert results["table6"] == [[label] + [value for scale in ("Test-1", "Test-2", "Test-3", "Test-4")
                                               for value in formatted(scale, model, 40320)]
                                    for model, label in (("small", "Train-S"), ("large", "Train-L"))]
        scenes = self.csv_rows("sensitivity_points.csv")
        signatures = {scene: [(row["solution_id"], float(row["cost"]), float(row["risk"]))
                             for row in scenes if row["scenario"] == scene]
                      for scene in ("baseline", "capacity-70", "capacity-80", "capacity-120", "capacity-130")}
        identical = all(signatures[scene] == signatures["baseline"] for scene in signatures)
        assert results["sensitivity_capacity_union_identical"] == identical
        interpretation = results["nsga_budget_response_interpretation"]
        assert verification["nsga_budget_response_interpretation"] == interpretation
        pairs = {(row["repeat"], row["nsga_budget"]): row for row in self.pair_rows
                 if row["scale"] == "Test-4" and row["model"] == "large"}
        differences = []
        for repeat in range(3):
            before, after = pairs[repeat, 40320], pairs[repeat, 120960]
            assert before["status"] == after["status"] == "valid"
            differences.extend((after["A_covers_B"] - before["A_covers_B"], before["B_covers_A"] - after["B_covers_A"]))
        if all(abs(value) <= 1e-12 for value in differences):
            assert "均相同，未观察到追加预算的覆盖收益" in interpretation
        elif min(differences) >= -1e-12 and max(differences) > 1e-12:
            assert "均未出现对NSGA-II不利的变化，且至少一项严格改善" in interpretation
        else:
            assert "不能概括为一致改善" in interpretation
        self.verified["nsga_search_effectiveness_groups_checked"] = 5
        self.verified["report_metadata_records_checked"] = 2

    def manuscript(self):
        super().manuscript()
        text = (ROOT / "供应链管理写作/数值实验与结果分析_论文稿.md").read_text(encoding="utf-8")
        for phrase in ("本轮重新训练后冻结", "configs/ppo_models_parameter_revision_v4.json",
                       "新版训练库48个实例", "processing_capacity和coload_risk两个参数字段",
                       "旧版以全规划期总量设置单设施能力的冗余性证明不适用于本轮",
                       "不将最大一对强制校准到5倍", "同一弧段、两类等正载量"):
            assert phrase in text, "missing v4 interpretation/lineage " + phrase
        assert "本轮没有训练、微调或检查点择优" not in text
        assert "本次五个能力水平的处理能力上限均为冗余约束" not in text
        cap = self.capacity_versions["Test-4"]["waste_type_conditions"]
        for values in ([row["average_period_generation"] for row in cap],
                       [row["effective_capacity_each_period"][0] for row in cap]):
            assert ", ".join(f"{value:.4f}" for value in values) in text
        assert "前者即使为100%也不表示基线进行了充分有效搜索" in text
        for row in self.nsga_diagnostics:
            label = f'{row["instance"]}/B{1 if row["budget"] == 40320 else 2}'
            by_run = "/".join(str(item["nonreference_front_points"]) for item in row["nonreference_front_points_by_run"])
            claim = (f'{label}：不可行尝试{row["invalid_candidate_attempts"]}/{row["candidate_attempts"]}'
                     f'（{row["invalid_attempt_rate"] * 100:.2f}%），严格可行尝试{row["feasible_candidate_attempts"]}次，'
                     f'缓存未命中后的可行评价{row["feasible_objective_evaluations_excluding_preparation"]}次；'
                     f'仅参考方案的档案{row["reference_only_run_count"]}/{row["runs"]}次，非参考前沿点逐次为{by_run}'
                     f'（合计{row["nonreference_front_point_occurrences"]}点位，按方案标识去重{row["nonreference_unique_front_solutions"]}份）')
            assert claim in text, "NSGA search effectiveness disclosure " + label
        rows = {row["scenario"]: row for row in self.csv_rows("sensitivity_scenarios.csv")}
        for scene in ("baseline", "coload-70", "coload-130"):
            assert_risk_endpoint_prose(text, rows[scene])
        ordered = [rows[scene] for scene in ("capacity-70", "capacity-80", "baseline", "capacity-120", "capacity-130")]
        baseline = rows["baseline"]
        for field, precision, label in (("minimum_cost", 3, "最低成本依次为"), ("minimum_risk", 4, "最低风险依次为")):
            values = [float(row[field]) for row in ordered]
            assert label + "、".join(f"{value:.{precision}f}" for value in values) in text
            base = float(baseline[field])
            assert base > 0
            changes = [100 * (value - base) / base for value in (values[0], values[-1])]
            formatted = "和".join(f"{value:+.2f}%".replace("-", "−") for value in changes)
            response_label = "最低成本变化分别为" if field == "minimum_cost" else "最低风险变化分别为"
            assert response_label + formatted in text
            self.sensitivity_narrative_evidence[field] = {"five_capacity_levels": values,
                                                         "minus30_and_plus30_change_percent": changes}
        share = 100 * float(baseline["coload_risk_at_minimum_risk"]) / float(baseline["minimum_risk"])
        assert f"基准最低风险方案的共载风险占总风险{share:.2f}%" in text
        self.sensitivity_narrative_evidence["baseline_risk_endpoint_coload_share_percent"] = share
        comparison = [row for row in self.pair_rows if row["scale"] == "Test-4" and row["model"] == "large"]
        assert len(comparison) == 6
        if all(row["A_covers_B"] == 1. and row["B_covers_A"] == 0. for row in comparison):
            assert "本次L20的Train-L比较中，B1与B2各自三次配对均得到𝒞(N,P)=1、𝒞(P,N)=0" in text
            assert "NSGA-II档案均覆盖对应PPO档案的全部点，而PPO均未覆盖NSGA-II档案中的任何点" in text
        assert "py -B audit_parameter_revision.py --require-report" in text
        self.verified["risk_components_and_capacity_response_prose_checked"] = 1
        self.verified["v4_manuscript_parameter_and_training_claims"] = 1

    def figure_variants(self):
        """Check new display coordinates independently against the original CSV."""
        from PIL import Image
        evidence = read(OUT / "figures/sensitivity_variants.json")
        assert evidence["source_sha256"] == sha(OUT / "tables/sensitivity_points.csv")
        assert evidence["plot_script_sha256"] == sha(ROOT / "plot_pareto_sensitivity_variants.py")
        assert evidence["visual_only"] is True and evidence["new_optimization_runs"] == 0
        rows = self.csv_rows("sensitivity_points.csv")
        assert set(evidence["scenarios"]) == {row["scenario"] for row in rows}
        for name, entry in evidence["scenarios"].items():
            knots = sorted([float(row["cost"]), float(row["risk"])]
                           for row in rows if row["scenario"] == name)
            assert entry["original_points"] == knots
            curve = entry["smooth_visual_points"]
            assert curve[0] == knots[0]
            for field in (0, 1):
                _close(curve[-1][field], knots[-1][field], "smooth endpoint")
            assert all(a[0] < b[0] and a[1] + 1e-8 >= b[1] for a, b in zip(curve, curve[1:]))
            lookup = dict(curve)
            for cost, risk in knots:
                _close(lookup[cost], risk, "smooth original knot")
            for left, right in zip(knots, knots[1:]):
                assert all(right[1] - 1e-8 <= risk <= left[1] + 1e-8
                           for cost, risk in curve if left[0] <= cost <= right[0])
        paper = (ROOT / "供应链管理写作/数值实验与结果分析_论文稿.md").read_text(encoding="utf8")
        for name, digest in evidence["artifacts"].items():
            assert sha(OUT / "figures" / name) == digest
            if name.endswith(".png"):
                assert "/figures/" + name + ")" in paper
                with Image.open(OUT / "figures" / name) as img:
                    assert min(img.info["dpi"]) >= 300 and img.width >= 3000
        assert "不表示两方案之间的成本—风险组合均可行" in paper
        assert "PCHIP" in paper
        self.verified["sensitivity_curve_variants_checked"] = 2

    def report(self):
        report = super().report()
        report.pop("global_processing_capacity_redundancy_certificate", None)
        report.update(schema="independent-parameter-revision-audit-v4", auditor_sha256=sha(Path(__file__)),
                      shared_auditor_sha256=sha(ROOT / "audit_pareto_delivery.py"),
                      training_lineage=self.training_lineage, revised_baseline_records=list(self.revision_records.values()),
                      effective_capacity_and_suffix_conditions=self.capacity_versions,
                      actual_processing_capacity_on_fronts_and_small_solutions=self.actual_capacity_usage,
                      encoder_capacity_checks=self.encoder_capacity_checks,
                      nsga_search_effectiveness=self.nsga_diagnostics,
                      sensitivity_endpoint_narrative_evidence=self.sensitivity_narrative_evidence,
                      capacity_interpretation="Effective-capacity and necessary suffix checks do not prove global redundancy or feasibility. Actual archived flows are measured separately; binding and slack are observations, not optimality claims.")
        for path in (Path(__file__), REGISTRY, ROOT / "report_parameter_revision.py", OUT / "results.json", OUT / "report_verification.json", ROOT / "plot_pareto_sensitivity_variants.py", OUT / "figures/sensitivity_variants.json"):
            if path.exists():
                report["delivery_evidence_sha256"][path.relative_to(ROOT).as_posix()] = sha(path)
        return report


def parse_arguments(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    # The already source-locked orchestration entry dispatches its audit stage
    # without replacing argv. Accept only that explicit dispatch token.
    if arguments and arguments[0] == "audit":
        arguments.pop(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-report", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parsed = parser.parse_args(arguments)
    if parsed.preflight_only and (parsed.require_report or parsed.allow_partial):
        parser.error("--preflight-only cannot request a partial or complete result audit")
    return parsed


def main():
    args = parse_arguments()
    with revision_context():
        audit = RevisionAuditor()
        audit.item("revised source/dataset/model lineage", audit.identities)
        for name in audit.protocol["instances"]:
            audit.item("instance " + name, lambda name=name: audit.instance(name))
        audit.item("effective capacity and suffix necessary conditions", audit.capacity_certificate)
        if not args.preflight_only:
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
            audit.item("report metadata and revised parameter evidence", audit.report_metadata)
            audit.item("main manuscript numerical table rows", audit.manuscript)
            audit.item("SVG markers and PNG resolution", audit.figures)
            audit.item("line and smooth display variants", audit.figure_variants)
        expected = {"bound_instance_versions": 13, "effective_capacity_and_suffix_certificates": 13,
                    "retrained_model_lineages": 2, "locked_revised_train_and_test_instances": 52}
        if not args.preflight_only:
            expected.update({"formal_fronts": 63, "completed_tasks": 60, "coverage_pairs_recomputed": 27,
                             "small_ppo_solutions_replayed": 15, "milp_full_vectors_replayed": 5,
                             "ppo_final_weight_solutions_replayed": 1008})
        if args.require_report:
            expected.update({"report_metadata_records_checked": 2, "v4_manuscript_parameter_and_training_claims": 1,
                             "nsga_search_effectiveness_groups_checked": 5,
                             "risk_components_and_capacity_response_prose_checked": 1,
                             "csv_tables_cell_checked": 6, "main_manuscript_table_rows_verified": 13,
                             "two_panel_350dpi_and_vector_figure": 1,
                             "sensitivity_curve_variants_checked": 2})
        for name, count in expected.items():
            if audit.verified[name] != count:
                audit.missing.append({"item": name, "detail": f"Expected {count}; verified {audit.verified[name]}"})
        report = audit.report()
        if args.preflight_only:
            report["status"] = "PRECHECK PASS" if not audit.errors and not audit.missing else "PRECHECK FAIL"
            report["scope"] = "Pre-experiment identity/data/reference/encoder/capacity checks only. No saved experimental outputs or final manuscript have been audited; this is not final numerical acceptance."
        destination = OUT / ("audit/independent_preflight.json" if args.preflight_only else "audit/independent_numerics.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            history = destination.parent / "history"
            history.mkdir(exist_ok=True)
            previous = history / (destination.stem + "-" + sha(destination) + ".json")
            if not previous.exists():
                shutil.copy2(destination, previous)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("status", "verified", "errors", "missing")},
                         ensure_ascii=False, indent=2))
        if audit.errors or (audit.missing and not args.allow_partial):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
