from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple
from urllib.parse import quote


_SCALE_FIELDS = (
    "producers_count",
    "waste_types_count",
    "facilities_count",
    "vehicles_count",
    "periods_count",
)
_SCALE_SPECS = {
    "Test-1": (3, 2, 2, 3, 2),
    "Test-2": (6, 2, 2, 4, 3),
    "Test-3": (10, 3, 3, 5, 3),
    "Test-4": (20, 4, 3, 8, 4),
}
_GENERATOR_PROFILE_KEYS = {
    "generation_range",
    "vehicle_capacity_factor",
    "producer_capacity_factor",
    "facility_capacity_factor",
    "processing_capacity_factor",
    "coordinate_range",
    "accident_probability_range",
    "vehicle_fixed_cost",
    "distance_cost",
    "processing_cost_range",
    "waste_consequence_range",
    "producer_inventory_risk_range",
    "facility_inventory_risk_range",
    "coload_risk_range",
    "compatibility_probability",
    "technology_probability",
    "require_terminal_clear",
}
_PPO_KEYS = {
    "train_iterations",
    "num_parallel_episodes",
    "episode_steps",
    "update_epochs",
    "gamma",
    "gae_lambda",
    "clip_ratio",
    "learning_rate",
    "entropy_coef",
    "value_coef",
    "repair_failure_penalty",
    "no_change_penalty",
    "eval_steps",
    "eval_candidate_samples",
    "evaluation_restarts",
    "preference_sampling",
}
_GA_KEYS = {
    "population_size",
    "generations",
    "crossover_rate",
    "mutation_rate",
    "elite_size",
    "tournament_size",
}
_NETWORK_KEYS = {
    "network_type",
    "embedding_dim",
    "transformer_layers",
    "attention_heads",
    "ff_hidden_dim",
    "dropout",
    "use_preference_token",
    "use_preference_in_global",
    "use_route_arc_tokens",
    "use_facility_tokens",
    "max_tokens",
    "operator_count",
    "object_count",
}


@dataclass(frozen=True)
class ExperimentCell:
    """One independently resumable supplementary-experiment execution."""

    protocol_id: str
    method: str
    model_id: str
    test_scale: str
    instance_index: int
    preference_id: str
    cost_weight: float
    risk_weight: float
    restart_index: int
    initial_solution_id: str
    seed: int


def load_protocol_config(path: str | Path) -> Dict[str, Any]:
    """Load a supplementary-experiment protocol from UTF-8 JSON."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError("protocol config must be a JSON object")
    return data


def generator_config_for_scale(config: Mapping[str, Any], scale_name: str) -> Dict[str, Any]:
    """Build an instance-generator config from the shared profile and one scale."""

    profile = dict(config["generator_profile"])
    scale = dict(config["test_scales"][scale_name]["scale"])
    overlap = set(profile).intersection(scale)
    if overlap:
        raise ValueError(f"scale fields leaked into generator_profile: {sorted(overlap)!r}")
    return {**profile, **scale}


def derive_protocol_seed(
    config: Mapping[str, Any],
    namespace: str,
    *parts: Any,
) -> int:
    """Derive a call-order-independent uint32 seed in a disjoint namespace lane."""

    namespaces = config["seed_namespaces"]
    if namespace not in namespaces:
        raise KeyError(f"unknown seed namespace: {namespace!r}")
    ordered_names = sorted(namespaces)
    if len(ordered_names) > 8:
        raise ValueError("at most eight uint32 seed namespaces are supported")
    lane = ordered_names.index(namespace)
    payload = json.dumps(
        {
            "protocol_id": config["protocol_id"],
            "seed_root": config["seed_root"],
            "namespace": namespaces[namespace],
            "parts": list(parts),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    low_bits = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 29) - 1)
    return (lane << 29) | low_bits


def enumerate_cells(
    config: Mapping[str, Any],
    *,
    initial_solution_ids: Mapping[tuple[str, int], str] | None = None,
) -> Tuple[ExperimentCell, ...]:
    """Expand the protocol into a stable ledger, optionally locked to plan hashes."""

    method_namespaces = {
        "ppo": "ppo_inference",
        "ga": "ga_restart",
        "heuristic": "heuristic",
        "milp": "milp",
    }
    cells = []
    for method, scope in config["methods"].items():
        namespace = method_namespaces[method]
        for model_id in scope["models"]:
            for scale_name in scope["scales"]:
                instance_count = int(config["test_scales"][scale_name]["instances"])
                for instance_index in range(instance_count):
                    initial_seed = derive_protocol_seed(
                        config,
                        "initial_solution",
                        scale_name,
                        instance_index,
                    )
                    if initial_solution_ids is None:
                        initial_solution_id = (
                            f"{scale_name}-i{instance_index:02d}-seed{initial_seed:010d}"
                        )
                    else:
                        identity_key = (scale_name, instance_index)
                        if identity_key not in initial_solution_ids:
                            raise KeyError(f"missing initial solution identity: {identity_key!r}")
                        initial_solution_id = initial_solution_ids[identity_key]
                        if not isinstance(initial_solution_id, str) or not initial_solution_id:
                            raise TypeError("initial solution identities must be non-empty strings")
                    for preference in config["preferences"]:
                        for restart_index in range(int(scope["restarts"])):
                            seed = derive_protocol_seed(
                                config,
                                namespace,
                                model_id,
                                scale_name,
                                instance_index,
                                preference["id"],
                                restart_index,
                                initial_solution_id,
                            )
                            cells.append(
                                ExperimentCell(
                                    protocol_id=str(config["protocol_id"]),
                                    method=str(method),
                                    model_id=str(model_id),
                                    test_scale=str(scale_name),
                                    instance_index=instance_index,
                                    preference_id=str(preference["id"]),
                                    cost_weight=float(preference["cost_weight"]),
                                    risk_weight=float(preference["risk_weight"]),
                                    restart_index=restart_index,
                                    initial_solution_id=initial_solution_id,
                                    seed=seed,
                                )
                            )
    return tuple(cells)


def build_cell_id(cell: ExperimentCell) -> str:
    """Return an injectively framed, path-safe identifier for ``cell``."""

    values = (
        ("p", cell.protocol_id),
        ("a", cell.method),
        ("m", cell.model_id),
        ("s", cell.test_scale),
        ("i", str(cell.instance_index)),
        ("w", cell.preference_id),
        ("wc", json.dumps(cell.cost_weight, allow_nan=False)),
        ("wr", json.dumps(cell.risk_weight, allow_nan=False)),
        ("r", str(cell.restart_index)),
        ("x", cell.initial_solution_id),
        ("z", str(cell.seed)),
    )
    framed = []
    for label, value in values:
        encoded = quote(value, safe="-._~")
        framed.append(f"{label}{len(encoded)}-{encoded}")
    return "cell-v1__" + "__".join(framed)


def aggregate_restarts_then_instances(
    rows: Tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    value_key: str,
    *,
    instance_key: str = "instance_index",
    restart_key: str = "restart_index",
) -> Dict[str, Any]:
    """Average restarts within instances, then summarize independent instances."""

    grouped: Dict[Any, list[float]] = {}
    seen = set()
    for row in rows:
        instance_id = row[instance_key]
        restart_id = row[restart_key]
        identity = (instance_id, restart_id)
        if identity in seen:
            raise ValueError(f"duplicate instance/restart row: {identity!r}")
        seen.add(identity)
        raw_value = row[value_key]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError(f"{value_key} must be a finite number")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f"{value_key} must be finite")
        grouped.setdefault(instance_id, []).append(value)

    instance_means = {
        instance_id: statistics.fmean(values)
        for instance_id, values in grouped.items()
    }
    values = list(instance_means.values())
    return {
        "instance_means": instance_means,
        "n_instances": len(values),
        "mean": statistics.fmean(values) if values else None,
        "sample_std": statistics.stdev(values) if len(values) >= 2 else None,
        "ddof": 1,
    }


def milp_gap_summary(
    rows: Tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    *,
    candidate_key: str = "objective",
    reference_key: str = "milp_objective",
    status_key: str = "milp_status",
    instance_key: str = "instance_index",
    restart_key: str = "restart_index",
) -> Dict[str, Any]:
    """Summarize minimization gaps against proven-optimal MILP references only."""

    grouped: Dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row[instance_key], []).append(row)

    gaps: Dict[Any, float] = {}
    for instance_id, instance_rows in grouped.items():
        statuses = {
            str(row[status_key]).strip().casefold()
            for row in instance_rows
        }
        if len(statuses) != 1:
            raise ValueError(f"inconsistent MILP status for instance {instance_id!r}")
        if statuses != {"optimal"}:
            continue

        references = []
        for row in instance_rows:
            raw_reference = row[reference_key]
            if isinstance(raw_reference, bool) or not isinstance(raw_reference, (int, float)):
                raise TypeError(f"{reference_key} must be a finite number")
            reference = float(raw_reference)
            if not math.isfinite(reference):
                raise ValueError(f"{reference_key} must be finite")
            references.append(reference)
        if len(set(references)) != 1:
            raise ValueError(f"inconsistent MILP objective for instance {instance_id!r}")
        reference = references[0]
        if reference == 0.0:
            continue

        candidate_summary = aggregate_restarts_then_instances(
            instance_rows,
            candidate_key,
            instance_key=instance_key,
            restart_key=restart_key,
        )
        candidate_mean = candidate_summary["mean"]
        gaps[instance_id] = (candidate_mean - reference) / abs(reference) * 100.0

    gap_values = list(gaps.values())
    return {
        "gaps_by_instance": gaps,
        "valid_n": len(gap_values),
        "mean_gap_percent": statistics.fmean(gap_values) if gap_values else None,
        "sample_std_gap_percent": (
            statistics.stdev(gap_values) if len(gap_values) >= 2 else None
        ),
        "ddof": 1,
    }


def capacity_preflight(config: Mapping[str, Any]) -> Dict[str, Dict[str, int | bool]]:
    """Compute conservative encoder and action-head capacity needs per scale.

    With all current token families enabled, the state upper bound is
    ``2 + T + 2*V*T + 2*P*W*T + F*W*T``.  The action decoder in
    :mod:`src.ppo_improver` requires the larger of pickup-period and
    vehicle-period address spaces.
    """

    capacity = config["network_capacity"]
    max_tokens = int(capacity["max_tokens"])
    object_count = int(capacity["object_count"])
    report: Dict[str, Dict[str, int | bool]] = {}
    for name, item in config["test_scales"].items():
        scale = item["scale"]
        producers = int(scale["producers_count"])
        waste_types = int(scale["waste_types_count"])
        facilities = int(scale["facilities_count"])
        vehicles = int(scale["vehicles_count"])
        periods = int(scale["periods_count"])
        required_tokens = (
            2
            + periods
            + 2 * vehicles * periods
            + 2 * producers * waste_types * periods
            + facilities * waste_types * periods
        )
        required_objects = max(
            producers * waste_types * periods,
            vehicles * periods,
        )
        report[name] = {
            "required_tokens": required_tokens,
            "required_object_slots": required_objects,
            "max_tokens": max_tokens,
            "object_count": object_count,
            "fits": required_tokens <= max_tokens and required_objects <= object_count,
        }
    return report


def protocol_cell_counts(config: Mapping[str, Any]) -> Dict[str, int]:
    """Return the number of unique execution cells requested per method."""

    preferences = len(config["preferences"])
    test_scales = config["test_scales"]
    counts: Dict[str, int] = {}
    for method, scope in config["methods"].items():
        instance_count = sum(test_scales[name]["instances"] for name in scope["scales"])
        counts[method] = (
            len(scope["models"])
            * instance_count
            * preferences
            * int(scope["restarts"])
        )
    counts["total"] = sum(counts.values())
    return counts


def validate_protocol(config: Mapping[str, Any]) -> None:
    """Reject a protocol that does not match the frozen supplementary design."""

    if not isinstance(config, Mapping):
        raise TypeError("protocol config must be a mapping")
    required_top_level = {
        "protocol_id",
        "output_dir",
        "seed_root",
        "seed_namespaces",
        "scale_fields",
        "generator_profile",
        "models",
        "test_scales",
        "preferences",
        "restarts",
        "methods",
        "algorithm",
        "network",
        "network_capacity",
    }
    if set(config) != required_top_level:
        raise ValueError("protocol top-level keys do not match supplementary-experiment-v1")
    if config["protocol_id"] != "supplementary-experiment-v1":
        raise ValueError("unexpected protocol_id")
    if config["output_dir"] != "outputs/supplementary_experiment_v1":
        raise ValueError("unexpected output_dir")
    if config["seed_root"] != 42:
        raise ValueError("seed_root must remain fixed at 42")

    seed_namespaces = config["seed_namespaces"]
    required_seed_namespaces = {
        "training_instance",
        "training_algorithm",
        "test_instance",
        "initial_solution",
        "ppo_inference",
        "ga_restart",
        "heuristic",
        "milp",
    }
    if not isinstance(seed_namespaces, Mapping) or set(seed_namespaces) != required_seed_namespaces:
        raise ValueError("seed namespaces are incomplete")
    seed_tags = list(seed_namespaces.values())
    if not all(isinstance(tag, str) and tag for tag in seed_tags):
        raise TypeError("seed namespace tags must be non-empty strings")
    if len(seed_tags) != len(set(seed_tags)):
        raise ValueError("seed namespace tags must be mutually exclusive")
    expected_seed_namespaces = {
        "training_instance": "training-instance-v1",
        "training_algorithm": "training-algorithm-v1",
        "test_instance": "test-instance-v1",
        "initial_solution": "initial-solution-v1",
        "ppo_inference": "ppo-inference-v1",
        "ga_restart": "ga-restart-v1",
        "heuristic": "heuristic-v1",
        "milp": "milp-v1",
    }
    if seed_namespaces != expected_seed_namespaces:
        raise ValueError("seed namespace tags differ from the frozen protocol")

    if tuple(config["scale_fields"]) != _SCALE_FIELDS:
        raise ValueError("scale_fields must contain the five frozen size fields")
    profile = config["generator_profile"]
    if not isinstance(profile, Mapping) or set(profile) != _GENERATOR_PROFILE_KEYS:
        raise ValueError("generator_profile keys are incomplete")
    if set(profile).intersection(_SCALE_FIELDS):
        raise ValueError("generator_profile must not contain scale fields")
    if profile["require_terminal_clear"] is not True:
        raise ValueError("terminal inventory clearing must remain enabled")
    expected_profile = {
        "generation_range": [0.5, 2.0],
        "vehicle_capacity_factor": 1.8,
        "producer_capacity_factor": 2.5,
        "facility_capacity_factor": 3.0,
        "processing_capacity_factor": 2.0,
        "coordinate_range": [0.0, 100.0],
        "accident_probability_range": [0.0005, 0.003],
        "vehicle_fixed_cost": 50.0,
        "distance_cost": 3.0,
        "processing_cost_range": [4.0, 20.0],
        "waste_consequence_range": [1.0, 4.0],
        "producer_inventory_risk_range": [0.2, 1.0],
        "facility_inventory_risk_range": [0.1, 0.8],
        "coload_risk_range": [0.05, 0.5],
        "compatibility_probability": 1.0,
        "technology_probability": 0.85,
        "require_terminal_clear": True,
    }
    if profile != expected_profile:
        raise ValueError("generator_profile differs from the frozen protocol")

    test_scales = config["test_scales"]
    if not isinstance(test_scales, Mapping) or set(test_scales) != set(_SCALE_SPECS):
        raise ValueError("test_scales must be exactly Test-1 through Test-4")
    for scale_name, expected_values in _SCALE_SPECS.items():
        item = test_scales[scale_name]
        if not isinstance(item, Mapping) or set(item) != {"instances", "scale"}:
            raise ValueError(f"invalid test scale entry: {scale_name}")
        if item["instances"] != 50:
            raise ValueError(f"{scale_name} must contain 50 instances")
        expected_scale = dict(zip(_SCALE_FIELDS, expected_values))
        if item["scale"] != expected_scale:
            raise ValueError(f"unexpected size fields for {scale_name}")

    expected_models = [
        {"id": "Train-S", "train_scale": "Test-1", "checkpoint": "small_model.pt"},
        {"id": "Train-L", "train_scale": "Test-4", "checkpoint": "large_model.pt"},
    ]
    if config["models"] != expected_models:
        raise ValueError("exactly the frozen Train-S and Train-L models are required")
    expected_preferences = [
        {"id": "C100-R000", "cost_weight": 1.0, "risk_weight": 0.0},
        {"id": "C075-R025", "cost_weight": 0.75, "risk_weight": 0.25},
        {"id": "C050-R050", "cost_weight": 0.5, "risk_weight": 0.5},
        {"id": "C025-R075", "cost_weight": 0.25, "risk_weight": 0.75},
        {"id": "C000-R100", "cost_weight": 0.0, "risk_weight": 1.0},
    ]
    if config["preferences"] != expected_preferences:
        raise ValueError("the five frozen preferences are required")
    if config["restarts"] != 3:
        raise ValueError("the protocol requires three stochastic restarts")

    expected_methods = {
        "ppo": {
            "models": ["Train-S", "Train-L"],
            "scales": list(_SCALE_SPECS),
            "restarts": 3,
        },
        "ga": {"models": ["GA"], "scales": list(_SCALE_SPECS), "restarts": 3},
        "heuristic": {
            "models": ["Heuristic"],
            "scales": ["Test-1", "Test-4"],
            "restarts": 1,
        },
        "milp": {"models": ["MILP"], "scales": ["Test-1"], "restarts": 1},
    }
    if config["methods"] != expected_methods:
        raise ValueError("method scopes do not match the frozen execution grid")

    algorithm = config["algorithm"]
    if not isinstance(algorithm, Mapping) or set(algorithm) != {
        "ppo",
        "ga_by_scale",
        "milp_time_limit_seconds",
    }:
        raise ValueError("algorithm configuration is incomplete")
    ppo = algorithm["ppo"]
    if not isinstance(ppo, Mapping) or set(ppo) != _PPO_KEYS:
        raise ValueError("complete PPO hyperparameters are required")
    if set(ppo["preference_sampling"]) != {
        "endpoint_probability_each",
        "beta_concentration",
    }:
        raise ValueError("PPO preference_sampling is incomplete")
    if ppo["evaluation_restarts"] != config["restarts"]:
        raise ValueError("PPO evaluation_restarts must match protocol restarts")
    expected_ppo = {
        "train_iterations": 20,
        "num_parallel_episodes": 24,
        "episode_steps": 24,
        "update_epochs": 24,
        "gamma": 0.95,
        "gae_lambda": 0.9,
        "clip_ratio": 0.2,
        "learning_rate": 0.0003,
        "entropy_coef": 0.02,
        "value_coef": 0.5,
        "repair_failure_penalty": 0.2,
        "no_change_penalty": 0.01,
        "eval_steps": 60,
        "eval_candidate_samples": 32,
        "evaluation_restarts": 3,
        "preference_sampling": {
            "endpoint_probability_each": 0.1,
            "beta_concentration": 0.5,
        },
    }
    if ppo != expected_ppo:
        raise ValueError("PPO hyperparameters differ from the frozen protocol")
    ga_by_scale = algorithm["ga_by_scale"]
    if not isinstance(ga_by_scale, Mapping) or set(ga_by_scale) != set(_SCALE_SPECS):
        raise ValueError("GA configuration must cover all four test scales")
    for scale_name, ga in ga_by_scale.items():
        if not isinstance(ga, Mapping) or set(ga) != _GA_KEYS:
            raise ValueError(f"complete GA hyperparameters are required for {scale_name}")
        expected_size = (30, 60) if scale_name in {"Test-1", "Test-2"} else (20, 40)
        expected_ga = {
            "population_size": expected_size[0],
            "generations": expected_size[1],
            "crossover_rate": 0.85 if scale_name in {"Test-1", "Test-2"} else 0.9,
            "mutation_rate": 0.2 if scale_name in {"Test-1", "Test-2"} else 0.25,
            "elite_size": 2 if scale_name in {"Test-1", "Test-2"} else 4,
            "tournament_size": 3,
        }
        if ga != expected_ga:
            raise ValueError(f"GA hyperparameters differ from the frozen protocol: {scale_name}")
    if algorithm["milp_time_limit_seconds"] != 180.0:
        raise ValueError("MILP time limit must be 180 seconds")

    network = config["network"]
    if not isinstance(network, Mapping) or set(network) != _NETWORK_KEYS:
        raise ValueError("complete network configuration is required")
    for switch in (
        "use_preference_token",
        "use_preference_in_global",
        "use_route_arc_tokens",
        "use_facility_tokens",
    ):
        if not isinstance(network[switch], bool):
            raise TypeError(f"network switch {switch} must be boolean")
    expected_network = {
        "network_type": "full_transformer",
        "embedding_dim": 48,
        "transformer_layers": 1,
        "attention_heads": 4,
        "ff_hidden_dim": 96,
        "dropout": 0.0,
        "use_preference_token": True,
        "use_preference_in_global": False,
        "use_route_arc_tokens": True,
        "use_facility_tokens": True,
        "max_tokens": 896,
        "operator_count": 8,
        "object_count": 512,
    }
    if network != expected_network:
        raise ValueError("network configuration differs from the frozen protocol")
    capacity = config["network_capacity"]
    if not isinstance(capacity, Mapping) or set(capacity) != {"max_tokens", "object_count"}:
        raise ValueError("network_capacity must contain max_tokens and object_count")
    if network["max_tokens"] != capacity["max_tokens"]:
        raise ValueError("network max_tokens does not match network_capacity")
    if network["object_count"] != capacity["object_count"]:
        raise ValueError("network object_count does not match network_capacity")
    if capacity != {"max_tokens": 896, "object_count": 512}:
        raise ValueError("network_capacity differs from the frozen protocol")
    if not all(item["fits"] for item in capacity_preflight(config).values()):
        raise ValueError("at least one scale exceeds frozen network capacity")

    expected = {
        "ppo": 6000,
        "ga": 3000,
        "heuristic": 500,
        "milp": 250,
        "total": 9750,
    }
    actual = protocol_cell_counts(config)
    if actual != expected:
        raise ValueError(f"unexpected execution grid: {actual!r}")
