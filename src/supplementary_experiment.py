from __future__ import annotations

import json
import hashlib
import io
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

# CUDA deterministic algorithms require this workspace contract before PyTorch
# creates a cuBLAS handle.  Child evaluation processes inherit the setting.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from hazardous_waste_model import ModelParams
from sample_params import params_from_json_data, params_to_json_data
from src.heuristics import build_greedy_initial_plan
from src.instance_generator import generate_random_params
from src.reproducibility import (
    deserialize_plan,
    environment_snapshot,
    file_sha256,
    git_status_snapshot,
    json_sha256,
    plan_sha256,
    plan_to_canonical_data,
)
from src.solution_utils import evaluate_solution, route_plan_to_solution


CELL_SCHEMA = "supplementary-cell-v1"
SOLUTION_SCHEMA = "supplementary-solution-v1"
MANIFEST_SCHEMA = "supplementary-manifest-v1"
SCHEDULE_SCHEMA = "supplementary-schedule-v1"

SUMMARY_TABLE_NAMES = (
    "raw_results.csv",
    "instance_level_results.csv",
    "summary_by_preference.csv",
    "paired_ppo_vs_ga.csv",
    "paired_ppo_vs_ga_summary.csv",
    "milp_gap_by_instance.csv",
    "milp_gap_summary.csv",
    "generalization_matrix.csv",
    "generalization_operational.csv",
    "cell_file_ledger.csv",
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = ROOT / "configs" / "supplementary_experiment.json"

SOURCE_FILES = (
    "hazardous_waste_model.py",
    "sample_params.py",
    "run_supplementary_experiments.py",
    "configs/supplementary_experiment.json",
    "src/heuristics.py",
    "src/instance_generator.py",
    "src/operators.py",
    "src/ppo_improver.py",
    "src/reproducibility.py",
    "src/solution_utils.py",
    "src/supplementary_experiment.py",
    "src/supplementary_protocol.py",
    "遗传算法/genetic_algorithm.py",
)


class ArtifactIntegrityError(RuntimeError):
    """Raised when a resumable artifact no longer matches its locked hash."""


def atomic_write_json(path: str | Path, data: Any) -> None:
    """Write one JSON document and make it visible only after a full flush."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _encode_raw(raw: Mapping[tuple[Any, ...], Any]) -> list[dict[str, Any]]:
    return [
        {"key": list(key), "value": value}
        for key, value in sorted(raw.items(), key=lambda item: repr(item[0]))
    ]


def _decode_raw(items: list[Mapping[str, Any]]) -> Dict[tuple[Any, ...], Any]:
    return {tuple(item["key"]): item["value"] for item in items}


def serialize_solution(solution: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert tuple-keyed solver output to lossless JSON-compatible data."""

    record: Dict[str, Any] = {
        "schema_version": SOLUTION_SCHEMA,
        "raw": _encode_raw(solution.get("raw", {})),
        "summary": solution.get("summary", {}),
    }
    if "plan" in solution:
        record["plan"] = plan_to_canonical_data(solution["plan"])
    if "routes" in solution:
        record["routes"] = [
            {
                "vehicle": key[0],
                "period": int(key[1]),
                "route": list(route),
            }
            for key, route in sorted(
                solution["routes"].items(), key=lambda item: (item[0][1], item[0][0])
            )
        ]
    return record


def deserialize_solution(data: Mapping[str, Any]) -> Dict[str, Any]:
    if data.get("schema_version") != SOLUTION_SCHEMA:
        raise ValueError("unsupported supplementary solution schema")
    solution: Dict[str, Any] = {
        "raw": _decode_raw(list(data.get("raw", []))),
        "summary": dict(data.get("summary", {})),
    }
    if "plan" in data:
        solution["plan"] = deserialize_plan(data["plan"])
    if "routes" in data:
        solution["routes"] = {
            (str(item["vehicle"]), int(item["period"])): list(item["route"])
            for item in data["routes"]
        }
    return solution


def terminal_inventory_clear(
    params: ModelParams,
    solution: Mapping[str, Any],
    tolerance: float = 1e-5,
) -> bool:
    raw = solution.get("raw", solution)
    if not raw:
        return False
    last_period = params.periods[-1]
    producer_clear = all(
        math.isclose(
            float(raw.get(("IG", node, last_period), 0.0)),
            0.0,
            abs_tol=tolerance,
        )
        for node in params.pickup_nodes
    )
    facility_clear = all(
        math.isclose(
            float(raw.get(("ID", facility, waste, last_period), 0.0)),
            0.0,
            abs_tol=tolerance,
        )
        for facility in params.facilities
        for waste in params.waste_types
    )
    return producer_clear and facility_clear


def commit_cell(path: str | Path, record: Mapping[str, Any]) -> None:
    if record.get("schema_version") != CELL_SCHEMA:
        raise ValueError(f"cell record must use schema {CELL_SCHEMA!r}")
    if record.get("status") not in {"complete", "failed"}:
        raise ValueError("cell status must be complete or failed")
    atomic_write_json(path, dict(record))


def load_valid_completed_cell(
    run_dir: str | Path,
    cell_path: str | Path,
    expected_input_hashes: Mapping[str, str],
) -> Dict[str, Any] | None:
    """Return a committed cell for resume, or fail on any evidence of tampering."""

    root = Path(run_dir)
    path = Path(cell_path)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"invalid cell record: {path}") from exc
    if record.get("schema_version") != CELL_SCHEMA:
        raise ArtifactIntegrityError(f"unsupported cell schema: {path}")
    if record.get("status") != "complete":
        return None
    if record.get("input_hashes") != dict(expected_input_hashes):
        raise ArtifactIntegrityError(f"locked input hash mismatch for {path.name}")
    for path_key, hash_key in (
        ("instance_file", "instance_file_sha256"),
        ("initial_solution_file", "initial_solution_file_sha256"),
        ("checkpoint_file", "checkpoint_sha256"),
    ):
        if hash_key not in expected_input_hashes:
            continue
        relative_input = record.get(path_key)
        if not relative_input:
            raise ArtifactIntegrityError(
                f"complete cell lacks locked input path {path_key}: {path.name}"
            )
        input_path = root / str(relative_input)
        if not input_path.is_file() or file_sha256(input_path) != expected_input_hashes[hash_key]:
            raise ArtifactIntegrityError(f"locked input artifact changed: {relative_input}")
    relative_solution = record.get("solution_file")
    expected_solution_hash = record.get("solution_file_sha256")
    if not relative_solution or not expected_solution_hash:
        raise ArtifactIntegrityError(f"complete cell lacks solution identity: {path.name}")
    solution_path = root / str(relative_solution)
    if not solution_path.is_file():
        raise ArtifactIntegrityError(f"missing solution artifact: {relative_solution}")
    if file_sha256(solution_path) != expected_solution_hash:
        raise ArtifactIntegrityError(f"solution hash mismatch for {path.name}")
    relative_trace = record.get("trace_file")
    expected_trace_hash = record.get("trace_file_sha256")
    if relative_trace or expected_trace_hash:
        if not relative_trace or not expected_trace_hash:
            raise ArtifactIntegrityError(f"incomplete trace identity for {path.name}")
        trace_path = root / str(relative_trace)
        if not trace_path.is_file() or file_sha256(trace_path) != expected_trace_hash:
            raise ArtifactIntegrityError(f"trace hash mismatch for {path.name}")
    return record


def _read_json(path: str | Path) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return data


def _utc_now() -> str:
    return datetime.now().astimezone().isoformat()


def _effective_smoke_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a conspicuously non-publication protocol for integration testing."""

    effective = json.loads(json.dumps(config))
    effective["protocol_id"] = f"{effective['protocol_id']}-smoke"
    effective["output_dir"] = "outputs/supplementary_experiment_smoke"
    for item in effective["test_scales"].values():
        item["instances"] = 1
    effective["preferences"] = [effective["preferences"][2]]
    for scope in effective["methods"].values():
        scope["restarts"] = 1
    ppo = effective["algorithm"]["ppo"]
    ppo.update(
        {
            "train_iterations": 1,
            "num_parallel_episodes": 2,
            "episode_steps": 2,
            "update_epochs": 1,
            "eval_steps": 2,
            "eval_candidate_samples": 2,
            "evaluation_restarts": 1,
        }
    )
    for ga in effective["algorithm"]["ga_by_scale"].values():
        ga.update(
            {
                "population_size": 6,
                "generations": 2,
                "elite_size": 2,
                "tournament_size": 2,
            }
        )
    effective["algorithm"]["milp_time_limit_seconds"] = 5.0
    return effective


def resolve_run_dir(
    config: Mapping[str, Any],
    run_dir: str | Path | None = None,
) -> Path:
    if run_dir is not None:
        return Path(run_dir).resolve()
    configured = Path(str(config["output_dir"]))
    return (ROOT / configured).resolve() if not configured.is_absolute() else configured.resolve()


def _source_hashes() -> Dict[str, str | None]:
    return {
        relative: file_sha256(ROOT / relative) if (ROOT / relative).is_file() else None
        for relative in SOURCE_FILES
    }


def _expected_source_bundle_sha256(manifest: Mapping[str, Any]) -> str:
    return str(
        manifest.get("source_bundle_sha256_at_init")
        or json_sha256(manifest["source_files_sha256_at_init"])
    )


def _assert_source_lock(manifest: Mapping[str, Any]) -> str:
    current = _source_hashes()
    expected_files = manifest["source_files_sha256_at_init"]
    changes = {
        name: {"at_init": digest, "now": current.get(name)}
        for name, digest in expected_files.items()
        if digest != current.get(name)
    }
    if changes:
        raise ArtifactIntegrityError(
            "source files changed after run initialization: "
            + ", ".join(sorted(changes))
        )
    digest = json_sha256(current)
    if digest != _expected_source_bundle_sha256(manifest):
        raise ArtifactIntegrityError("source bundle hash no longer matches initialization")
    return digest


def _invalidate_verification(manifest: Dict[str, Any]) -> None:
    manifest.pop("verification", None)


def _cell_file_ledger(
    root: Path,
    schedule: Sequence[Mapping[str, Any]],
) -> tuple[list[Dict[str, Any]], str]:
    rows: list[Dict[str, Any]] = []
    for entry in schedule:
        path = root / "cells" / f"{entry['cell_id']}.json"
        if path.is_file():
            record = _read_json(path)
            status = record.get("status")
            digest = file_sha256(path)
        else:
            status = "missing"
            digest = None
        rows.append(
            {
                "cell_id": entry["cell_id"],
                "status": status,
                "cell_file_sha256": digest,
            }
        )
    return rows, json_sha256(rows)


def _table_integrity(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[Dict[str, str], list[str]]:
    recorded = manifest.get("tables")
    errors: list[str] = []
    observed: Dict[str, str] = {}
    if not isinstance(recorded, Mapping):
        return observed, ["manifest has no summarized table registry"]
    if set(recorded) != set(SUMMARY_TABLE_NAMES):
        errors.append("summarized table registry does not contain the exact expected files")
    for name in SUMMARY_TABLE_NAMES:
        item = recorded.get(name)
        if not isinstance(item, Mapping):
            errors.append(f"missing summarized table record: {name}")
            continue
        path = root / str(item.get("file", ""))
        if not path.is_file():
            errors.append(f"missing summarized table file: {name}")
            continue
        digest = file_sha256(path)
        observed[name] = digest
        if digest != item.get("file_sha256"):
            errors.append(f"summarized table hash mismatch: {name}")
    return observed, errors


def initialize_run(
    protocol_path: str | Path = DEFAULT_PROTOCOL_PATH,
    run_dir: str | Path | None = None,
    *,
    smoke: bool = False,
) -> Path:
    """Create the immutable protocol and the two disjoint training instances."""

    from src.supplementary_protocol import (
        capacity_preflight,
        derive_protocol_seed,
        generator_config_for_scale,
        load_protocol_config,
        validate_protocol,
    )

    source_config = load_protocol_config(protocol_path)
    validate_protocol(source_config)
    config = _effective_smoke_config(source_config) if smoke else source_config
    capacity = capacity_preflight(config)
    if not all(bool(item["fits"]) for item in capacity.values()):
        raise ValueError(f"network capacity preflight failed: {capacity!r}")
    destination = resolve_run_dir(config, run_dir)
    manifest_path = destination / "manifest.json"
    protocol_hash = json_sha256(config)
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("protocol_sha256") != protocol_hash:
            raise ArtifactIntegrityError("existing run directory uses a different protocol")
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(
            f"refusing to initialize a non-empty directory without a manifest: {destination}"
        )

    for folder in (
        "training_instances",
        "test_instances",
        "initial_solutions",
        "models",
        "training_history",
        "solutions",
        "traces",
        "cells",
        "attempts",
        "tables",
        "verification",
    ):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    atomic_write_json(destination / "protocol_config.json", config)

    training_instances: Dict[str, Dict[str, Any]] = {}
    for model in config["models"]:
        model_id = str(model["id"])
        scale = str(model["train_scale"])
        seed = derive_protocol_seed(config, "training_instance", model_id, scale)
        params = generate_random_params(generator_config_for_scale(config, scale), seed)
        instance_path = destination / "training_instances" / f"{model_id}.json"
        atomic_write_json(instance_path, params_to_json_data(params))
        training_instances[model_id] = {
            "model_id": model_id,
            "scale": scale,
            "seed": seed,
            "file": instance_path.relative_to(destination).as_posix(),
            "file_sha256": file_sha256(instance_path),
            "instance_sha256": json_sha256(params_to_json_data(params)),
        }

    source_hashes = _source_hashes()
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_id": config["protocol_id"],
        "protocol_sha256": protocol_hash,
        "smoke": smoke,
        "created_at": _utc_now(),
        "stage": "initialized",
        "training_scope": "one-instance-per-model; frozen held-out evaluation",
        "seed_scheme": "sha256-seed-v1",
        "capacity_preflight": capacity,
        "source_files_sha256_at_init": source_hashes,
        "source_bundle_sha256_at_init": json_sha256(source_hashes),
        "git_at_init": git_status_snapshot(ROOT),
        "environment": environment_snapshot(),
        "training_instances": training_instances,
        "models": {},
        "test_sets": {},
        "schedule": None,
    }
    atomic_write_json(manifest_path, manifest)
    return destination


def _load_run(run_dir: str | Path) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    root = Path(run_dir).resolve()
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported supplementary manifest schema")
    config = _read_json(root / "protocol_config.json")
    if json_sha256(config) != manifest.get("protocol_sha256"):
        raise ArtifactIntegrityError("protocol_config.json no longer matches manifest")
    return root, config, manifest


def _load_locked_file(root: Path, record: Mapping[str, Any]) -> Dict[str, Any]:
    path = root / str(record["file"])
    if file_sha256(path) != record["file_sha256"]:
        raise ArtifactIntegrityError(f"locked artifact changed: {record['file']}")
    return _read_json(path)


def _policy_state_sha256(policy: Any) -> str:
    digest = hashlib.sha256()
    state = policy.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _csv_text(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    import csv

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _atomic_write_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def train_models(
    run_dir: str | Path,
    models: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Train each requested model once; completed hash-valid checkpoints are reused."""

    import torch

    from src.ppo_improver import PPOImprover
    from src.supplementary_protocol import derive_protocol_seed

    root, config, manifest = _load_run(run_dir)
    _invalidate_verification(manifest)
    source_bundle_sha256 = _assert_source_lock(manifest)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    selected = set(models or [str(item["id"]) for item in config["models"]])
    known = {str(item["id"]): item for item in config["models"]}
    unknown = selected - set(known)
    if unknown:
        raise ValueError(f"unknown model ids: {sorted(unknown)!r}")

    for model_id in [str(item["id"]) for item in config["models"] if str(item["id"]) in selected]:
        existing = manifest["models"].get(model_id)
        if existing:
            checkpoint = root / existing["checkpoint_file"]
            if checkpoint.is_file() and file_sha256(checkpoint) == existing["checkpoint_sha256"]:
                continue
            raise ArtifactIntegrityError(f"recorded checkpoint changed: {model_id}")

        model_commit_path = root / "models" / f"{model_id}.record.json"
        model_prepared_path = root / "models" / f".{model_id}.prepared.json"
        if model_commit_path.is_file():
            recovered = _read_json(model_commit_path)
            checkpoint = root / str(recovered["checkpoint_file"])
            history_path = root / str(recovered["history_file"])
            if (
                recovered.get("protocol_sha256") != manifest["protocol_sha256"]
                or recovered.get("source_bundle_sha256") != source_bundle_sha256
                or not checkpoint.is_file()
                or file_sha256(checkpoint) != recovered.get("checkpoint_sha256")
                or not history_path.is_file()
                or file_sha256(history_path) != recovered.get("history_sha256")
            ):
                raise ArtifactIntegrityError(f"invalid model commit record: {model_id}")
            recovered.pop("protocol_sha256", None)
            manifest["models"][model_id] = recovered
            atomic_write_json(root / "manifest.json", manifest)
            model_prepared_path.unlink(missing_ok=True)
            continue

        train_record = manifest["training_instances"][model_id]
        params = params_from_json_data(_load_locked_file(root, train_record))
        train_seed = derive_protocol_seed(config, "training_algorithm", model_id)
        checkpoint = root / "models" / str(known[model_id]["checkpoint"])
        temporary_checkpoint = checkpoint.with_name(f".{checkpoint.name}.training.tmp")
        if model_prepared_path.is_file():
            prepared = _read_json(model_prepared_path)
            history_path = root / str(prepared["history_file"])
            if (
                prepared.get("protocol_sha256") != manifest["protocol_sha256"]
                or prepared.get("source_bundle_sha256") != source_bundle_sha256
                or not history_path.is_file()
                or file_sha256(history_path) != prepared.get("history_sha256")
            ):
                raise ArtifactIntegrityError(f"invalid prepared model record: {model_id}")
            expected_checkpoint_hash = prepared.get("checkpoint_sha256")
            if checkpoint.is_file():
                if file_sha256(checkpoint) != expected_checkpoint_hash:
                    raise ArtifactIntegrityError(
                        f"prepared checkpoint changed: {model_id}"
                    )
                temporary_checkpoint.unlink(missing_ok=True)
            elif (
                temporary_checkpoint.is_file()
                and file_sha256(temporary_checkpoint) == expected_checkpoint_hash
            ):
                temporary_checkpoint.replace(checkpoint)
            else:
                raise ArtifactIntegrityError(
                    f"prepared checkpoint is unavailable or invalid: {model_id}"
                )
            recovered = dict(prepared)
            recovered.pop("protocol_sha256", None)
            atomic_write_json(
                model_commit_path,
                {"protocol_sha256": manifest["protocol_sha256"], **recovered},
            )
            model_prepared_path.unlink(missing_ok=True)
            manifest["models"][model_id] = recovered
            atomic_write_json(root / "manifest.json", manifest)
            continue
        if checkpoint.exists():
            raise ArtifactIntegrityError(
                f"orphan checkpoint exists without a manifest record: {checkpoint}"
            )
        improver = PPOImprover(
            params,
            {"ppo": config["algorithm"]["ppo"]},
            config["network"],
            seed=train_seed,
        )
        temporary_checkpoint.unlink(missing_ok=True)
        started = time.perf_counter()
        history = improver.train(save_path=temporary_checkpoint)
        if improver.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        history_rows = [
            {
                "iteration": index,
                **{name: values[index] for name, values in history.items()},
            }
            for index in range(len(next(iter(history.values()), [])))
        ]
        history_path = root / "training_history" / f"{model_id}.csv"
        history_fields = ["iteration", *history.keys()]
        _atomic_write_text(history_path, _csv_text(history_rows, history_fields))
        model_record = {
            "model_id": model_id,
            "train_scale": known[model_id]["train_scale"],
            "training_instance_file_sha256": train_record["file_sha256"],
            "training_instance_sha256": train_record["instance_sha256"],
            "training_seed": train_seed,
            "checkpoint_file": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": file_sha256(temporary_checkpoint),
            "policy_state_sha256": _policy_state_sha256(improver.policy),
            "source_bundle_sha256": source_bundle_sha256,
            "training_seconds": elapsed,
            "history_file": history_path.relative_to(root).as_posix(),
            "history_sha256": file_sha256(history_path),
            "frozen_at": _utc_now(),
        }
        atomic_write_json(
            model_prepared_path,
            {"protocol_sha256": manifest["protocol_sha256"], **model_record},
        )
        temporary_checkpoint.replace(checkpoint)
        atomic_write_json(
            model_commit_path,
            {"protocol_sha256": manifest["protocol_sha256"], **model_record},
        )
        model_prepared_path.unlink(missing_ok=True)
        manifest["models"][model_id] = model_record
        manifest["stage"] = (
            "models_frozen"
            if set(manifest["models"]) == set(known)
            else "training_in_progress"
        )
        atomic_write_json(root / "manifest.json", manifest)
        _assert_source_lock(manifest)
        del improver
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    manifest["stage"] = (
        "models_frozen"
        if set(manifest["models"]) == set(known)
        else "training_in_progress"
    )
    atomic_write_json(root / "manifest.json", manifest)
    return dict(manifest["models"])


def lock_test_sets(run_dir: str | Path) -> Dict[str, Any]:
    """Generate the four held-out test sets only after both checkpoints are frozen."""

    from src.supplementary_protocol import derive_protocol_seed, generator_config_for_scale

    root, config, manifest = _load_run(run_dir)
    _invalidate_verification(manifest)
    _assert_source_lock(manifest)
    expected_models = {str(item["id"]) for item in config["models"]}
    if set(manifest["models"]) != expected_models:
        raise RuntimeError("both frozen models must exist before test sets are locked")
    for record in manifest["models"].values():
        checkpoint = root / record["checkpoint_file"]
        if file_sha256(checkpoint) != record["checkpoint_sha256"]:
            raise ArtifactIntegrityError(f"checkpoint changed before test locking: {checkpoint}")
    if manifest["test_sets"]:
        for scale, records in manifest["test_sets"].items():
            expected_count = int(config["test_scales"][scale]["instances"])
            if len(records) != expected_count:
                raise ArtifactIntegrityError(
                    f"manifest contains a partial locked test scale: {scale}"
                )
            for record in records:
                _load_locked_file(root, record)
                initial_path = root / record["initial_solution_file"]
                if file_sha256(initial_path) != record["initial_solution_file_sha256"]:
                    raise ArtifactIntegrityError(
                        f"locked initial solution changed: {record['initial_solution_file']}"
                    )
    test_sets: Dict[str, list[Dict[str, Any]]] = dict(manifest["test_sets"])
    for scale in config["test_scales"]:
        if scale in test_sets:
            continue
        records: list[Dict[str, Any]] = []
        count = int(config["test_scales"][scale]["instances"])
        for index in range(count):
            instance_id = f"{scale}-I{index + 1:03d}"
            instance_seed = derive_protocol_seed(config, "test_instance", scale, index)
            initial_seed = derive_protocol_seed(config, "initial_solution", scale, index)
            params = generate_random_params(
                generator_config_for_scale(config, scale), instance_seed
            )
            instance_path = root / "test_instances" / scale / f"{index:03d}.json"
            atomic_write_json(instance_path, params_to_json_data(params))
            started = time.perf_counter()
            plan = build_greedy_initial_plan(params, seed=initial_seed)
            solution = route_plan_to_solution(params, plan)
            initialization_seconds = time.perf_counter() - started
            initial_metrics = evaluate_solution(params, solution, (0.5, 0.5))
            if not initial_metrics["feasible"]:
                raise RuntimeError(
                    f"strictly feasible initial solution not found for {instance_id}: "
                    f"{initial_metrics['violations']}"
                )
            refs = {
                "cost": max(float(initial_metrics["cost"]), 1e-9),
                "risk": max(float(initial_metrics["risk"]), 1e-9),
            }
            initial_path = root / "initial_solutions" / scale / f"{index:03d}.json"
            initial_data = {
                "schema_version": "supplementary-initial-solution-v1",
                "instance_id": instance_id,
                "seed": initial_seed,
                "plan": plan_to_canonical_data(plan),
                "plan_sha256": plan_sha256(plan),
                "objective_refs": refs,
                "metrics": initial_metrics,
            }
            atomic_write_json(initial_path, initial_data)
            records.append(
                {
                    "instance_id": instance_id,
                    "index": index,
                    "scale": scale,
                    "seed": instance_seed,
                    "file": instance_path.relative_to(root).as_posix(),
                    "file_sha256": file_sha256(instance_path),
                    "instance_sha256": json_sha256(params_to_json_data(params)),
                    "initial_solution_seed": initial_seed,
                    "initial_solution_file": initial_path.relative_to(root).as_posix(),
                    "initial_solution_file_sha256": file_sha256(initial_path),
                    "initial_plan_sha256": initial_data["plan_sha256"],
                    "objective_refs_sha256": json_sha256(refs),
                    "initialization_seconds": initialization_seconds,
                }
            )
        test_sets[scale] = records
        manifest["test_sets"] = test_sets
        atomic_write_json(root / "manifest.json", manifest)
    manifest["tests_locked_at"] = _utc_now()
    manifest["stage"] = "test_sets_locked"
    atomic_write_json(root / "manifest.json", manifest)
    return test_sets


def prepare_schedule(run_dir: str | Path) -> list[Dict[str, Any]]:
    """Bind the logical protocol grid to immutable model/instance artifacts."""

    from src.supplementary_protocol import build_cell_id, enumerate_cells

    root, config, manifest = _load_run(run_dir)
    if not manifest["test_sets"]:
        raise RuntimeError("test sets must be locked before scheduling")
    if set(manifest["models"]) != {str(item["id"]) for item in config["models"]}:
        raise RuntimeError("both frozen checkpoints are required before scheduling")
    schedule_path = root / "schedule.json"
    if manifest.get("schedule"):
        record = manifest["schedule"]
        if file_sha256(root / record["file"]) != record["file_sha256"]:
            raise ArtifactIntegrityError("schedule.json changed after it was locked")
        data = _read_json(root / record["file"])
        return list(data["cells"])
    source_bundle_sha256 = _assert_source_lock(manifest)

    initial_solution_ids = {
        (scale, int(record["index"])): str(record["initial_plan_sha256"])
        for scale, records in manifest["test_sets"].items()
        for record in records
    }
    bound_cells: list[Dict[str, Any]] = []
    for cell in enumerate_cells(config, initial_solution_ids=initial_solution_ids):
        logical_id = build_cell_id(cell)
        instance = manifest["test_sets"][cell.test_scale][cell.instance_index]
        input_hashes: Dict[str, str] = {
            "protocol_sha256": manifest["protocol_sha256"],
            "instance_file_sha256": instance["file_sha256"],
            "initial_solution_file_sha256": instance["initial_solution_file_sha256"],
            "initial_plan_sha256": instance["initial_plan_sha256"],
            "objective_refs_sha256": instance["objective_refs_sha256"],
            "source_bundle_sha256": source_bundle_sha256,
        }
        if cell.method == "ppo":
            model_record = manifest["models"][cell.model_id]
            input_hashes.update(
                {
                    "checkpoint_sha256": model_record["checkpoint_sha256"],
                    "policy_state_sha256": model_record["policy_state_sha256"],
                }
            )
            checkpoint_file = model_record["checkpoint_file"]
        else:
            checkpoint_file = None
        compact_id = "cell-" + json_sha256(
            {"logical_cell_id": logical_id, "input_hashes": input_hashes}
        )[:32]
        views: list[str] = []
        if cell.test_scale == "Test-1" and (
            cell.method in {"heuristic", "ga", "milp"}
            or (cell.method == "ppo" and cell.model_id == "Train-S")
        ):
            views.append("E1")
        if cell.test_scale == "Test-4" and (
            cell.method in {"heuristic", "ga"}
            or (cell.method == "ppo" and cell.model_id == "Train-L")
        ):
            views.append("E2")
        if cell.method in {"ppo", "ga"} or (
            cell.method == "milp" and cell.test_scale == "Test-1"
        ):
            views.append("G")
        bound_cells.append(
            {
                "cell_id": compact_id,
                "logical_cell_id": logical_id,
                "cell": asdict(cell),
                "views": views,
                "instance_file": instance["file"],
                "initial_solution_file": instance["initial_solution_file"],
                "initialization_seconds": instance.get("initialization_seconds", 0.0),
                "checkpoint_file": checkpoint_file,
                "input_hashes": input_hashes,
            }
        )
    if len({item["cell_id"] for item in bound_cells}) != len(bound_cells):
        raise RuntimeError("compact bound-cell hash collision")
    schedule = {
        "schema_version": SCHEDULE_SCHEMA,
        "protocol_sha256": manifest["protocol_sha256"],
        "created_at": _utc_now(),
        "cells": bound_cells,
    }
    atomic_write_json(schedule_path, schedule)
    manifest["schedule"] = {
        "file": schedule_path.relative_to(root).as_posix(),
        "file_sha256": file_sha256(schedule_path),
        "cell_count": len(bound_cells),
    }
    manifest["stage"] = "scheduled"
    atomic_write_json(root / "manifest.json", manifest)
    return bound_cells


def _load_instance_and_initial(
    root: Path, entry: Mapping[str, Any]
) -> tuple[ModelParams, Dict[str, Any], Any, tuple[float, float]]:
    instance_path = root / str(entry["instance_file"])
    initial_path = root / str(entry["initial_solution_file"])
    hashes = entry["input_hashes"]
    if file_sha256(instance_path) != hashes["instance_file_sha256"]:
        raise ArtifactIntegrityError(f"instance changed for {entry['cell_id']}")
    if file_sha256(initial_path) != hashes["initial_solution_file_sha256"]:
        raise ArtifactIntegrityError(f"initial solution changed for {entry['cell_id']}")
    params = params_from_json_data(_read_json(instance_path))
    initial = _read_json(initial_path)
    plan = deserialize_plan(initial["plan"])
    if plan_sha256(plan) != hashes["initial_plan_sha256"]:
        raise ArtifactIntegrityError(f"initial plan changed for {entry['cell_id']}")
    refs_data = initial["objective_refs"]
    refs = (float(refs_data["cost"]), float(refs_data["risk"]))
    if json_sha256(refs_data) != hashes["objective_refs_sha256"]:
        raise ArtifactIntegrityError(f"objective references changed for {entry['cell_id']}")
    return params, initial, plan, refs


def _preference(entry: Mapping[str, Any]) -> tuple[float, float]:
    cell = entry["cell"]
    return float(cell["cost_weight"]), float(cell["risk_weight"])


def _with_normalized_objective(
    params: ModelParams,
    preference: tuple[float, float],
    objective_refs: tuple[float, float],
) -> ModelParams:
    return replace(
        params,
        cost_weight=preference[0] / max(objective_refs[0], 1e-9),
        risk_weight=preference[1] / max(objective_refs[1], 1e-9),
    )


def _initial_chromosome(params: ModelParams, plan: Mapping[tuple[str, int], Sequence[str]]) -> list[str]:
    ordered = [
        node
        for key in sorted(plan, key=lambda item: (item[1], item[0]))
        for node in plan[key][1:-1]
        if node in params.pickup_nodes
    ]
    unique = list(dict.fromkeys(ordered))
    if sorted(unique) != sorted(params.pickup_nodes):
        raise ValueError("locked initial plan does not contain every pickup node exactly once")
    return unique


def _cuda_sync_if_needed(device: Any | None = None) -> None:
    import torch

    if torch.cuda.is_available() and (device is None or getattr(device, "type", None) == "cuda"):
        torch.cuda.synchronize()


def _quality_metrics(
    params: ModelParams,
    solution: Mapping[str, Any],
    preference: tuple[float, float],
    refs: tuple[float, float],
) -> Dict[str, Any]:
    metrics = evaluate_solution(params, solution, preference, objective_refs=refs)
    metrics["terminal_inventory_clear"] = terminal_inventory_clear(params, solution)
    metrics["quality_included"] = bool(metrics["feasible"])
    if not metrics["feasible"]:
        for field in (
            "cost",
            "risk",
            "normalized_cost",
            "normalized_risk",
            "weighted_objective_raw",
            "weighted_objective_normalized",
            "weighted_objective",
            "fixed_cost",
            "distance_cost",
            "processing_cost",
            "transport_risk",
            "coload_risk",
            "producer_inventory_risk",
            "facility_inventory_risk",
        ):
            metrics[field] = None
    return metrics


def _commit_success(
    root: Path,
    entry: Mapping[str, Any],
    solution: Mapping[str, Any],
    trace: Any,
    metrics: Mapping[str, Any],
    runtime_seconds: float,
    solver_status: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    cell_id = str(entry["cell_id"])
    solution_path = root / "solutions" / f"{cell_id}.json"
    trace_path = root / "traces" / f"{cell_id}.json"
    atomic_write_json(solution_path, serialize_solution(solution))
    atomic_write_json(
        trace_path,
        {
            "schema_version": "supplementary-trace-v1",
            "cell_id": cell_id,
            "trace": trace,
        },
    )
    record = {
        "schema_version": CELL_SCHEMA,
        "status": "complete",
        "cell_id": cell_id,
        "logical_cell_id": entry["logical_cell_id"],
        "cell": entry["cell"],
        "views": entry["views"],
        "input_hashes": entry["input_hashes"],
        "instance_file": entry["instance_file"],
        "initial_solution_file": entry["initial_solution_file"],
        "checkpoint_file": entry.get("checkpoint_file"),
        "solution_file": solution_path.relative_to(root).as_posix(),
        "solution_file_sha256": file_sha256(solution_path),
        "trace_file": trace_path.relative_to(root).as_posix(),
        "trace_file_sha256": file_sha256(trace_path),
        "result_plan_sha256": (
            plan_sha256(solution["plan"]) if solution.get("plan") is not None else None
        ),
        "solver_status": solver_status,
        "runtime_seconds": float(runtime_seconds),
        "execution_source_bundle_sha256": entry["input_hashes"][
            "source_bundle_sha256"
        ],
        "metrics": dict(metrics),
        "completed_at": _utc_now(),
    }
    if extra:
        record.update(dict(extra))
    commit_cell(root / "cells" / f"{cell_id}.json", record)


def _commit_failure(
    root: Path,
    entry: Mapping[str, Any],
    error: BaseException,
    runtime_seconds: float,
    failure_stage: str,
) -> None:
    record = {
        "schema_version": CELL_SCHEMA,
        "status": "failed",
        "cell_id": entry["cell_id"],
        "logical_cell_id": entry["logical_cell_id"],
        "cell": entry["cell"],
        "views": entry["views"],
        "input_hashes": entry["input_hashes"],
        "instance_file": entry.get("instance_file"),
        "initial_solution_file": entry.get("initial_solution_file"),
        "checkpoint_file": entry.get("checkpoint_file"),
        "runtime_seconds": float(runtime_seconds),
        "failure_stage": failure_stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": "".join(traceback.format_exception(error)),
        "execution_source_bundle_sha256": entry["input_hashes"][
            "source_bundle_sha256"
        ],
        "failed_at": _utc_now(),
    }
    commit_cell(root / "cells" / f"{entry['cell_id']}.json", record)


def _execute_ppo_group(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    from src.ppo_improver import PPOImprover

    first = entries[0]
    model_id = str(first["cell"]["model_id"])
    model = manifest["models"][model_id]
    checkpoint = root / model["checkpoint_file"]
    if file_sha256(checkpoint) != model["checkpoint_sha256"]:
        raise ArtifactIntegrityError(f"checkpoint changed: {model_id}")
    params, _, initial_plan, refs = _load_instance_and_initial(root, first)
    improver = PPOImprover.from_frozen_checkpoint(
        params,
        checkpoint,
        algorithm_config={"ppo": config["algorithm"]["ppo"]},
        network_config=config["network"],
        seed=int(first["cell"]["seed"]),
    )
    before_state = (
        improver.policy_state_sha256()
        if hasattr(improver, "policy_state_sha256")
        else _policy_state_sha256(improver.policy)
    )
    if before_state != model["policy_state_sha256"]:
        raise ArtifactIntegrityError(f"loaded policy state mismatch: {model_id}")
    completed = failed = 0
    for entry in entries:
        attempt_started = time.perf_counter()
        elapsed: float | None = None
        failure_stage = "prepare"
        try:
            preference = _preference(entry)
            _cuda_sync_if_needed(improver.device)
            started = time.perf_counter()
            failure_stage = "solve"
            result = improver.improve(
                preference,
                seed=int(entry["cell"]["seed"]),
                initial_plan=initial_plan,
            )
            _cuda_sync_if_needed(improver.device)
            elapsed = time.perf_counter() - started
            failure_stage = "validate_and_commit"
            metrics = _quality_metrics(params, result.solution, preference, refs)
            after_state = (
                improver.policy_state_sha256()
                if hasattr(improver, "policy_state_sha256")
                else _policy_state_sha256(improver.policy)
            )
            if after_state != before_state:
                raise RuntimeError("frozen policy state changed during inference")
            checkpoint_hash_after = file_sha256(checkpoint)
            if checkpoint_hash_after != model["checkpoint_sha256"]:
                raise ArtifactIntegrityError("checkpoint file changed during inference")
            _commit_success(
                root,
                entry,
                result.solution,
                result.trace,
                metrics,
                elapsed,
                "completed",
                {
                    "policy_state_sha256_before": before_state,
                    "policy_state_sha256_after": after_state,
                    "checkpoint_sha256_after": checkpoint_hash_after,
                },
            )
            completed += 1
        except Exception as exc:
            if elapsed is None:
                try:
                    _cuda_sync_if_needed(improver.device)
                except Exception:
                    pass
                elapsed = time.perf_counter() - attempt_started
            _commit_failure(root, entry, exc, elapsed, failure_stage)
            failed += 1
    del improver
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return completed, failed


def _execute_nonppo_group(
    root: Path,
    config: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    params, initial, initial_plan, refs = _load_instance_and_initial(root, entries[0])
    initial_solution = route_plan_to_solution(params, initial_plan)
    completed = failed = 0
    for entry in entries:
        attempt_started = time.perf_counter()
        elapsed: float | None = None
        failure_stage = "prepare"
        try:
            method = str(entry["cell"]["method"])
            preference = _preference(entry)
            seed = int(entry["cell"]["seed"])
            trace: Any = []
            solver_status = "completed"
            if method == "heuristic":
                solution = initial_solution
                elapsed = float(entry.get("initialization_seconds", 0.0))
            elif method == "ga":
                failure_stage = "solve"
                ga_dir = ROOT / "遗传算法"
                if str(ga_dir) not in sys.path:
                    sys.path.insert(0, str(ga_dir))
                from genetic_algorithm import ClassicGeneticAlgorithm, GAConfig

                optimizer_params = _with_normalized_objective(params, preference, refs)
                ga_config = GAConfig(
                    random_seed=seed,
                    **config["algorithm"]["ga_by_scale"][entry["cell"]["test_scale"]],
                )
                started = time.perf_counter()
                result = ClassicGeneticAlgorithm(
                    optimizer_params,
                    ga_config,
                    initial_chromosome=_initial_chromosome(params, initial_plan),
                    initial_solution=initial_solution,
                ).run()
                elapsed = time.perf_counter() - started
                solution = result.best_solution
                trace = {"best_objective_history": result.history}
                solver_status = "completed" if result.feasible else "infeasible"
            elif method == "milp":
                failure_stage = "solve"
                from hazardous_waste_model import solve_with_milp

                optimizer_params = _with_normalized_objective(params, preference, refs)
                started = time.perf_counter()
                result = solve_with_milp(
                    optimizer_params,
                    time_limit=float(config["algorithm"]["milp_time_limit_seconds"]),
                )
                elapsed = time.perf_counter() - started
                solution = result.solution
                trace = {"diagnostics": result.diagnostics, "solver_objective": result.objective}
                solver_status = result.status
            else:
                raise ValueError(f"unsupported method: {method}")
            failure_stage = "validate_and_commit"
            metrics = _quality_metrics(params, solution, preference, refs)
            _commit_success(
                root,
                entry,
                solution,
                trace,
                metrics,
                elapsed,
                solver_status,
                {"evaluation_seed": seed},
            )
            completed += 1
        except Exception as exc:
            if elapsed is None:
                elapsed = time.perf_counter() - attempt_started
            _commit_failure(root, entry, exc, elapsed, failure_stage)
            failed += 1
    return completed, failed


def _execute_group_worker(
    run_dir: str,
    entries: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    root, config, manifest = _load_run(run_dir)
    source_bundle_sha256 = _assert_source_lock(manifest)
    if entries[0]["input_hashes"].get("source_bundle_sha256") != source_bundle_sha256:
        raise ArtifactIntegrityError("scheduled cell source bundle does not match worker source")
    if entries[0]["cell"]["method"] == "ppo":
        result = _execute_ppo_group(root, config, manifest, entries)
    else:
        result = _execute_nonppo_group(root, config, entries)
    _assert_source_lock(manifest)
    return result


def evaluate_run(
    run_dir: str | Path,
    *,
    methods: Sequence[str] | None = None,
    models: Sequence[str] | None = None,
    scales: Sequence[str] | None = None,
    workers: int = 1,
    max_cells: int | None = None,
    retry_failed: bool = False,
    allow_concurrent_ppo: bool = False,
) -> Dict[str, int]:
    """Execute missing cells, grouping by instance so each checkpoint loads once."""

    if int(workers) < 1:
        raise ValueError("workers must be at least 1")
    if max_cells is not None and int(max_cells) < 0:
        raise ValueError("max_cells must be non-negative")
    evaluation_started_at = _utc_now()
    wall_started = time.perf_counter()
    root, _, manifest = _load_run(run_dir)
    _assert_source_lock(manifest)
    schedule = prepare_schedule(root)
    method_filter = set(methods or [])
    model_filter = set(models or [])
    scale_filter = set(scales or [])
    domains = {
        "methods": {str(entry["cell"]["method"]) for entry in schedule},
        "models": {str(entry["cell"]["model_id"]) for entry in schedule},
        "scales": {str(entry["cell"]["test_scale"]) for entry in schedule},
    }
    for label, supplied, selected in (
        ("methods", methods, method_filter),
        ("models", models, model_filter),
        ("scales", scales, scale_filter),
    ):
        if supplied is not None and not selected:
            raise ValueError(f"{label} filter must not be empty")
        unknown = selected - domains[label]
        if unknown:
            raise ValueError(f"unknown {label} filter values: {sorted(unknown)!r}")
    selected_methods = method_filter or domains["methods"]
    if "ppo" in selected_methods and int(workers) > 1 and not allow_concurrent_ppo:
        raise ValueError(
            "concurrent PPO workers share one GPU; pass allow_concurrent_ppo=True "
            "only for a declared throughput run"
        )
    pending: list[Dict[str, Any]] = []
    skipped = skipped_failed = 0
    for entry in schedule:
        if max_cells is not None and len(pending) >= int(max_cells):
            break
        cell = entry["cell"]
        if method_filter and cell["method"] not in method_filter:
            continue
        if model_filter and cell["model_id"] not in model_filter:
            continue
        if scale_filter and cell["test_scale"] not in scale_filter:
            continue
        cell_path = root / "cells" / f"{entry['cell_id']}.json"
        if cell_path.exists():
            raw = _read_json(cell_path)
            if (
                raw.get("cell_id") != entry["cell_id"]
                or raw.get("logical_cell_id") != entry["logical_cell_id"]
                or raw.get("cell") != entry["cell"]
            ):
                raise ArtifactIntegrityError(
                    f"cell identity changed: {entry['cell_id']}"
                )
            if raw.get("status") == "complete":
                load_valid_completed_cell(root, cell_path, entry["input_hashes"])
                skipped += 1
                continue
            if not retry_failed:
                skipped_failed += 1
                continue
            attempt_path = root / "attempts" / f"{entry['cell_id']}-{_utc_now().replace(':', '-')}.json"
            atomic_write_json(attempt_path, raw)
        pending.append(dict(entry))

    groups: Dict[tuple[str, str, str, int], list[Dict[str, Any]]] = {}
    for entry in pending:
        cell = entry["cell"]
        key = (
            str(cell["method"]),
            str(cell["model_id"]),
            str(cell["test_scale"]),
            int(cell["instance_index"]),
        )
        groups.setdefault(key, []).append(entry)
    completed = failed = 0
    method_order = {"ppo": 0, "ga": 1, "heuristic": 2, "milp": 3}
    ordered_groups = [
        groups[key]
        for key in sorted(
            groups,
            key=lambda key: (method_order.get(key[0], 99), key[1], key[2], key[3]),
        )
    ]
    if int(workers) <= 1 or len(ordered_groups) <= 1:
        for index, group in enumerate(ordered_groups, start=1):
            done, bad = _execute_group_worker(str(root), group)
            completed += done
            failed += bad
            print(
                f"[evaluate] group {index}/{len(ordered_groups)} "
                f"completed={completed} failed={failed}",
                flush=True,
            )
    else:
        context = get_context("spawn")
        pool = ProcessPoolExecutor(max_workers=int(workers), mp_context=context)
        futures = {}
        try:
            futures = {
                pool.submit(_execute_group_worker, str(root), group): group
                for group in ordered_groups
            }
            for index, future in enumerate(as_completed(futures), start=1):
                done, bad = future.result()
                completed += done
                failed += bad
                print(
                    f"[evaluate] group {index}/{len(ordered_groups)} "
                    f"completed={completed} failed={failed}",
                    flush=True,
                )
        except BaseException:
            for future in futures:
                future.cancel()
            processes = list(getattr(pool, "_processes", {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join(timeout=5.0)
            raise
        else:
            pool.shutdown(wait=True)
    _, _, latest_manifest = _load_run(root)
    _invalidate_verification(latest_manifest)
    latest_manifest["last_evaluation_at"] = _utc_now()
    latest_manifest.setdefault("evaluation_runs", []).append(
        {
            "started_at": evaluation_started_at,
            "completed_at": latest_manifest["last_evaluation_at"],
            "methods": sorted(method_filter) if method_filter else "all",
            "models": sorted(model_filter) if model_filter else "all",
            "scales": sorted(scale_filter) if scale_filter else "all",
            "workers": int(workers),
            "allow_concurrent_ppo": bool(allow_concurrent_ppo),
            "max_cells": max_cells,
            "wall_seconds": time.perf_counter() - wall_started,
            "completed_cells": completed,
            "failed_cells": failed,
        }
    )
    latest_manifest["stage"] = "evaluation_in_progress"
    atomic_write_json(root / "manifest.json", latest_manifest)
    return {
        "scheduled": len(schedule),
        "selected_pending": len(pending),
        "completed": completed,
        "failed": failed,
        "skipped_complete": skipped,
        "skipped_failed": skipped_failed,
    }


QUALITY_FIELDS = (
    "cost",
    "risk",
    "normalized_cost",
    "normalized_risk",
    "weighted_objective_raw",
    "weighted_objective_normalized",
    "fixed_cost",
    "distance_cost",
    "processing_cost",
    "transport_risk",
    "coload_risk",
    "producer_inventory_risk",
    "facility_inventory_risk",
)


def _cell_record_to_row(record: Mapping[str, Any]) -> Dict[str, Any]:
    cell = record.get("cell", {})
    metrics = record.get("metrics", {})
    row = {
        "cell_id": record.get("cell_id"),
        "status": record.get("status"),
        "method": cell.get("method"),
        "model_id": cell.get("model_id"),
        "test_scale": cell.get("test_scale"),
        "instance_index": cell.get("instance_index"),
        "preference_id": cell.get("preference_id"),
        "cost_weight": cell.get("cost_weight"),
        "risk_weight": cell.get("risk_weight"),
        "restart_index": cell.get("restart_index"),
        "evaluation_seed": cell.get("seed"),
        "runtime_seconds": record.get("runtime_seconds"),
        "solver_status": record.get("solver_status"),
        "feasible": metrics.get("feasible"),
        "terminal_inventory_clear": metrics.get("terminal_inventory_clear"),
        "quality_included": metrics.get("quality_included"),
        "violation_count": len(metrics.get("violations", [])),
        "violations": json.dumps(metrics.get("violations", []), ensure_ascii=False),
        "instance_file": record.get("instance_file"),
        "initial_solution_file": record.get("initial_solution_file"),
        "solution_file": record.get("solution_file"),
        "result_plan_sha256": record.get("result_plan_sha256"),
        "error_type": record.get("error_type"),
        "error": record.get("error"),
    }
    for field in QUALITY_FIELDS:
        row[field] = metrics.get(field)
    return row


def _write_dataframe(path: Path, frame: Any) -> None:
    _atomic_write_text(path, frame.to_csv(index=False, encoding="utf-8-sig"))


def _instance_level_frame(raw: Any, config: Mapping[str, Any]) -> Any:
    import numpy as np
    import pandas as pd

    keys = ["method", "model_id", "test_scale", "instance_index", "preference_id"]
    expected_restarts = {
        method: int(scope["restarts"]) for method, scope in config["methods"].items()
    }
    rows: list[Dict[str, Any]] = []
    for identity, group in raw.groupby(keys, dropna=False, sort=True):
        method, model_id, scale, instance_index, preference_id = identity
        expected = expected_restarts[str(method)]
        observed = len(group)
        feasible_values = group["feasible"].fillna(False).astype(bool)
        all_quality = (
            observed == expected
            and bool((group["status"] == "complete").all())
            and bool(feasible_values.all())
            and all(group[field].notna().all() for field in QUALITY_FIELDS)
        )
        runtimes = pd.to_numeric(group["runtime_seconds"], errors="coerce")
        complete_runtime = observed == expected and bool(runtimes.notna().all())
        successful_runtimes = runtimes[group["status"] == "complete"].dropna()
        failed_runtimes = runtimes[group["status"] == "failed"].dropna()
        row: Dict[str, Any] = {
            "method": method,
            "model_id": model_id,
            "test_scale": scale,
            "instance_index": int(instance_index),
            "preference_id": preference_id,
            "cost_weight": float(group["cost_weight"].iloc[0]),
            "risk_weight": float(group["risk_weight"].iloc[0]),
            "expected_restarts": expected,
            "observed_restarts": observed,
            "feasible_rate": float(feasible_values.mean()),
            "all_restarts_feasible": bool(feasible_values.all() and observed == expected),
            "terminal_clear_rate": float(
                group["terminal_inventory_clear"].fillna(False).astype(bool).mean()
            ),
            "technical_failure_rate": float((group["status"] == "failed").mean()),
            "missing_rate": float((group["status"] == "missing").mean()),
            "runtime_seconds": float(runtimes.mean()) if complete_runtime else np.nan,
            "successful_runtime_seconds_conditional": (
                float(successful_runtimes.mean()) if len(successful_runtimes) else np.nan
            ),
            "failed_runtime_seconds_conditional": (
                float(failed_runtimes.mean()) if len(failed_runtimes) else np.nan
            ),
            "solver_status": "|".join(
                sorted(
                    set(
                        group.get("solver_status", pd.Series(index=group.index, dtype=object))
                        .dropna()
                        .astype(str)
                    )
                )
            ),
        }
        for field in QUALITY_FIELDS:
            row[field] = float(pd.to_numeric(group[field]).mean()) if all_quality else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_by_preference(instance: Any) -> Any:
    import pandas as pd

    keys = ["method", "model_id", "test_scale", "preference_id"]
    rows: list[Dict[str, Any]] = []
    for identity, group in instance.groupby(keys, dropna=False, sort=True):
        row: Dict[str, Any] = dict(zip(keys, identity))
        row.update(
            {
                "cost_weight": float(group["cost_weight"].iloc[0]),
                "risk_weight": float(group["risk_weight"].iloc[0]),
                "n_instances_total": int(group["instance_index"].nunique()),
                "n_instances_quality": int(group["weighted_objective_normalized"].notna().sum()),
                "strict_feasible_rate": float(group["feasible_rate"].mean()),
                "all_restarts_feasible_rate": float(group["all_restarts_feasible"].mean()),
                "terminal_clear_rate": float(group["terminal_clear_rate"].mean()),
                "technical_failure_rate": float(group["technical_failure_rate"].mean()),
                "missing_rate": float(group["missing_rate"].mean()),
            }
        )
        for field in (
            *QUALITY_FIELDS,
            "runtime_seconds",
            "successful_runtime_seconds_conditional",
            "failed_runtime_seconds_conditional",
        ):
            values = pd.to_numeric(group[field], errors="coerce").dropna()
            complete_quality = len(values) == len(group)
            row[f"{field}_mean"] = (
                float(values.mean()) if complete_quality and len(values) else None
            )
            row[f"{field}_sd"] = (
                float(values.std(ddof=1))
                if complete_quality and len(values) >= 2
                else None
            )
            row[f"{field}_conditional_mean"] = float(values.mean()) if len(values) else None
            row[f"{field}_conditional_sd"] = (
                float(values.std(ddof=1)) if len(values) >= 2 else None
            )
            row[f"{field}_median"] = (
                float(values.median()) if complete_quality and len(values) else None
            )
            row[f"{field}_q1"] = (
                float(values.quantile(0.25)) if complete_quality and len(values) else None
            )
            row[f"{field}_q3"] = (
                float(values.quantile(0.75)) if complete_quality and len(values) else None
            )
            row[f"{field}_conditional_median"] = (
                float(values.median()) if len(values) else None
            )
            row[f"{field}_conditional_q1"] = (
                float(values.quantile(0.25)) if len(values) else None
            )
            row[f"{field}_conditional_q3"] = (
                float(values.quantile(0.75)) if len(values) else None
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_ppo_ga(instance: Any) -> Any:
    import pandas as pd

    ppo = instance[instance["method"] == "ppo"].copy()
    ga = instance[instance["method"] == "ga"][
        ["test_scale", "instance_index", "preference_id", "weighted_objective_normalized"]
    ].rename(columns={"weighted_objective_normalized": "ga_j"})
    paired = ppo.merge(
        ga,
        on=["test_scale", "instance_index", "preference_id"],
        how="left",
        validate="many_to_one",
    )
    paired = paired.rename(columns={"weighted_objective_normalized": "ppo_j"})
    valid = paired["ppo_j"].notna() & paired["ga_j"].notna() & (paired["ga_j"].abs() > 1e-12)
    paired["delta_j_percent"] = None
    paired.loc[valid, "delta_j_percent"] = (
        (paired.loc[valid, "ga_j"] - paired.loc[valid, "ppo_j"])
        / paired.loc[valid, "ga_j"]
        * 100.0
    )
    return paired[
        [
            "model_id",
            "test_scale",
            "instance_index",
            "preference_id",
            "ppo_j",
            "ga_j",
            "delta_j_percent",
        ]
    ]


def _milp_gap_table(instance: Any) -> Any:
    import pandas as pd

    milp = instance[(instance["method"] == "milp") & (instance["test_scale"] == "Test-1")][
        ["instance_index", "preference_id", "weighted_objective_normalized", "solver_status"]
    ].rename(
        columns={
            "weighted_objective_normalized": "milp_j",
            "solver_status": "milp_status",
        }
    )
    candidates = instance[
        (instance["test_scale"] == "Test-1")
        & (
            (instance["method"].isin(["ga", "heuristic"]))
            | ((instance["method"] == "ppo") & (instance["model_id"] == "Train-S"))
        )
    ].copy()
    paired = candidates.merge(
        milp,
        on=["instance_index", "preference_id"],
        how="left",
        validate="many_to_one",
    )
    valid = (
        paired["weighted_objective_normalized"].notna()
        & paired["milp_j"].notna()
        & (paired["milp_status"].str.casefold() == "optimal")
        & (paired["milp_j"].abs() > 1e-12)
    )
    paired["gap_percent"] = None
    paired.loc[valid, "gap_percent"] = (
        (paired.loc[valid, "weighted_objective_normalized"] - paired.loc[valid, "milp_j"])
        / paired.loc[valid, "milp_j"].abs()
        * 100.0
    )
    return paired[
        [
            "method",
            "model_id",
            "instance_index",
            "preference_id",
            "weighted_objective_normalized",
            "milp_j",
            "milp_status",
            "gap_percent",
        ]
    ]


def _generalization_tables(instance: Any, paired: Any) -> tuple[Any, Any]:
    import pandas as pd

    ppo = instance[instance["method"] == "ppo"].copy()
    rows: list[Dict[str, Any]] = []
    for (model, scale), group in ppo.groupby(["model_id", "test_scale"], sort=True):
        quality = pd.to_numeric(group["weighted_objective_normalized"], errors="coerce").dropna()
        paired_group = paired[
            (paired["model_id"] == model) & (paired["test_scale"] == scale)
        ]
        delta = pd.to_numeric(
            paired_group["delta_j_percent"],
            errors="coerce",
        ).dropna()
        runtime = pd.to_numeric(group["runtime_seconds"], errors="coerce").dropna()
        rows.append(
            {
                "model_id": model,
                "test_scale": scale,
                "independent_instances": int(group["instance_index"].nunique()),
                "instance_preference_cells": int(len(group)),
                "quality_cells": int(len(quality)),
                "macro_j": (
                    float(quality.mean()) if len(quality) == len(group) and len(group) else None
                ),
                "conditional_macro_j": float(quality.mean()) if len(quality) else None,
                "macro_j_sd_instance_preference": (
                    float(quality.std(ddof=1))
                    if len(quality) == len(group) and len(quality) >= 2
                    else None
                ),
                "conditional_macro_j_sd_instance_preference": (
                    float(quality.std(ddof=1)) if len(quality) >= 2 else None
                ),
                "strict_feasible_rate": float(group["feasible_rate"].mean()),
                "terminal_clear_rate": float(group["terminal_clear_rate"].mean()),
                "mean_inference_seconds": (
                    float(runtime.mean()) if len(runtime) == len(group) and len(group) else None
                ),
                "conditional_mean_inference_seconds": (
                    float(runtime.mean()) if len(runtime) else None
                ),
                "relative_ga_improvement_percent": (
                    float(delta.mean())
                    if len(delta) == len(paired_group) and len(paired_group)
                    else None
                ),
                "conditional_relative_ga_improvement_percent": (
                    float(delta.mean()) if len(delta) else None
                ),
                "expected_paired_ga_n": int(len(paired_group)),
                "paired_ga_n": int(len(delta)),
            }
        )
    operational = pd.DataFrame(rows)
    if not operational.empty:
        operational["two_model_relative_gap_percent"] = None
        for scale, indices in operational.groupby("test_scale").groups.items():
            values = pd.to_numeric(operational.loc[indices, "macro_j"], errors="coerce")
            if len(values) == 2 and values.notna().all():
                best = float(values.min())
                if abs(best) > 1e-12:
                    operational.loc[indices, "two_model_relative_gap_percent"] = (
                        (values - best) / abs(best) * 100.0
                    )
    matrix = operational.pivot(
        index="model_id",
        columns="test_scale",
        values=["macro_j", "two_model_relative_gap_percent"],
    ).reset_index()
    if hasattr(matrix.columns, "to_flat_index"):
        matrix.columns = [
            "_".join(str(part) for part in item if str(part))
            if isinstance(item, tuple)
            else str(item)
            for item in matrix.columns.to_flat_index()
        ]
    return matrix, operational


def summarize_run(run_dir: str | Path) -> Dict[str, str]:
    """Rebuild every result table solely from committed cell records."""

    import pandas as pd

    root, config, manifest = _load_run(run_dir)
    _assert_source_lock(manifest)
    schedule = prepare_schedule(root)
    records: list[Dict[str, Any]] = []
    input_cache: Dict[tuple[str, int], tuple[ModelParams, tuple[float, float]]] = {}
    for entry in schedule:
        path = root / "cells" / f"{entry['cell_id']}.json"
        if path.exists():
            record = _read_json(path)
            if (
                record.get("cell_id") != entry["cell_id"]
                or record.get("logical_cell_id") != entry["logical_cell_id"]
                or record.get("cell") != entry["cell"]
                or record.get("input_hashes") != entry["input_hashes"]
            ):
                raise ArtifactIntegrityError(
                    f"cell identity changed before summary: {entry['cell_id']}"
                )
            if record.get("execution_source_bundle_sha256") != entry["input_hashes"].get(
                "source_bundle_sha256"
            ):
                raise ArtifactIntegrityError(
                    f"execution source identity changed before summary: {entry['cell_id']}"
                )
            runtime = record.get("runtime_seconds")
            if runtime is None or not math.isfinite(float(runtime)) or float(runtime) < 0.0:
                raise ArtifactIntegrityError(
                    f"invalid recorded runtime before summary: {entry['cell_id']}"
                )
            status = record.get("status")
            if status == "complete":
                checked = load_valid_completed_cell(root, path, entry["input_hashes"])
                assert checked is not None
                cache_key = (
                    str(entry["cell"]["test_scale"]),
                    int(entry["cell"]["instance_index"]),
                )
                if cache_key not in input_cache:
                    params, _, _, refs = _load_instance_and_initial(root, entry)
                    input_cache[cache_key] = (params, refs)
                params, refs = input_cache[cache_key]
                solution = deserialize_solution(_read_json(root / record["solution_file"]))
                preference = (
                    float(entry["cell"]["cost_weight"]),
                    float(entry["cell"]["risk_weight"]),
                )
                recalculated = _quality_metrics(params, solution, preference, refs)
                for field in (*QUALITY_FIELDS, "feasible", "terminal_inventory_clear"):
                    if field in {"feasible", "terminal_inventory_clear"}:
                        same = bool(record["metrics"].get(field)) == bool(
                            recalculated.get(field)
                        )
                    else:
                        same = _same_number(
                            record["metrics"].get(field), recalculated.get(field)
                        )
                    if not same:
                        raise ArtifactIntegrityError(
                            f"recorded metric changed before summary: "
                            f"{entry['cell_id']}:{field}"
                        )
                recalculated_plan_hash = (
                    plan_sha256(solution["plan"])
                    if solution.get("plan") is not None
                    else None
                )
                if recalculated_plan_hash != record.get("result_plan_sha256"):
                    raise ArtifactIntegrityError(
                        f"recorded plan hash changed before summary: {entry['cell_id']}"
                    )
            elif status != "failed":
                raise ArtifactIntegrityError(
                    f"unsupported cell status before summary: {entry['cell_id']}"
                )
            records.append(record)
        else:
            records.append(
                {
                    "cell_id": entry["cell_id"],
                    "logical_cell_id": entry["logical_cell_id"],
                    "cell": entry["cell"],
                    "status": "missing",
                    "instance_file": entry["instance_file"],
                    "initial_solution_file": entry["initial_solution_file"],
                }
            )
    raw = pd.DataFrame([_cell_record_to_row(record) for record in records])
    if raw.empty:
        raise RuntimeError("no committed cells are available to summarize")
    instance = _instance_level_frame(raw, config)
    summary = _summary_by_preference(instance)
    paired = _paired_ppo_ga(instance)
    gaps = _milp_gap_table(instance)
    gap_summary_rows = []
    for identity, group in gaps.groupby(
        ["method", "model_id", "preference_id"], dropna=False, sort=True
    ):
        values = pd.to_numeric(group["gap_percent"], errors="coerce").dropna()
        optimal_n = int((group["milp_status"].str.casefold() == "optimal").sum())
        fully_observed = len(values) == optimal_n and optimal_n > 0
        gap_summary_rows.append(
            {
                "method": identity[0],
                "model_id": identity[1],
                "preference_id": identity[2],
                "optimal_reference_n": optimal_n,
                "valid_n": int(len(values)),
                "mean_gap_percent": float(values.mean()) if fully_observed else None,
                "sample_sd_gap_percent": (
                    float(values.std(ddof=1))
                    if fully_observed and len(values) >= 2
                    else None
                ),
                "conditional_mean_gap_percent": float(values.mean()) if len(values) else None,
            }
        )
    gap_summary = pd.DataFrame(gap_summary_rows)
    delta_summary_rows = []
    for identity, group in paired.groupby(
        ["model_id", "test_scale", "preference_id"], dropna=False, sort=True
    ):
        values = pd.to_numeric(group["delta_j_percent"], errors="coerce").dropna()
        fully_observed = len(values) == len(group) and len(group) > 0
        delta_summary_rows.append(
            {
                "model_id": identity[0],
                "test_scale": identity[1],
                "preference_id": identity[2],
                "expected_paired_n": int(len(group)),
                "paired_n": int(len(values)),
                "mean_delta_j_percent": float(values.mean()) if fully_observed else None,
                "sample_sd_delta_j_percent": (
                    float(values.std(ddof=1))
                    if fully_observed and len(values) >= 2
                    else None
                ),
                "conditional_mean_delta_j_percent": float(values.mean()) if len(values) else None,
            }
        )
    delta_summary = pd.DataFrame(delta_summary_rows)
    gen_matrix, gen_operational = _generalization_tables(instance, paired)
    cell_ledger_rows, cell_ledger_sha256 = _cell_file_ledger(root, schedule)

    tables = {
        "raw_results.csv": raw,
        "instance_level_results.csv": instance,
        "summary_by_preference.csv": summary,
        "paired_ppo_vs_ga.csv": paired,
        "paired_ppo_vs_ga_summary.csv": delta_summary,
        "milp_gap_by_instance.csv": gaps,
        "milp_gap_summary.csv": gap_summary,
        "generalization_matrix.csv": gen_matrix,
        "generalization_operational.csv": gen_operational,
        "cell_file_ledger.csv": pd.DataFrame(cell_ledger_rows),
    }
    output: Dict[str, str] = {}
    for filename, frame in tables.items():
        path = root / "tables" / filename
        _write_dataframe(path, frame)
        output[filename] = file_sha256(path)
    manifest["tables"] = {
        name: {"file": f"tables/{name}", "file_sha256": digest}
        for name, digest in output.items()
    }
    manifest["summary_input_cell_ledger_sha256"] = cell_ledger_sha256
    _invalidate_verification(manifest)
    manifest["summarized_at"] = _utc_now()
    manifest["stage"] = "summarized"
    atomic_write_json(root / "manifest.json", manifest)
    return output


def _same_number(expected: Any, observed: Any, tolerance: float = 1e-8) -> bool:
    if expected is None:
        return observed is None
    return math.isclose(float(expected), float(observed), rel_tol=tolerance, abs_tol=tolerance)


def _rerun_cell_solution(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
) -> tuple[Dict[str, Any], str]:
    entry = {
        "cell_id": record["cell_id"],
        "instance_file": record["instance_file"],
        "initial_solution_file": record["initial_solution_file"],
        "input_hashes": record["input_hashes"],
    }
    params, _, initial_plan, refs = _load_instance_and_initial(root, entry)
    cell = record["cell"]
    preference = (float(cell["cost_weight"]), float(cell["risk_weight"]))
    seed = int(cell["seed"])
    method = str(cell["method"])
    if method == "heuristic":
        return route_plan_to_solution(params, initial_plan), "completed"
    if method == "ga":
        ga_dir = ROOT / "遗传算法"
        if str(ga_dir) not in sys.path:
            sys.path.insert(0, str(ga_dir))
        from genetic_algorithm import ClassicGeneticAlgorithm, GAConfig

        optimizer_params = _with_normalized_objective(params, preference, refs)
        ga_config = GAConfig(
            random_seed=seed,
            **config["algorithm"]["ga_by_scale"][cell["test_scale"]],
        )
        initial_solution = route_plan_to_solution(params, initial_plan)
        result = ClassicGeneticAlgorithm(
            optimizer_params,
            ga_config,
            initial_chromosome=_initial_chromosome(params, initial_plan),
            initial_solution=initial_solution,
        ).run()
        return result.best_solution, "completed" if result.feasible else "infeasible"
    if method == "ppo":
        from src.ppo_improver import PPOImprover

        model = manifest["models"][cell["model_id"]]
        checkpoint = root / model["checkpoint_file"]
        improver = PPOImprover.from_frozen_checkpoint(
            params,
            checkpoint,
            algorithm_config={"ppo": config["algorithm"]["ppo"]},
            network_config=config["network"],
            seed=seed,
        )
        result = improver.improve(preference, seed=seed, initial_plan=initial_plan)
        return result.solution, "completed"
    if method == "milp":
        from hazardous_waste_model import solve_with_milp

        optimizer_params = _with_normalized_objective(params, preference, refs)
        result = solve_with_milp(
            optimizer_params,
            time_limit=float(config["algorithm"]["milp_time_limit_seconds"]),
        )
        return result.solution, result.status
    raise ValueError(f"unsupported replay method: {method}")


def verify_run(run_dir: str | Path, *, replay_per_stratum: int = 1) -> Dict[str, Any]:
    """Audit counts and hashes, then replay a deterministic stratified sample."""

    from collections import Counter

    from src.supplementary_protocol import protocol_cell_counts

    if int(replay_per_stratum) <= 0:
        raise ValueError("replay_per_stratum must be positive")
    root, config, manifest = _load_run(run_dir)
    schedule = prepare_schedule(root)
    cell_ledger_rows, cell_ledger_sha256 = _cell_file_ledger(root, schedule)
    summary_ledger_match = (
        manifest.get("summary_input_cell_ledger_sha256") == cell_ledger_sha256
    )
    observed_table_hashes, table_integrity_errors = _table_integrity(root, manifest)
    expected_counts = protocol_cell_counts(config)
    observed_counts: Counter[str] = Counter()
    missing: list[str] = []
    failed: list[str] = []
    integrity_errors: list[str] = []
    completed_records: list[Dict[str, Any]] = []
    for entry in schedule:
        path = root / "cells" / f"{entry['cell_id']}.json"
        if not path.exists():
            missing.append(entry["cell_id"])
            continue
        record = _read_json(path)
        if (
            record.get("cell_id") != entry["cell_id"]
            or record.get("logical_cell_id") != entry["logical_cell_id"]
            or record.get("cell") != entry["cell"]
        ):
            integrity_errors.append(f"cell identity changed: {entry['cell_id']}")
            continue
        if record.get("status") == "failed":
            failed.append(entry["cell_id"])
            continue
        try:
            checked = load_valid_completed_cell(root, path, entry["input_hashes"])
            assert checked is not None
            trace_path = root / checked["trace_file"]
            if file_sha256(trace_path) != checked["trace_file_sha256"]:
                raise ArtifactIntegrityError(f"trace hash mismatch: {entry['cell_id']}")
            if checked["cell"]["method"] == "ppo":
                before = checked.get("policy_state_sha256_before")
                after = checked.get("policy_state_sha256_after")
                expected_state = entry["input_hashes"]["policy_state_sha256"]
                if before != after or before != expected_state:
                    raise ArtifactIntegrityError(
                        f"frozen policy identity mismatch: {entry['cell_id']}"
                    )
            if checked.get("execution_source_bundle_sha256") != entry["input_hashes"].get(
                "source_bundle_sha256"
            ):
                raise ArtifactIntegrityError(
                    f"execution source identity mismatch: {entry['cell_id']}"
                )
            observed_counts[checked["cell"]["method"]] += 1
            completed_records.append(checked)
        except (ArtifactIntegrityError, KeyError, OSError, ValueError) as exc:
            integrity_errors.append(str(exc))

    strata: Dict[tuple[str, str, str], list[Dict[str, Any]]] = {}
    for record in completed_records:
        cell = record["cell"]
        key = (str(cell["model_id"]), str(cell["test_scale"]), str(cell["preference_id"]))
        strata.setdefault(key, []).append(record)
    artifact_rechecked = 0
    artifact_errors: list[str] = []
    for key in sorted(strata):
        chosen = sorted(strata[key], key=lambda item: item["cell_id"])[:replay_per_stratum]
        for record in chosen:
            try:
                entry = {
                    "cell_id": record["cell_id"],
                    "instance_file": record["instance_file"],
                    "initial_solution_file": record["initial_solution_file"],
                    "input_hashes": record["input_hashes"],
                }
                params, _, _, refs = _load_instance_and_initial(root, entry)
                solution = deserialize_solution(_read_json(root / record["solution_file"]))
                pref = (
                    float(record["cell"]["cost_weight"]),
                    float(record["cell"]["risk_weight"]),
                )
                metrics = _quality_metrics(params, solution, pref, refs)
                if bool(metrics["feasible"]) != bool(record["metrics"]["feasible"]):
                    raise AssertionError("feasibility changed on replay")
                if solution.get("plan") is not None and plan_sha256(solution["plan"]) != record.get(
                    "result_plan_sha256"
                ):
                    raise AssertionError("plan hash changed on replay")
                for field in QUALITY_FIELDS:
                    if not _same_number(record["metrics"].get(field), metrics.get(field)):
                        raise AssertionError(f"metric changed on replay: {field}")
                artifact_rechecked += 1
            except Exception as exc:
                artifact_errors.append(f"{record['cell_id']}: {type(exc).__name__}: {exc}")

    computational_strata: Dict[tuple[str, str, str], list[Dict[str, Any]]] = {}
    for record in completed_records:
        cell = record["cell"]
        key = (str(cell["method"]), str(cell["model_id"]), str(cell["test_scale"]))
        computational_strata.setdefault(key, []).append(record)
    solver_replayed = 0
    solver_replay_errors: list[str] = []
    milp_nonoptimal_replay_differences: list[Dict[str, Any]] = []
    replay_method_order = {"ppo": 0, "ga": 1, "heuristic": 2, "milp": 3}
    for key in sorted(
        computational_strata,
        key=lambda item: (replay_method_order.get(item[0], 99), item[1], item[2]),
    ):
        ranked = sorted(
            computational_strata[key],
            key=lambda item: json_sha256(
                {
                    "purpose": "computational-replay",
                    "protocol": config["protocol_id"],
                    "cell_id": item["cell_id"],
                }
            ),
        )[:replay_per_stratum]
        for record in ranked:
            try:
                rerun_solution, rerun_status = _rerun_cell_solution(
                    root, config, manifest, record
                )
                entry = {
                    "cell_id": record["cell_id"],
                    "instance_file": record["instance_file"],
                    "initial_solution_file": record["initial_solution_file"],
                    "input_hashes": record["input_hashes"],
                }
                params, _, _, refs = _load_instance_and_initial(root, entry)
                preference = (
                    float(record["cell"]["cost_weight"]),
                    float(record["cell"]["risk_weight"]),
                )
                rerun_metrics = _quality_metrics(params, rerun_solution, preference, refs)
                method = str(record["cell"]["method"])
                if bool(rerun_metrics["feasible"]) != bool(record["metrics"]["feasible"]):
                    raise AssertionError("solver replay changed feasibility")
                if method != "milp":
                    if not _same_number(
                        record["metrics"].get("weighted_objective_normalized"),
                        rerun_metrics.get("weighted_objective_normalized"),
                        tolerance=1e-10,
                    ):
                        raise AssertionError("solver replay changed normalized objective")
                    rerun_plan_hash = (
                        plan_sha256(rerun_solution["plan"])
                        if rerun_solution.get("plan") is not None
                        else None
                    )
                    if rerun_plan_hash != record.get("result_plan_sha256"):
                        raise AssertionError("solver replay changed final plan hash")
                elif str(record.get("solver_status", "")).casefold() == "optimal":
                    if str(rerun_status).casefold() != "optimal":
                        raise AssertionError("optimal MILP did not replay as optimal")
                    if not _same_number(
                        record["metrics"].get("weighted_objective_normalized"),
                        rerun_metrics.get("weighted_objective_normalized"),
                        tolerance=1e-6,
                    ):
                        raise AssertionError("optimal MILP replay changed normalized objective")
                elif (
                    str(record.get("solver_status")) != str(rerun_status)
                    or not _same_number(
                        record["metrics"].get("weighted_objective_normalized"),
                        rerun_metrics.get("weighted_objective_normalized"),
                        tolerance=1e-6,
                    )
                ):
                    milp_nonoptimal_replay_differences.append(
                        {
                            "cell_id": record["cell_id"],
                            "original_status": record.get("solver_status"),
                            "rerun_status": rerun_status,
                            "original_objective": record["metrics"].get(
                                "weighted_objective_normalized"
                            ),
                            "rerun_objective": rerun_metrics.get(
                                "weighted_objective_normalized"
                            ),
                        }
                    )
                solver_replayed += 1
            except Exception as exc:
                solver_replay_errors.append(
                    f"{record['cell_id']}: {type(exc).__name__}: {exc}"
                )

    current_sources = _source_hashes()
    source_changes = {
        name: {"at_init": digest, "now": current_sources.get(name)}
        for name, digest in manifest["source_files_sha256_at_init"].items()
        if digest != current_sources.get(name)
    }
    expected_method_counts = {
        key: value for key, value in expected_counts.items() if key != "total"
    }
    count_match = dict(observed_counts) == expected_method_counts
    report = {
        "schema_version": "supplementary-verification-v1",
        "verified_at": _utc_now(),
        "protocol_id": config["protocol_id"],
        "schedule_file_sha256": manifest["schedule"]["file_sha256"],
        "summary_input_cell_ledger_sha256": cell_ledger_sha256,
        "summary_ledger_match": summary_ledger_match,
        "table_files_sha256": observed_table_hashes,
        "table_integrity_errors": table_integrity_errors,
        "expected_cell_counts": expected_counts,
        "observed_complete_counts": dict(observed_counts),
        "count_match": count_match,
        "model_count": len(manifest["models"]),
        "checkpoint_file_count": len(list((root / "models").glob("*.pt"))),
        "test_instance_count": sum(len(items) for items in manifest["test_sets"].values()),
        "missing_count": len(missing),
        "missing_cell_ids": missing,
        "failed_count": len(failed),
        "failed_cell_ids": failed,
        "integrity_error_count": len(integrity_errors),
        "integrity_errors": integrity_errors,
        "artifact_rechecked_count": artifact_rechecked,
        "artifact_error_count": len(artifact_errors),
        "artifact_errors": artifact_errors,
        "solver_replayed_count": solver_replayed,
        "solver_replay_error_count": len(solver_replay_errors),
        "solver_replay_errors": solver_replay_errors,
        "milp_nonoptimal_replay_differences": milp_nonoptimal_replay_differences,
        "source_changes_since_init": source_changes,
    }
    report["passed"] = bool(
        count_match
        and report["model_count"] == 2
        and report["checkpoint_file_count"] == 2
        and report["test_instance_count"]
        == sum(int(item["instances"]) for item in config["test_scales"].values())
        and not missing
        and not failed
        and not integrity_errors
        and not artifact_errors
        and not solver_replay_errors
        and not source_changes
        and summary_ledger_match
        and not table_integrity_errors
    )
    report_path = root / "verification" / "verification_report.json"
    atomic_write_json(report_path, report)
    manifest["verification"] = {
        "file": report_path.relative_to(root).as_posix(),
        "file_sha256": file_sha256(report_path),
        "passed": report["passed"],
    }
    manifest["stage"] = "verified" if report["passed"] else "verification_failed"
    atomic_write_json(root / "manifest.json", manifest)
    return report


def export_replication_summary(
    run_dir: str | Path,
    destination: str | Path,
) -> Path:
    """Copy compact auditable outputs (not thousands of raw cells) into Git."""

    root, _, manifest = _load_run(run_dir)
    _assert_source_lock(manifest)
    verification = manifest.get("verification")
    if (
        manifest.get("stage") != "verified"
        or not isinstance(verification, Mapping)
        or verification.get("passed") is not True
    ):
        raise RuntimeError("a passing verification report is required before export")
    report_path = root / str(verification["file"])
    if (
        not report_path.is_file()
        or file_sha256(report_path) != verification.get("file_sha256")
    ):
        raise ArtifactIntegrityError("verification report is missing, changed, or failed")
    report = _read_json(report_path)
    if report.get("passed") is not True:
        raise ArtifactIntegrityError("verification report is missing, changed, or failed")
    schedule = prepare_schedule(root)
    _, current_cell_ledger_sha256 = _cell_file_ledger(root, schedule)
    current_table_hashes, table_errors = _table_integrity(root, manifest)
    if (
        report.get("schedule_file_sha256") != manifest["schedule"]["file_sha256"]
        or report.get("summary_input_cell_ledger_sha256")
        != current_cell_ledger_sha256
        or manifest.get("summary_input_cell_ledger_sha256")
        != current_cell_ledger_sha256
        or report.get("table_files_sha256") != current_table_hashes
        or table_errors
    ):
        raise ArtifactIntegrityError(
            "verification proof is stale relative to cells, schedule, or tables"
        )
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to export over an existing destination: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for relative in ("manifest.json", "protocol_config.json", "schedule.json"):
            source = root / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, staging / relative)
        for folder in ("tables", "verification", "training_history"):
            source = root / folder
            if not source.is_dir():
                raise FileNotFoundError(source)
            destination_folder = staging / folder
            destination_folder.mkdir(parents=True, exist_ok=True)
            for item in source.iterdir():
                if item.is_file():
                    shutil.copy2(item, destination_folder / item.name)
        models_target = staging / "models"
        models_target.mkdir(parents=True, exist_ok=True)
        for model in manifest["models"].values():
            source = root / model["checkpoint_file"]
            if file_sha256(source) != model["checkpoint_sha256"]:
                raise ArtifactIntegrityError(f"checkpoint changed before export: {source}")
            shutil.copy2(source, models_target / source.name)
        inventory = [
            {
                "file": path.relative_to(staging).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file() and path.name != "inventory.json"
        ]
        atomic_write_json(staging / "inventory.json", {"files": inventory})
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target
