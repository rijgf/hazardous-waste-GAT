#!/usr/bin/env python3
"""Build publication tables from a verified supplementary-experiment run.

The script is deliberately downstream of ``summarize`` and ``verify``.  It never
reads individual solution files and never recomputes conditional-on-success
statistics.  Missing primary statistics remain missing in every paper table.

Usage
-----
python output/supplementary-experiments/scripts/build_paper_tables.py \
    outputs/supplementary_experiment_v1
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


SCRIPT_ID = "output/supplementary-experiments/scripts/build_paper_tables.py"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tables"
PAPER_TABLE_FILES = {
    "table3": "table3_small_scale",
    "table4": "table4_large_scale",
    "table5": "table5_scale_design",
    "table6": "table6_generalization_matrix",
    "table7": "table7_generalization_operational",
    "risk": "appendix_risk_components",
    "all_methods": "appendix_all_methods",
    "milp_status": "appendix_milp_status",
}
SUMMARY_TABLE_FILES = (
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
FORMAL_COUNTS = {"ppo": 6000, "ga": 3000, "heuristic": 500, "milp": 250, "total": 9750}
FORMAL_PROTOCOL_SHA256 = "24ff3429c5aac99bf01607d682cc54539dbd96a044e891697547b01b291959ba"
FORMAL_PREFERENCES = (
    ("C100-R000", 1.0, 0.0),
    ("C075-R025", 0.75, 0.25),
    ("C050-R050", 0.5, 0.5),
    ("C025-R075", 0.25, 0.75),
    ("C000-R100", 0.0, 1.0),
)
FORMAL_SCALE_SHAPES = {
    "Test-1": (3, 2, 2, 3, 2),
    "Test-2": (6, 2, 2, 4, 3),
    "Test-3": (10, 3, 3, 5, 3),
    "Test-4": (20, 4, 3, 8, 4),
}
SCALE_FIELDS = (
    "producers_count",
    "waste_types_count",
    "facilities_count",
    "vehicles_count",
    "periods_count",
)
ALLOWED_HYPOTHESIS_IDS = {"E1", "E2", "G-Train-S", "G-Train-L"}
RISK_COMPONENTS = (
    ("transport", "transport_risk", "运输风险"),
    ("coload", "coload_risk", "共载风险"),
    ("producer_inventory", "producer_inventory_risk", "产废端库存风险"),
    ("facility_inventory", "facility_inventory_risk", "处理处置端库存风险"),
)
PAPER_QUALITY_FIELDS = (
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


class PaperTableError(RuntimeError):
    """Raised when verified inputs do not support an auditable paper table."""


@dataclass(frozen=True)
class PrimaryStats:
    """A primary mean/SD whose sampling unit is the independent instance."""

    mean: float | None
    sd: float | None
    valid_n: int
    expected_n: int

    @property
    def se(self) -> float | None:
        if self.sd is None or self.valid_n < 2:
            return None
        return self.sd / math.sqrt(self.valid_n)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise PaperTableError(f"required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PaperTableError(f"invalid JSON file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PaperTableError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _recorded_path(root: Path, value: Any, label: str, exact: str | None = None) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PaperTableError(f"{label} has no valid relative path")
    normalized = value.replace("\\", "/")
    if exact is not None and normalized != exact:
        raise PaperTableError(f"{label} path must be {exact}, received {value!r}")
    relative = Path(value)
    if relative.is_absolute():
        raise PaperTableError(f"{label} path must be relative: {value!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PaperTableError(f"{label} path escapes the run directory: {value!r}") from exc
    return resolved


def _finite(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rate_percent(value: Any) -> float | None:
    number = _finite(value)
    return None if number is None else 100.0 * number


def _se(sd: Any, n: Any) -> float | None:
    spread = _finite(sd)
    count = int(n) if _finite(n) is not None else 0
    if spread is None or count < 2:
        return None
    return spread / math.sqrt(count)


def _assert_close(label: str, left: Any, right: Any, tolerance: float = 1e-9) -> None:
    a = _finite(left)
    b = _finite(right)
    if a is None and b is None:
        return
    if a is None or b is None or not math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance):
        raise PaperTableError(f"derived {label} disagrees with verified summary: {a!r} vs {b!r}")


def _require_columns(frame: pd.DataFrame, filename: str, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise PaperTableError(f"{filename} is missing columns: {', '.join(missing)}")


def _only_row(frame: pd.DataFrame, label: str, **filters: Any) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        if column not in frame.columns:
            raise PaperTableError(f"{label}: filter column is missing: {column}")
        mask &= frame[column].astype(str) == str(value)
    selected = frame.loc[mask]
    if len(selected) != 1:
        detail = ", ".join(f"{key}={value}" for key, value in filters.items())
        raise PaperTableError(f"{label}: expected one row for {detail}, found {len(selected)}")
    return selected.iloc[0]


def _protocol_counts(protocol: Mapping[str, Any]) -> dict[str, int]:
    preferences = _preferences(protocol)
    scales = protocol.get("test_scales")
    methods = protocol.get("methods")
    if not isinstance(scales, Mapping) or not isinstance(methods, Mapping):
        raise PaperTableError("protocol has invalid test_scales or methods")
    counts: dict[str, int] = {}
    for method in ("ppo", "ga", "heuristic", "milp"):
        scope = methods.get(method)
        if not isinstance(scope, Mapping):
            raise PaperTableError(f"protocol has no valid {method} scope")
        models = scope.get("models")
        method_scales = scope.get("scales")
        if not isinstance(models, list) or not isinstance(method_scales, list):
            raise PaperTableError(f"protocol has invalid {method} models/scales")
        try:
            restarts = int(scope["restarts"])
            instances = sum(int(scales[scale]["instances"]) for scale in method_scales)
        except (KeyError, TypeError, ValueError) as exc:
            raise PaperTableError(f"protocol has invalid {method} count inputs") from exc
        if restarts <= 0 or not models:
            raise PaperTableError(f"protocol has invalid {method} restart/model count")
        counts[method] = len(models) * instances * len(preferences) * restarts
    counts["total"] = sum(counts.values())
    return counts


def _verification_stratum_minima(protocol: Mapping[str, Any]) -> tuple[int, int]:
    """Return the minimum successful artifact and solver replays for the protocol.

    ``verify_run`` first validates every scheduled cell and its recorded file hashes.
    It then recomputes metrics for at least one artifact in each
    (model, test-scale, preference) stratum and reruns at least one solver cell in
    each (method, model, test-scale) stratum.  These sampled replay counts are not
    the total number of scheduled cells.
    """

    methods = protocol.get("methods")
    if not isinstance(methods, Mapping):
        raise PaperTableError("protocol has invalid methods")
    preference_ids = [str(item["id"]) for item in _preferences(protocol)]
    artifact_strata: set[tuple[str, str, str]] = set()
    solver_strata: set[tuple[str, str, str]] = set()
    for method, scope in methods.items():
        if not isinstance(scope, Mapping):
            raise PaperTableError(f"protocol has invalid {method} scope")
        models = scope.get("models")
        scales = scope.get("scales")
        if not isinstance(models, list) or not isinstance(scales, list):
            raise PaperTableError(f"protocol has invalid {method} models/scales")
        for model in models:
            for scale in scales:
                solver_strata.add((str(method), str(model), str(scale)))
                for preference_id in preference_ids:
                    artifact_strata.add((str(model), str(scale), preference_id))
    return len(artifact_strata), len(solver_strata)


def _validate_protocol_shape(protocol: Mapping[str, Any], *, smoke: bool) -> dict[str, int]:
    models = protocol.get("models")
    expected_models = [
        {"id": "Train-S", "train_scale": "Test-1", "checkpoint": "small_model.pt"},
        {"id": "Train-L", "train_scale": "Test-4", "checkpoint": "large_model.pt"},
    ]
    if models != expected_models:
        raise PaperTableError("protocol must contain exactly the fixed Train-S and Train-L models")
    scales = protocol.get("test_scales")
    if not isinstance(scales, Mapping) or set(scales) != set(FORMAL_SCALE_SHAPES):
        raise PaperTableError("protocol must contain exactly Test-1 through Test-4")
    for scale, expected_shape in FORMAL_SCALE_SHAPES.items():
        item = scales[scale]
        if not isinstance(item, Mapping) or not isinstance(item.get("scale"), Mapping):
            raise PaperTableError(f"invalid scale definition: {scale}")
        observed_shape = tuple(int(item["scale"].get(field, -1)) for field in SCALE_FIELDS)
        if observed_shape != expected_shape:
            raise PaperTableError(
                f"{scale} shape changed: expected {expected_shape}, observed {observed_shape}"
            )
        expected_instances = 1 if smoke else 50
        if int(item.get("instances", -1)) != expected_instances:
            raise PaperTableError(
                f"{scale} must contain {expected_instances} instances in "
                f"{'smoke' if smoke else 'formal'} mode"
            )
    observed_preferences = tuple(
        (
            str(item.get("id")),
            float(item.get("cost_weight", math.nan)),
            float(item.get("risk_weight", math.nan)),
        )
        for item in _preferences(protocol)
    )
    expected_preferences = (("C050-R050", 0.5, 0.5),) if smoke else FORMAL_PREFERENCES
    if observed_preferences != expected_preferences:
        raise PaperTableError(
            "protocol preferences changed; formal mode requires the five fixed preferences"
        )
    methods = protocol.get("methods")
    expected_scopes = {
        "ppo": ({"Train-S", "Train-L"}, set(FORMAL_SCALE_SHAPES), 1 if smoke else 3),
        "ga": ({"GA"}, set(FORMAL_SCALE_SHAPES), 1 if smoke else 3),
        "heuristic": ({"Heuristic"}, {"Test-1", "Test-4"}, 1),
        "milp": ({"MILP"}, {"Test-1"}, 1),
    }
    if not isinstance(methods, Mapping) or set(methods) != set(expected_scopes):
        raise PaperTableError("protocol must contain exactly PPO, GA, heuristic, and MILP")
    for method, (expected_model_ids, expected_scale_ids, expected_restarts) in expected_scopes.items():
        scope = methods[method]
        if not isinstance(scope, Mapping):
            raise PaperTableError(f"protocol method scope is invalid for {method}")
        if (
            set(scope.get("models", [])) != expected_model_ids
            or set(scope.get("scales", [])) != expected_scale_ids
            or int(scope.get("restarts", -1)) != expected_restarts
        ):
            raise PaperTableError(f"formal experiment scope changed for {method}")
    if not smoke and int(protocol.get("restarts", -1)) != 3:
        raise PaperTableError("formal protocol top-level restarts must equal 3")
    counts = _protocol_counts(protocol)
    if not smoke and counts != FORMAL_COUNTS:
        raise PaperTableError(f"formal protocol counts changed: {counts} != {FORMAL_COUNTS}")
    if not smoke and _canonical_json_sha256(protocol) != FORMAL_PROTOCOL_SHA256:
        raise PaperTableError("formal protocol canonical hash changed from the frozen design")
    return counts


def _validate_schedule(
    schedule: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    protocol_hash: str,
    protocol_id: str,
    expected_counts: Mapping[str, int],
) -> list[str]:
    if schedule.get("protocol_sha256") != protocol_hash:
        raise PaperTableError("schedule protocol hash does not match canonical protocol hash")
    cells = schedule.get("cells")
    if not isinstance(cells, list):
        raise PaperTableError("schedule has no cells list")
    preference_map = {
        str(item["id"]): (float(item["cost_weight"]), float(item["risk_weight"]))
        for item in _preferences(protocol)
    }
    method_scopes = protocol["methods"]
    counts: Counter[str] = Counter()
    cell_ids: list[str] = []
    seen_cell_ids: set[str] = set()
    logical_ids: set[str] = set()
    semantic_cells: set[tuple[str, str, str, int, str, int]] = set()
    for entry in cells:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("cell"), Mapping):
            raise PaperTableError("schedule contains an invalid cell entry")
        cell_id = str(entry.get("cell_id", ""))
        logical_id = str(entry.get("logical_cell_id", ""))
        if not cell_id or not logical_id or cell_id in seen_cell_ids or logical_id in logical_ids:
            raise PaperTableError("schedule cell identities are missing or duplicated")
        cell_ids.append(cell_id)
        seen_cell_ids.add(cell_id)
        logical_ids.add(logical_id)
        cell = entry["cell"]
        method = str(cell.get("method"))
        if method not in method_scopes or cell.get("protocol_id") != protocol_id:
            raise PaperTableError(f"schedule cell has invalid protocol/method: {cell_id}")
        scope = method_scopes[method]
        model_id = str(cell.get("model_id"))
        scale = str(cell.get("test_scale"))
        preference_id = str(cell.get("preference_id"))
        try:
            instance_index = int(cell.get("instance_index"))
            restart_index = int(cell.get("restart_index"))
        except (TypeError, ValueError) as exc:
            raise PaperTableError(f"schedule cell has invalid indices: {cell_id}") from exc
        expected_weights = preference_map.get(preference_id)
        if (
            model_id not in scope["models"]
            or scale not in scope["scales"]
            or expected_weights is None
            or not 0 <= instance_index < _expected_instances(protocol, scale)
            or not 0 <= restart_index < int(scope["restarts"])
            or _finite(cell.get("cost_weight")) != expected_weights[0]
            or _finite(cell.get("risk_weight")) != expected_weights[1]
        ):
            raise PaperTableError(f"schedule cell falls outside the locked protocol: {cell_id}")
        semantic_key = (
            method,
            model_id,
            scale,
            instance_index,
            preference_id,
            restart_index,
        )
        if semantic_key in semantic_cells:
            raise PaperTableError(f"schedule duplicates a semantic experiment cell: {semantic_key}")
        semantic_cells.add(semantic_key)
        counts[method] += 1
    observed = {method: counts[method] for method in ("ppo", "ga", "heuristic", "milp")}
    observed["total"] = len(cells)
    if observed != dict(expected_counts):
        raise PaperTableError(f"schedule counts disagree with protocol: {observed} != {expected_counts}")
    return cell_ids


def _validate_report_counts(
    verification: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    expected_counts: Mapping[str, int],
    smoke: bool,
) -> None:
    expected = verification.get("expected_cell_counts")
    observed = verification.get("observed_complete_counts")
    if not isinstance(expected, Mapping) or not isinstance(observed, Mapping):
        raise PaperTableError("verification report has no valid expected/observed counts")
    normalized_expected = {
        key: int(expected.get(key, -1)) for key in ("ppo", "ga", "heuristic", "milp", "total")
    }
    normalized_observed = {
        key: int(observed.get(key, -1)) for key in ("ppo", "ga", "heuristic", "milp")
    }
    if normalized_expected != dict(expected_counts):
        raise PaperTableError(
            f"verification expected counts changed: {normalized_expected} != {expected_counts}"
        )
    if normalized_observed != {key: expected_counts[key] for key in normalized_observed}:
        raise PaperTableError("verification observed complete counts do not match the protocol")
    required_true = ("count_match", "summary_ledger_match")
    if any(verification.get(field) is not True for field in required_true):
        raise PaperTableError("verification count/ledger checks did not pass")
    required_zero = (
        "missing_count",
        "failed_count",
        "integrity_error_count",
        "artifact_error_count",
        "solver_replay_error_count",
    )
    if any(int(verification.get(field, -1)) != 0 for field in required_zero):
        raise PaperTableError("verification report contains missing, failed, or integrity-error cells")
    minimum_artifact_replays, minimum_solver_replays = _verification_stratum_minima(protocol)
    artifact_replays = int(verification.get("artifact_rechecked_count", -1))
    solver_replays = int(verification.get("solver_replayed_count", -1))
    if not minimum_artifact_replays <= artifact_replays <= expected_counts["total"]:
        raise PaperTableError(
            "verification did not successfully recompute at least one artifact "
            "per model/test-scale/preference stratum"
        )
    if not minimum_solver_replays <= solver_replays <= expected_counts["total"]:
        raise PaperTableError(
            "verification did not successfully rerun at least one solver cell "
            "per method/model/test-scale stratum"
        )
    if verification.get("source_changes_since_init") != {}:
        raise PaperTableError("verification report records source changes since initialization")
    if int(verification.get("model_count", -1)) != 2 or int(
        verification.get("checkpoint_file_count", -1)
    ) != 2:
        raise PaperTableError("verification did not find exactly two frozen model files")
    expected_test_instances = sum(
        _expected_instances(protocol, scale) for scale in _scales(protocol)
    )
    if int(verification.get("test_instance_count", -1)) != expected_test_instances:
        raise PaperTableError("verification test-instance count disagrees with the protocol")
    if not smoke and expected_test_instances != 200:
        raise PaperTableError("formal mode requires exactly 200 test instances")
    test_sets = manifest.get("test_sets")
    if not isinstance(test_sets, Mapping) or set(test_sets) != set(_scales(protocol)):
        raise PaperTableError("manifest test-set registry does not match the protocol")
    for scale in _scales(protocol):
        if not isinstance(test_sets[scale], list) or len(test_sets[scale]) != _expected_instances(
            protocol, scale
        ):
            raise PaperTableError(f"manifest test-set size is invalid for {scale}")
    models = manifest.get("models")
    if not isinstance(models, Mapping) or set(models) != {"Train-S", "Train-L"}:
        raise PaperTableError("manifest does not contain exactly two frozen models")


def _load_inputs(
    run_dir: Path, *, allow_smoke: bool = False
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, pd.DataFrame],
    dict[str, str],
    dict[str, int],
    dict[str, Any],
]:
    root = run_dir.resolve()
    manifest_path = root / "manifest.json"
    protocol_path = root / "protocol_config.json"
    manifest = _read_json(manifest_path)
    protocol = _read_json(protocol_path)
    smoke_value = manifest.get("smoke")
    if not isinstance(smoke_value, bool):
        raise PaperTableError("manifest.smoke must be an explicit boolean")
    smoke = smoke_value
    if smoke and not allow_smoke:
        raise PaperTableError("smoke runs are rejected by default; pass --allow-smoke for tests")
    if manifest.get("stage") != "verified":
        raise PaperTableError(f"manifest stage must be verified, received {manifest.get('stage')!r}")
    expected_counts = _validate_protocol_shape(protocol, smoke=smoke)
    protocol_hash = _canonical_json_sha256(protocol)
    if manifest.get("protocol_sha256") != protocol_hash:
        raise PaperTableError("canonical protocol hash does not match manifest.protocol_sha256")
    verification_record = manifest.get("verification")
    if not isinstance(verification_record, Mapping) or verification_record.get("passed") is not True:
        raise PaperTableError("manifest does not record a passed verification")
    verification_path = _recorded_path(
        root,
        verification_record.get("file"),
        "manifest verification",
        exact="verification/verification_report.json",
    )
    verification_hash = _sha256(verification_path)
    if verification_hash != verification_record.get("file_sha256"):
        raise PaperTableError("verification report hash disagrees with the manifest record")
    verification = _read_json(verification_path)
    if verification.get("passed") is not True:
        raise PaperTableError("paper tables require a verification report with passed=true")
    protocol_ids = {
        str(protocol.get("protocol_id")),
        str(manifest.get("protocol_id")),
        str(verification.get("protocol_id")),
    }
    if len(protocol_ids) != 1 or "None" in protocol_ids:
        raise PaperTableError("protocol_id differs across protocol, manifest, and verification")
    protocol_id = next(iter(protocol_ids))
    expected_protocol_id = (
        "supplementary-experiment-v1-smoke" if smoke else "supplementary-experiment-v1"
    )
    if protocol_id != expected_protocol_id:
        raise PaperTableError(
            f"unexpected {'smoke' if smoke else 'formal'} protocol_id: {protocol_id}"
        )

    schedule_record = manifest.get("schedule")
    if not isinstance(schedule_record, Mapping):
        raise PaperTableError("manifest has no schedule record")
    schedule_path = _recorded_path(
        root, schedule_record.get("file"), "manifest schedule", exact="schedule.json"
    )
    schedule_hash = _sha256(schedule_path)
    if schedule_hash != schedule_record.get("file_sha256"):
        raise PaperTableError("schedule hash disagrees with the manifest record")
    if schedule_hash != verification.get("schedule_file_sha256"):
        raise PaperTableError("schedule hash disagrees with the verification report")
    schedule = _read_json(schedule_path)
    schedule_cell_ids = _validate_schedule(
        schedule,
        protocol=protocol,
        protocol_hash=protocol_hash,
        protocol_id=protocol_id,
        expected_counts=expected_counts,
    )
    if int(schedule_record.get("cell_count", -1)) != expected_counts["total"]:
        raise PaperTableError("manifest schedule cell_count disagrees with the protocol")

    _validate_report_counts(
        verification,
        manifest=manifest,
        protocol=protocol,
        expected_counts=expected_counts,
        smoke=smoke,
    )
    manifest_tables = manifest.get("tables")
    report_hashes = verification.get("table_files_sha256")
    if not isinstance(manifest_tables, Mapping) or set(manifest_tables) != set(SUMMARY_TABLE_FILES):
        raise PaperTableError("manifest table registry does not contain the exact summary table set")
    if not isinstance(report_hashes, Mapping) or set(report_hashes) != set(SUMMARY_TABLE_FILES):
        raise PaperTableError("verification table hash registry does not match the exact table set")
    tables: dict[str, pd.DataFrame] = {}
    input_hashes = {
        "manifest.json": _sha256(manifest_path),
        "protocol_config.json": _sha256(protocol_path),
        verification_path.relative_to(root).as_posix(): verification_hash,
        schedule_path.relative_to(root).as_posix(): schedule_hash,
    }
    for name in SUMMARY_TABLE_FILES:
        item = manifest_tables[name]
        if not isinstance(item, Mapping):
            raise PaperTableError(f"manifest table record is invalid: {name}")
        path = _recorded_path(
            root, item.get("file"), f"manifest table {name}", exact=f"tables/{name}"
        )
        observed_hash = _sha256(path)
        if observed_hash != item.get("file_sha256") or observed_hash != report_hashes.get(name):
            raise PaperTableError(f"verified table binding changed: {name}")
        input_hashes[path.relative_to(root).as_posix()] = observed_hash
        tables[name] = pd.read_csv(path)

    ledger = tables["cell_file_ledger.csv"]
    _require_columns(ledger, "cell_file_ledger.csv", {"cell_id", "status", "cell_file_sha256"})
    if ledger["cell_id"].astype(str).tolist() != schedule_cell_ids:
        raise PaperTableError("cell ledger IDs/order do not match the locked schedule")
    if not ledger["status"].astype(str).eq("complete").all():
        raise PaperTableError("verified cell ledger contains a non-complete status")
    ledger_digests = ledger["cell_file_sha256"].astype(str)
    if not ledger_digests.map(
        lambda value: len(value) == 64 and all(character in "0123456789abcdef" for character in value)
    ).all():
        raise PaperTableError("cell ledger contains an invalid SHA-256 digest")
    ledger_rows = [
        {
            "cell_id": str(row.cell_id),
            "status": str(row.status),
            "cell_file_sha256": (
                None if pd.isna(row.cell_file_sha256) else str(row.cell_file_sha256)
            ),
        }
        for row in ledger.itertuples(index=False)
    ]
    ledger_hash = _canonical_json_sha256(ledger_rows)
    if (
        ledger_hash != manifest.get("summary_input_cell_ledger_sha256")
        or ledger_hash != verification.get("summary_input_cell_ledger_sha256")
    ):
        raise PaperTableError("canonical cell-ledger hash is not bound to summary and verification")

    _require_columns(
        tables["summary_by_preference.csv"],
        "summary_by_preference.csv",
        {
            "method",
            "model_id",
            "test_scale",
            "preference_id",
            "n_instances_total",
            "n_instances_quality",
            "strict_feasible_rate",
            "all_restarts_feasible_rate",
            "terminal_clear_rate",
            "technical_failure_rate",
            "missing_rate",
            "cost_weight",
            "risk_weight",
            "runtime_seconds_mean",
            "runtime_seconds_sd",
            *(
                f"{field}_{suffix}"
                for field in PAPER_QUALITY_FIELDS
                for suffix in ("mean", "sd")
            ),
        },
    )
    _require_columns(
        tables["instance_level_results.csv"],
        "instance_level_results.csv",
        {
            "method",
            "model_id",
            "test_scale",
            "instance_index",
            "preference_id",
            "cost_weight",
            "risk_weight",
            "feasible_rate",
            "all_restarts_feasible",
            "terminal_clear_rate",
            "technical_failure_rate",
            "missing_rate",
            "runtime_seconds",
            "solver_status",
            *PAPER_QUALITY_FIELDS,
        },
    )
    _require_columns(
        tables["paired_ppo_vs_ga.csv"],
        "paired_ppo_vs_ga.csv",
        {
            "model_id",
            "test_scale",
            "instance_index",
            "preference_id",
            "ppo_j",
            "ga_j",
            "delta_j_percent",
        },
    )
    _require_columns(
        tables["paired_ppo_vs_ga_summary.csv"],
        "paired_ppo_vs_ga_summary.csv",
        {
            "model_id",
            "test_scale",
            "preference_id",
            "expected_paired_n",
            "paired_n",
            "mean_delta_j_percent",
            "sample_sd_delta_j_percent",
        },
    )
    _require_columns(
        tables["milp_gap_summary.csv"],
        "milp_gap_summary.csv",
        {
            "method",
            "model_id",
            "preference_id",
            "optimal_reference_n",
            "valid_n",
            "mean_gap_percent",
            "sample_sd_gap_percent",
        },
    )
    _require_columns(
        tables["milp_gap_by_instance.csv"],
        "milp_gap_by_instance.csv",
        {
            "method",
            "model_id",
            "instance_index",
            "preference_id",
            "weighted_objective_normalized",
            "milp_j",
            "milp_status",
            "gap_percent",
        },
    )
    _require_columns(
        tables["generalization_operational.csv"],
        "generalization_operational.csv",
        {
            "model_id",
            "test_scale",
            "independent_instances",
            "instance_preference_cells",
            "quality_cells",
            "macro_j",
            "strict_feasible_rate",
            "terminal_clear_rate",
            "mean_inference_seconds",
            "relative_ga_improvement_percent",
            "expected_paired_ga_n",
            "paired_ga_n",
            "two_model_relative_gap_percent",
        },
    )
    _validate_summary_semantics(
        protocol,
        tables["summary_by_preference.csv"],
        tables["instance_level_results.csv"],
    )
    _validate_paired_semantics(
        protocol,
        tables["instance_level_results.csv"],
        tables["paired_ppo_vs_ga.csv"],
        tables["paired_ppo_vs_ga_summary.csv"],
    )
    _validate_gap_semantics(
        protocol,
        tables["instance_level_results.csv"],
        tables["milp_gap_by_instance.csv"],
        tables["milp_gap_summary.csv"],
    )
    return manifest, protocol, tables, input_hashes, expected_counts, verification


def _preferences(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    values = protocol.get("preferences")
    if not isinstance(values, list) or not values:
        raise PaperTableError("protocol has no preferences")
    result: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, dict) or not {"id", "cost_weight", "risk_weight"} <= value.keys():
            raise PaperTableError("invalid preference in protocol")
        result.append(value)
    return result


def _scales(protocol: Mapping[str, Any]) -> list[str]:
    values = protocol.get("test_scales")
    if not isinstance(values, dict) or not values:
        raise PaperTableError("protocol has no test scales")
    return list(values)


def _expected_instances(protocol: Mapping[str, Any], scale: str) -> int:
    try:
        value = int(protocol["test_scales"][scale]["instances"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PaperTableError(f"invalid instance count for {scale}") from exc
    if value <= 0:
        raise PaperTableError(f"instance count must be positive for {scale}")
    return value


def _finite_numeric(series: pd.Series) -> pd.Series:
    numbers = pd.to_numeric(series, errors="coerce")
    finite = numbers.notna() & numbers.map(
        lambda value: False if pd.isna(value) else math.isfinite(float(value))
    )
    return numbers.where(finite)


def _integer(value: Any, label: str) -> int:
    number = _finite(value)
    if number is None or not float(number).is_integer():
        raise PaperTableError(f"{label} is not an integer")
    return int(number)


def _assert_instance_indices(group: pd.DataFrame, expected_n: int, label: str) -> None:
    values = _finite_numeric(group["instance_index"])
    if len(group) != expected_n or values.isna().any():
        raise PaperTableError(f"{label}: expected {expected_n} instance rows")
    indices = [int(value) for value in values]
    if any(float(value) != index for value, index in zip(values, indices)):
        raise PaperTableError(f"{label}: instance indices are not integers")
    if sorted(indices) != list(range(expected_n)):
        raise PaperTableError(f"{label}: instance indices are missing or duplicated")


def _summary_strata(
    protocol: Mapping[str, Any],
) -> dict[tuple[str, str, str, str], int]:
    result: dict[tuple[str, str, str, str], int] = {}
    preference_ids = [str(item["id"]) for item in _preferences(protocol)]
    for method, scope in protocol["methods"].items():
        for model_id in scope["models"]:
            for scale in scope["scales"]:
                for preference_id in preference_ids:
                    key = (str(method), str(model_id), str(scale), preference_id)
                    if key in result:
                        raise PaperTableError(f"protocol duplicates a summary stratum: {key}")
                    result[key] = _expected_instances(protocol, str(scale))
    return result


def _frame_key_set(
    frame: pd.DataFrame, columns: Sequence[str], label: str
) -> set[tuple[str, ...]]:
    keys = [tuple(str(row[column]) for column in columns) for _, row in frame.iterrows()]
    if len(keys) != len(set(keys)):
        raise PaperTableError(f"{label} contains duplicated stratum rows")
    return set(keys)


def _validate_summary_semantics(
    protocol: Mapping[str, Any], summary: pd.DataFrame, instance: pd.DataFrame
) -> None:
    """Recompute every published primary summary from instance-level rows."""

    key_columns = ("method", "model_id", "test_scale", "preference_id")
    expected = _summary_strata(protocol)
    observed_summary = _frame_key_set(summary, key_columns, "summary_by_preference.csv")
    observed_instance = {
        tuple(str(row[column]) for column in key_columns)
        for _, row in instance.iterrows()
    }
    if observed_summary != set(expected) or observed_instance != set(expected):
        raise PaperTableError("summary/instance strata do not exactly match the protocol")

    preferences = {str(item["id"]): item for item in _preferences(protocol)}
    rate_pairs = (
        ("strict_feasible_rate", "feasible_rate"),
        ("all_restarts_feasible_rate", "all_restarts_feasible"),
        ("terminal_clear_rate", "terminal_clear_rate"),
        ("technical_failure_rate", "technical_failure_rate"),
        ("missing_rate", "missing_rate"),
    )
    for key, expected_n in expected.items():
        method, model_id, scale, preference_id = key
        group = instance[
            (instance["method"].astype(str) == method)
            & (instance["model_id"].astype(str) == model_id)
            & (instance["test_scale"].astype(str) == scale)
            & (instance["preference_id"].astype(str) == preference_id)
        ]
        label = "/".join(key)
        _assert_instance_indices(group, expected_n, label)
        source = _only_row(
            summary,
            "summary_by_preference.csv",
            method=method,
            model_id=model_id,
            test_scale=scale,
            preference_id=preference_id,
        )
        if _integer(source["n_instances_total"], f"{label} n_instances_total") != expected_n:
            raise PaperTableError(f"{label}: n_instances_total disagrees with the protocol")

        preference = preferences[preference_id]
        for column, expected_weight in (
            ("cost_weight", float(preference["cost_weight"])),
            ("risk_weight", float(preference["risk_weight"])),
        ):
            _assert_close(f"{label} summary {column}", source[column], expected_weight)
            values = _finite_numeric(group[column])
            if values.isna().any() or not all(
                math.isclose(float(value), expected_weight, rel_tol=1e-12, abs_tol=1e-12)
                for value in values
            ):
                raise PaperTableError(f"{label}: instance {column} disagrees with the protocol")

        quality_reference = _finite_numeric(
            group["weighted_objective_normalized"]
        ).notna()
        quality_n = int(quality_reference.sum())
        if _integer(source["n_instances_quality"], f"{label} n_instances_quality") != quality_n:
            raise PaperTableError(f"{label}: n_instances_quality disagrees with instance rows")
        for field in PAPER_QUALITY_FIELDS:
            values = _finite_numeric(group[field])
            if not values.notna().equals(quality_reference):
                raise PaperTableError(f"{label}: primary quality fields have unequal coverage")
            fully_observed = quality_n == expected_n
            expected_mean = float(values.mean()) if fully_observed else None
            expected_sd = (
                float(values.std(ddof=1)) if fully_observed and expected_n >= 2 else None
            )
            _assert_close(f"{label} {field}_mean", source[f"{field}_mean"], expected_mean)
            _assert_close(f"{label} {field}_sd", source[f"{field}_sd"], expected_sd)

        # The upstream instance reducer emits primary quality only when every
        # scheduled restart is complete and feasible.  Enforce that implication
        # here so a finite J can never be paired with an infeasible/failed row,
        # even if all downstream CSV hashes were consistently rewritten.
        quality_implications = (
            ("feasible_rate", 1.0),
            ("all_restarts_feasible", 1.0),
            ("technical_failure_rate", 0.0),
            ("missing_rate", 0.0),
        )
        for column, required_value in quality_implications:
            values = _finite_numeric(group[column])
            violates = quality_reference & (
                values.isna() | ((values - required_value).abs() > 1e-12)
            )
            if bool(violates.any()):
                raise PaperTableError(
                    f"{label}: finite primary quality requires complete, strictly feasible restarts"
                )

        for summary_column, instance_column in rate_pairs:
            values = _finite_numeric(group[instance_column])
            if values.isna().any() or ((values < 0.0) | (values > 1.0)).any():
                raise PaperTableError(f"{label}: invalid instance-level {instance_column}")
            _assert_close(
                f"{label} {summary_column}", source[summary_column], float(values.mean())
            )

        runtimes = _finite_numeric(group["runtime_seconds"])
        complete_runtime = bool(runtimes.notna().all())
        expected_runtime_mean = float(runtimes.mean()) if complete_runtime else None
        expected_runtime_sd = (
            float(runtimes.std(ddof=1))
            if complete_runtime and expected_n >= 2
            else None
        )
        _assert_close(
            f"{label} runtime_seconds_mean",
            source["runtime_seconds_mean"],
            expected_runtime_mean,
        )
        _assert_close(
            f"{label} runtime_seconds_sd",
            source["runtime_seconds_sd"],
            expected_runtime_sd,
        )


def _validate_paired_semantics(
    protocol: Mapping[str, Any],
    instance: pd.DataFrame,
    paired: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> None:
    """Bind paired PPO-vs-GA rows and primary delta summaries to instance data."""

    preference_ids = [str(item["id"]) for item in _preferences(protocol)]
    ppo_scope = protocol["methods"]["ppo"]
    expected: dict[tuple[str, str, str], int] = {
        (str(model), str(scale), preference_id): _expected_instances(protocol, str(scale))
        for model in ppo_scope["models"]
        for scale in ppo_scope["scales"]
        for preference_id in preference_ids
    }
    key_columns = ("model_id", "test_scale", "preference_id")
    paired_keys = {
        tuple(str(row[column]) for column in key_columns) for _, row in paired.iterrows()
    }
    summary_keys = _frame_key_set(
        paired_summary, key_columns, "paired_ppo_vs_ga_summary.csv"
    )
    if paired_keys != set(expected) or summary_keys != set(expected):
        raise PaperTableError("paired PPO/GA strata do not exactly match the protocol")

    for key, expected_n in expected.items():
        model_id, scale, preference_id = key
        label = f"paired/{model_id}/{scale}/{preference_id}"
        group = paired[
            (paired["model_id"].astype(str) == model_id)
            & (paired["test_scale"].astype(str) == scale)
            & (paired["preference_id"].astype(str) == preference_id)
        ]
        _assert_instance_indices(group, expected_n, label)
        ppo = instance[
            (instance["method"].astype(str) == "ppo")
            & (instance["model_id"].astype(str) == model_id)
            & (instance["test_scale"].astype(str) == scale)
            & (instance["preference_id"].astype(str) == preference_id)
        ].set_index("instance_index")
        ga = instance[
            (instance["method"].astype(str) == "ga")
            & (instance["model_id"].astype(str) == "GA")
            & (instance["test_scale"].astype(str) == scale)
            & (instance["preference_id"].astype(str) == preference_id)
        ].set_index("instance_index")
        expected_deltas: list[float] = []
        for _, row in group.iterrows():
            index = int(row["instance_index"])
            ppo_j = ppo.loc[index, "weighted_objective_normalized"]
            ga_j = ga.loc[index, "weighted_objective_normalized"]
            _assert_close(f"{label}/{index} ppo_j", row["ppo_j"], ppo_j)
            _assert_close(f"{label}/{index} ga_j", row["ga_j"], ga_j)
            ppo_value = _finite(ppo_j)
            ga_value = _finite(ga_j)
            expected_delta = (
                (ga_value - ppo_value) / ga_value * 100.0
                if ppo_value is not None
                and ga_value is not None
                and abs(ga_value) > 1e-12
                else None
            )
            _assert_close(f"{label}/{index} delta_j_percent", row["delta_j_percent"], expected_delta)
            if expected_delta is not None:
                expected_deltas.append(expected_delta)

        source = _only_row(
            paired_summary,
            "paired_ppo_vs_ga_summary.csv",
            model_id=model_id,
            test_scale=scale,
            preference_id=preference_id,
        )
        if _integer(source["expected_paired_n"], f"{label} expected_paired_n") != expected_n:
            raise PaperTableError(f"{label}: expected paired count changed")
        if _integer(source["paired_n"], f"{label} paired_n") != len(expected_deltas):
            raise PaperTableError(f"{label}: paired_n disagrees with paired rows")
        complete = len(expected_deltas) == expected_n
        expected_mean = sum(expected_deltas) / expected_n if complete else None
        expected_sd = (
            float(pd.Series(expected_deltas, dtype=float).std(ddof=1))
            if complete and expected_n >= 2
            else None
        )
        _assert_close(f"{label} mean_delta_j_percent", source["mean_delta_j_percent"], expected_mean)
        _assert_close(
            f"{label} sample_sd_delta_j_percent",
            source["sample_sd_delta_j_percent"],
            expected_sd,
        )


def _text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _validate_gap_semantics(
    protocol: Mapping[str, Any],
    instance: pd.DataFrame,
    gaps: pd.DataFrame,
    gap_summary: pd.DataFrame,
) -> None:
    """Apply one optimal-only Gap rule to PPO, GA, and heuristic outputs."""

    candidates = (
        ("ppo", "Train-S"),
        ("ga", "GA"),
        ("heuristic", "Heuristic"),
    )
    preference_ids = [str(item["id"]) for item in _preferences(protocol)]
    expected_keys = {
        (method, model_id, preference_id)
        for method, model_id in candidates
        for preference_id in preference_ids
    }
    key_columns = ("method", "model_id", "preference_id")
    gap_keys = {tuple(str(row[column]) for column in key_columns) for _, row in gaps.iterrows()}
    summary_keys = _frame_key_set(gap_summary, key_columns, "milp_gap_summary.csv")
    if gap_keys != expected_keys or summary_keys != expected_keys:
        raise PaperTableError("MILP Gap strata do not exactly match the optimal-only scope")

    expected_n = _expected_instances(protocol, "Test-1")
    for method, model_id in candidates:
        for preference_id in preference_ids:
            label = f"Gap/{method}/{model_id}/{preference_id}"
            group = gaps[
                (gaps["method"].astype(str) == method)
                & (gaps["model_id"].astype(str) == model_id)
                & (gaps["preference_id"].astype(str) == preference_id)
            ]
            _assert_instance_indices(group, expected_n, label)
            candidate = instance[
                (instance["method"].astype(str) == method)
                & (instance["model_id"].astype(str) == model_id)
                & (instance["test_scale"].astype(str) == "Test-1")
                & (instance["preference_id"].astype(str) == preference_id)
            ].set_index("instance_index")
            milp = instance[
                (instance["method"].astype(str) == "milp")
                & (instance["model_id"].astype(str) == "MILP")
                & (instance["test_scale"].astype(str) == "Test-1")
                & (instance["preference_id"].astype(str) == preference_id)
            ].set_index("instance_index")
            expected_gaps: list[float] = []
            optimal_n = 0
            for _, row in group.iterrows():
                index = int(row["instance_index"])
                candidate_j = candidate.loc[index, "weighted_objective_normalized"]
                milp_j = milp.loc[index, "weighted_objective_normalized"]
                milp_status = _text(milp.loc[index, "solver_status"])
                _assert_close(
                    f"{label}/{index} candidate J",
                    row["weighted_objective_normalized"],
                    candidate_j,
                )
                _assert_close(f"{label}/{index} MILP J", row["milp_j"], milp_j)
                if _text(row["milp_status"]) != milp_status:
                    raise PaperTableError(f"{label}/{index}: MILP status changed")
                if milp_status.casefold() == "optimal":
                    optimal_n += 1
                candidate_value = _finite(candidate_j)
                milp_value = _finite(milp_j)
                expected_gap = (
                    (candidate_value - milp_value) / abs(milp_value) * 100.0
                    if milp_status.casefold() == "optimal"
                    and candidate_value is not None
                    and milp_value is not None
                    and abs(milp_value) > 1e-12
                    else None
                )
                _assert_close(f"{label}/{index} gap_percent", row["gap_percent"], expected_gap)
                if expected_gap is not None:
                    expected_gaps.append(expected_gap)

            source = _only_row(
                gap_summary,
                "milp_gap_summary.csv",
                method=method,
                model_id=model_id,
                preference_id=preference_id,
            )
            if _integer(source["optimal_reference_n"], f"{label} optimal_reference_n") != optimal_n:
                raise PaperTableError(f"{label}: optimal reference count changed")
            if _integer(source["valid_n"], f"{label} valid_n") != len(expected_gaps):
                raise PaperTableError(f"{label}: valid Gap count changed")
            complete = optimal_n > 0 and len(expected_gaps) == optimal_n
            expected_mean = (
                sum(expected_gaps) / len(expected_gaps) if complete else None
            )
            expected_sd = (
                float(pd.Series(expected_gaps, dtype=float).std(ddof=1))
                if complete and len(expected_gaps) >= 2
                else None
            )
            _assert_close(f"{label} mean_gap_percent", source["mean_gap_percent"], expected_mean)
            _assert_close(
                f"{label} sample_sd_gap_percent",
                source["sample_sd_gap_percent"],
                expected_sd,
            )


def _preference_label(preference: Mapping[str, Any]) -> str:
    return f"({float(preference['cost_weight']):g},{float(preference['risk_weight']):g})"


def _training_seconds(manifest: Mapping[str, Any], model_id: str) -> float | None:
    try:
        return _finite(manifest["models"][model_id]["training_seconds"])
    except (KeyError, TypeError):
        return None


def _run_method_tokens(value: Any) -> set[str]:
    if isinstance(value, str):
        return {token.strip().casefold() for token in value.replace(",", " ").split() if token.strip()}
    if isinstance(value, Sequence):
        return {str(token).strip().casefold() for token in value if str(token).strip()}
    return set()


def _method_in_run(value: Any, method: str) -> bool:
    tokens = _run_method_tokens(value)
    return "all" in tokens or method.casefold() in tokens


def _scope_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return ",".join(str(item) for item in value)
    return "?"


def _batch_description(index: int, run: Mapping[str, Any], method: str) -> str:
    tokens = _run_method_tokens(run.get("methods"))
    methods_text = "all" if "all" in tokens else ",".join(sorted(tokens)) or "?"
    completion_label = (
        "整批总完成数" if "all" in tokens or len(tokens) != 1 else "该方法完成数"
    )
    wall = _finite(run.get("wall_seconds"))
    text = (
        f"batch{index}[methods={methods_text}; models={_scope_text(run.get('models'))}; "
        f"scales={_scope_text(run.get('scales'))}; workers={run.get('workers', '?')}; "
        f"{completion_label}={run.get('completed_cells', '?')}"
    )
    if wall is not None:
        text += f"; 整批墙钟={wall:.3f}s"
    if run.get("max_cells") is not None:
        text += f"; max_cells={run.get('max_cells')}"
    if method == "ppo":
        text += f"; concurrent_ppo={str(bool(run.get('allow_concurrent_ppo'))).lower()}"
    return text + "]"


def _concurrency_scope(manifest: Mapping[str, Any], method: str) -> str:
    included: list[str] = []
    excluded: list[str] = []
    included_wall = 0.0
    excluded_wall = 0.0
    for index, run in enumerate(manifest.get("evaluation_runs", []), start=1):
        if not isinstance(run, Mapping) or int(run.get("completed_cells", 0) or 0) <= 0:
            continue
        if not _method_in_run(run.get("methods"), method):
            continue
        wall = _finite(run.get("wall_seconds"))
        description = _batch_description(index, run, method)
        if run.get("superseded_for_timing") is True:
            excluded.append(description)
            excluded_wall += wall or 0.0
        else:
            included.append(description)
            included_wall += wall or 0.0
    if not included and not excluded:
        return "未记录完成单元的批次"
    parts = [
        (
            f"纳入最终时间口径批次={len(included)}，整批墙钟合计={included_wall:.3f}s："
            + ("；".join(included) if included else "无")
        )
    ]
    parts.append(
        f"已排除superseded_for_timing历史批次={len(excluded)}，"
        f"整批墙钟合计={excluded_wall:.3f}s："
        + ("；".join(excluded) if excluded else "无")
    )
    return "；".join(parts)


def _summary_row(
    summary: pd.DataFrame,
    *,
    method: str,
    model_id: str,
    scale: str,
    preference_id: str,
) -> pd.Series:
    return _only_row(
        summary,
        "summary_by_preference.csv",
        method=method,
        model_id=model_id,
        test_scale=scale,
        preference_id=preference_id,
    )


def _primary_instance_macro(
    frame: pd.DataFrame,
    *,
    value_column: str,
    expected_preferences: Sequence[str],
    expected_instances: int,
) -> PrimaryStats:
    """Aggregate preferences within instance, then independent instances.

    A value contributes only when every expected preference is present and all
    underlying primary values are finite.  The reported mean/SD is suppressed
    unless all expected independent instances contribute.
    """

    expected_set = set(expected_preferences)
    values: list[float] = []
    for instance_index in range(expected_instances):
        group = frame[frame["instance_index"] == instance_index]
        observed_set = set(group["preference_id"].astype(str))
        series = pd.to_numeric(group[value_column], errors="coerce")
        if len(group) == len(expected_preferences) and observed_set == expected_set and series.notna().all():
            values.append(float(series.mean()))
    valid_n = len(values)
    if valid_n != expected_instances:
        return PrimaryStats(None, None, valid_n, expected_instances)
    mean = float(pd.Series(values, dtype=float).mean())
    sd = float(pd.Series(values, dtype=float).std(ddof=1)) if valid_n >= 2 else None
    return PrimaryStats(mean, sd, valid_n, expected_instances)


def _single_preference_stats(
    instance: pd.DataFrame,
    *,
    method: str,
    model_id: str,
    scale: str,
    preference_id: str,
    value_column: str,
    expected_instances: int,
) -> PrimaryStats:
    group = instance[
        (instance["method"] == method)
        & (instance["model_id"] == model_id)
        & (instance["test_scale"] == scale)
        & (instance["preference_id"] == preference_id)
    ]
    values = pd.to_numeric(group[value_column], errors="coerce").dropna()
    valid_n = int(len(values))
    if len(group) != expected_instances or valid_n != expected_instances:
        return PrimaryStats(None, None, valid_n, expected_instances)
    return PrimaryStats(
        float(values.mean()),
        float(values.std(ddof=1)) if valid_n >= 2 else None,
        valid_n,
        expected_instances,
    )


def _outcome_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Count distinct instances affected by mutually interpretable outcomes.

    Technical failure and missingness take precedence.  ``infeasible`` therefore
    means an evaluated instance-preference row with no technical/missing cell but
    at least one restart failing strict feasibility.  ``incomplete_quality`` is
    reported separately and may overlap the three diagnostic categories.
    """

    if frame.empty:
        return {
            "technical_failure_instances": 0,
            "infeasible_instances": 0,
            "missing_instances": 0,
            "incomplete_quality_instances": 0,
            "technical_failure_instance_preferences": 0,
            "infeasible_instance_preferences": 0,
            "missing_instance_preferences": 0,
            "incomplete_quality_instance_preferences": 0,
        }
    technical = pd.to_numeric(frame["technical_failure_rate"], errors="coerce").fillna(0) > 0
    missing = pd.to_numeric(frame["missing_rate"], errors="coerce").fillna(0) > 0
    feasible = pd.to_numeric(frame["feasible_rate"], errors="coerce").fillna(0)
    infeasible = (feasible < 1.0 - 1e-12) & ~technical & ~missing
    incomplete = pd.to_numeric(
        frame["weighted_objective_normalized"], errors="coerce"
    ).isna()

    def affected_instances(mask: pd.Series) -> int:
        return int(frame.loc[mask, "instance_index"].nunique())

    return {
        "technical_failure_instances": affected_instances(technical),
        "infeasible_instances": affected_instances(infeasible),
        "missing_instances": affected_instances(missing),
        "incomplete_quality_instances": affected_instances(incomplete),
        "technical_failure_instance_preferences": int(technical.sum()),
        "infeasible_instance_preferences": int(infeasible.sum()),
        "missing_instance_preferences": int(missing.sum()),
        "incomplete_quality_instance_preferences": int(incomplete.sum()),
    }


def _outcome_text(counts: Mapping[str, int]) -> str:
    return (
        f"技术={counts['technical_failure_instances']}; "
        f"不可行={counts['infeasible_instances']}; "
        f"缺失={counts['missing_instances']}; "
        f"质量不完整={counts['incomplete_quality_instances']}"
    )


def _paired_macro_stats(
    paired: pd.DataFrame,
    *,
    model_id: str,
    scale: str,
    preference_ids: Sequence[str],
    expected_instances: int,
) -> PrimaryStats:
    group = paired[(paired["model_id"] == model_id) & (paired["test_scale"] == scale)]
    return _primary_instance_macro(
        group,
        value_column="delta_j_percent",
        expected_preferences=preference_ids,
        expected_instances=expected_instances,
    )


def _mean_sd(mean: Any, sd: Any, digits: int = 4) -> str:
    centre = _finite(mean)
    spread = _finite(sd)
    if centre is None:
        return "—"
    if spread is None:
        return f"{centre:.{digits}f}"
    return f"{centre:.{digits}f} ± {spread:.{digits}f}"


def _percent(value: Any, digits: int = 2) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:.{digits}f}%"


def _mean_sd_percent(mean: Any, sd: Any, digits: int = 2) -> str:
    text = _mean_sd(mean, sd, digits)
    return text if text == "—" else text + "%"


def _markdown_escape(value: Any) -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_table(
    *,
    title: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    notes: Sequence[str],
) -> str:
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_markdown_escape(value) for value in row) + " |")
    lines.extend(["", "注：" + " ".join(notes), ""])
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig", na_rep="", float_format="%.10g")
    os.replace(temporary, path)


def _table3(
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    summary = tables["summary_by_preference.csv"]
    instance = tables["instance_level_results.csv"]
    deltas = tables["paired_ppo_vs_ga_summary.csv"]
    gaps = tables["milp_gap_summary.csv"]
    expected_n = _expected_instances(protocol, "Test-1")
    training_seconds = _training_seconds(manifest, "Train-S")
    rows: list[dict[str, Any]] = []
    for preference in _preferences(protocol):
        preference_id = str(preference["id"])
        ppo = _summary_row(
            summary,
            method="ppo",
            model_id="Train-S",
            scale="Test-1",
            preference_id=preference_id,
        )
        ga = _summary_row(
            summary,
            method="ga",
            model_id="GA",
            scale="Test-1",
            preference_id=preference_id,
        )
        milp = _summary_row(
            summary,
            method="milp",
            model_id="MILP",
            scale="Test-1",
            preference_id=preference_id,
        )
        delta = _only_row(
            deltas,
            "paired_ppo_vs_ga_summary.csv",
            model_id="Train-S",
            test_scale="Test-1",
            preference_id=preference_id,
        )
        ppo_gap = _only_row(
            gaps,
            "milp_gap_summary.csv",
            method="ppo",
            model_id="Train-S",
            preference_id=preference_id,
        )
        ga_gap = _only_row(
            gaps,
            "milp_gap_summary.csv",
            method="ga",
            model_id="GA",
            preference_id=preference_id,
        )
        milp_instances = instance[
            (instance["method"] == "milp")
            & (instance["model_id"] == "MILP")
            & (instance["test_scale"] == "Test-1")
            & (instance["preference_id"] == preference_id)
        ]
        if len(milp_instances) != expected_n:
            raise PaperTableError(
                f"Test-1/MILP/{preference_id}: expected {expected_n} instance rows, "
                f"found {len(milp_instances)}"
            )
        optimal_n = int(milp_instances["solver_status"].astype(str).str.casefold().eq("optimal").sum())
        for label, gap_row in (("PPO", ppo_gap), ("GA", ga_gap)):
            gap_optimal_n = int(gap_row["optimal_reference_n"])
            gap_valid_n = int(gap_row["valid_n"])
            gap_mean = _finite(gap_row["mean_gap_percent"])
            if gap_optimal_n != optimal_n or gap_valid_n > optimal_n:
                raise PaperTableError(
                    f"{label} Gap eligibility disagrees with MILP optimal statuses for {preference_id}"
                )
            should_have_primary_gap = optimal_n > 0 and gap_valid_n == optimal_n
            if (gap_mean is not None) != should_have_primary_gap:
                raise PaperTableError(
                    f"{label} primary Gap violates the complete optimal-reference rule for {preference_id}"
                )
        ppo_feas = _single_preference_stats(
            instance,
            method="ppo",
            model_id="Train-S",
            scale="Test-1",
            preference_id=preference_id,
            value_column="feasible_rate",
            expected_instances=expected_n,
        )
        ga_feas = _single_preference_stats(
            instance,
            method="ga",
            model_id="GA",
            scale="Test-1",
            preference_id=preference_id,
            value_column="feasible_rate",
            expected_instances=expected_n,
        )
        ppo_instances = instance[
            (instance["method"] == "ppo")
            & (instance["model_id"] == "Train-S")
            & (instance["test_scale"] == "Test-1")
            & (instance["preference_id"] == preference_id)
        ]
        ga_instances = instance[
            (instance["method"] == "ga")
            & (instance["model_id"] == "GA")
            & (instance["test_scale"] == "Test-1")
            & (instance["preference_id"] == preference_id)
        ]
        ppo_outcomes = _outcome_counts(ppo_instances)
        ga_outcomes = _outcome_counts(ga_instances)
        outcome_fields = {
            f"ppo_{key}": value for key, value in ppo_outcomes.items()
        } | {f"ga_{key}": value for key, value in ga_outcomes.items()}
        rows.append(
            {
                "preference_id": preference_id,
                "preference": _preference_label(preference),
                "cost_weight": float(preference["cost_weight"]),
                "risk_weight": float(preference["risk_weight"]),
                "ppo_j_mean": _finite(ppo["weighted_objective_normalized_mean"]),
                "ppo_j_sd": _finite(ppo["weighted_objective_normalized_sd"]),
                "ppo_j_n": int(ppo["n_instances_quality"]),
                "ppo_j_mean_sd": _mean_sd(
                    ppo["weighted_objective_normalized_mean"],
                    ppo["weighted_objective_normalized_sd"],
                ),
                "ga_j_mean": _finite(ga["weighted_objective_normalized_mean"]),
                "ga_j_sd": _finite(ga["weighted_objective_normalized_sd"]),
                "ga_j_n": int(ga["n_instances_quality"]),
                "ga_j_mean_sd": _mean_sd(
                    ga["weighted_objective_normalized_mean"],
                    ga["weighted_objective_normalized_sd"],
                ),
                "milp_optimal_n": optimal_n,
                "milp_total_n": expected_n,
                "milp_optimal_percent": 100.0 * optimal_n / expected_n,
                "ppo_gap_mean_percent": _finite(ppo_gap["mean_gap_percent"]),
                "ppo_gap_sd_percent": _finite(ppo_gap["sample_sd_gap_percent"]),
                "ppo_gap_valid_n": int(ppo_gap["valid_n"]),
                "ppo_gap_mean_sd": _mean_sd(
                    ppo_gap["mean_gap_percent"], ppo_gap["sample_sd_gap_percent"], 2
                ),
                "ga_gap_mean_percent": _finite(ga_gap["mean_gap_percent"]),
                "ga_gap_sd_percent": _finite(ga_gap["sample_sd_gap_percent"]),
                "ga_gap_valid_n": int(ga_gap["valid_n"]),
                "ga_gap_mean_sd": _mean_sd(
                    ga_gap["mean_gap_percent"], ga_gap["sample_sd_gap_percent"], 2
                ),
                "delta_j_mean_percent": _finite(delta["mean_delta_j_percent"]),
                "delta_j_sd_percent": _finite(delta["sample_sd_delta_j_percent"]),
                "delta_j_valid_n": int(delta["paired_n"]),
                "delta_j_mean_sd": _mean_sd(
                    delta["mean_delta_j_percent"], delta["sample_sd_delta_j_percent"], 2
                ),
                "ppo_feasible_percent": _rate_percent(ppo["strict_feasible_rate"]),
                "ppo_feasible_sd_percent": (
                    None if ppo_feas.sd is None else 100.0 * ppo_feas.sd
                ),
                "ppo_feasible_n": ppo_feas.valid_n,
                "ppo_runtime_mean_seconds": _finite(ppo["runtime_seconds_mean"]),
                "ppo_runtime_sd_seconds": _finite(ppo["runtime_seconds_sd"]),
                "ppo_runtime_n": expected_n if _finite(ppo["runtime_seconds_mean"]) is not None else 0,
                "ppo_feasible_runtime": (
                    f"{_percent(_rate_percent(ppo['strict_feasible_rate']))} / "
                    f"{_mean_sd(ppo['runtime_seconds_mean'], ppo['runtime_seconds_sd'], 4)}"
                ),
                "ga_feasible_percent": _rate_percent(ga["strict_feasible_rate"]),
                "ga_feasible_sd_percent": None if ga_feas.sd is None else 100.0 * ga_feas.sd,
                "ga_feasible_n": ga_feas.valid_n,
                "ga_runtime_mean_seconds": _finite(ga["runtime_seconds_mean"]),
                "ga_runtime_sd_seconds": _finite(ga["runtime_seconds_sd"]),
                "ga_runtime_n": expected_n if _finite(ga["runtime_seconds_mean"]) is not None else 0,
                "ga_feasible_runtime": (
                    f"{_percent(_rate_percent(ga['strict_feasible_rate']))} / "
                    f"{_mean_sd(ga['runtime_seconds_mean'], ga['runtime_seconds_sd'], 4)}"
                ),
                "milp_runtime_mean_seconds": _finite(milp["runtime_seconds_mean"]),
                "milp_runtime_sd_seconds": _finite(milp["runtime_seconds_sd"]),
                "milp_runtime_n": expected_n if _finite(milp["runtime_seconds_mean"]) is not None else 0,
                "milp_runtime_mean_sd": _mean_sd(
                    milp["runtime_seconds_mean"], milp["runtime_seconds_sd"], 3
                ),
                "ppo_training_seconds_once": training_seconds,
                "ppo_runtime_concurrency_scope": _concurrency_scope(manifest, "ppo"),
                "ga_runtime_concurrency_scope": _concurrency_scope(manifest, "ga"),
                "milp_runtime_concurrency_scope": _concurrency_scope(manifest, "milp"),
                "outcome_disclosure": (
                    f"PPO[{_outcome_text(ppo_outcomes)}]; GA[{_outcome_text(ga_outcomes)}]"
                ),
                **outcome_fields,
            }
        )
    frame = pd.DataFrame(rows)
    markdown_rows = [
        (
            row["preference"],
            f"{row['ppo_j_mean_sd']} (n={row['ppo_j_n']})",
            f"{row['ga_j_mean_sd']} (n={row['ga_j_n']})",
            f"{row['milp_optimal_n']}/{row['milp_total_n']} ({row['milp_optimal_percent']:.1f}%)",
            f"{row['ppo_gap_mean_sd']} (n={row['ppo_gap_valid_n']})",
            f"{row['ga_gap_mean_sd']} (n={row['ga_gap_valid_n']})",
            f"{row['delta_j_mean_sd']} (n={row['delta_j_valid_n']})",
            row["ppo_feasible_runtime"],
            row["ga_feasible_runtime"],
            row["milp_runtime_mean_sd"],
            row["outcome_disclosure"],
        )
        for row in rows
    ]
    markdown = _markdown_table(
        title=f"表3  小规模{expected_n}个未见实例的求解质量与最优性检验",
        headers=(
            "偏好 (wC,wR)",
            "PPO J",
            "GA J",
            "MILP证明最优",
            "PPO Gap%",
            "GA Gap%",
            "ΔJ%",
            "PPO可行率/时间(s)",
            "GA可行率/时间(s)",
            "MILP时间(s)",
            "异常实例计数",
        ),
        rows=markdown_rows,
        notes=(
            f"统计单位为独立实例（计划n={expected_n}）；J、Gap、ΔJ与时间为均值±样本SD；异常计数依次为技术失败、纯求解不可行、缺失、质量不完整。",
            "Gap只使用MILP已证明optimal的实例，括号内为有效n；任何未证明optimal的incumbent（包括限时或其他中止）均不进入Gap；ΔJ为正表示PPO的J较低。",
            (
                f"Train-S一次性训练耗时为{training_seconds:.3f}s，不计入单元推理时间。"
                if training_seconds is not None
                else "Train-S一次性训练耗时未记录，且训练耗时不计入单元推理时间。"
            ),
            "PPO、GA与MILP时间均为并发条件下的求解段观测值，不能据此作隔离运行下纯算法速度的因果比较；完整批次、workers与整批墙钟见附录运行表及manifest.evaluation_runs。",
        ),
    )
    return frame, markdown


def _table4(
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    summary = tables["summary_by_preference.csv"]
    instance = tables["instance_level_results.csv"]
    deltas = tables["paired_ppo_vs_ga_summary.csv"]
    expected_n = _expected_instances(protocol, "Test-4")
    training_seconds = _training_seconds(manifest, "Train-L")
    rows: list[dict[str, Any]] = []
    for preference in _preferences(protocol):
        preference_id = str(preference["id"])
        ppo = _summary_row(
            summary,
            method="ppo",
            model_id="Train-L",
            scale="Test-4",
            preference_id=preference_id,
        )
        ga = _summary_row(
            summary,
            method="ga",
            model_id="GA",
            scale="Test-4",
            preference_id=preference_id,
        )
        delta = _only_row(
            deltas,
            "paired_ppo_vs_ga_summary.csv",
            model_id="Train-L",
            test_scale="Test-4",
            preference_id=preference_id,
        )
        ppo_feas = _single_preference_stats(
            instance,
            method="ppo",
            model_id="Train-L",
            scale="Test-4",
            preference_id=preference_id,
            value_column="feasible_rate",
            expected_instances=expected_n,
        )
        ga_feas = _single_preference_stats(
            instance,
            method="ga",
            model_id="GA",
            scale="Test-4",
            preference_id=preference_id,
            value_column="feasible_rate",
            expected_instances=expected_n,
        )
        ppo_instances = instance[
            (instance["method"] == "ppo")
            & (instance["model_id"] == "Train-L")
            & (instance["test_scale"] == "Test-4")
            & (instance["preference_id"] == preference_id)
        ]
        ga_instances = instance[
            (instance["method"] == "ga")
            & (instance["model_id"] == "GA")
            & (instance["test_scale"] == "Test-4")
            & (instance["preference_id"] == preference_id)
        ]
        ppo_outcomes = _outcome_counts(ppo_instances)
        ga_outcomes = _outcome_counts(ga_instances)
        outcome_fields = {
            f"ppo_{key}": value for key, value in ppo_outcomes.items()
        } | {f"ga_{key}": value for key, value in ga_outcomes.items()}
        rows.append(
            {
                "preference_id": preference_id,
                "preference": _preference_label(preference),
                "cost_weight": float(preference["cost_weight"]),
                "risk_weight": float(preference["risk_weight"]),
                "ppo_cost_mean": _finite(ppo["cost_mean"]),
                "ppo_cost_sd": _finite(ppo["cost_sd"]),
                "ppo_cost_n": int(ppo["n_instances_quality"]),
                "ppo_cost_mean_sd": _mean_sd(ppo["cost_mean"], ppo["cost_sd"], 2),
                "ppo_risk_mean": _finite(ppo["risk_mean"]),
                "ppo_risk_sd": _finite(ppo["risk_sd"]),
                "ppo_risk_n": int(ppo["n_instances_quality"]),
                "ppo_risk_mean_sd": _mean_sd(ppo["risk_mean"], ppo["risk_sd"], 2),
                "ppo_j_mean": _finite(ppo["weighted_objective_normalized_mean"]),
                "ppo_j_sd": _finite(ppo["weighted_objective_normalized_sd"]),
                "ppo_j_n": int(ppo["n_instances_quality"]),
                "ppo_j_mean_sd": _mean_sd(
                    ppo["weighted_objective_normalized_mean"],
                    ppo["weighted_objective_normalized_sd"],
                ),
                "ga_j_mean": _finite(ga["weighted_objective_normalized_mean"]),
                "ga_j_sd": _finite(ga["weighted_objective_normalized_sd"]),
                "ga_j_n": int(ga["n_instances_quality"]),
                "ga_j_mean_sd": _mean_sd(
                    ga["weighted_objective_normalized_mean"],
                    ga["weighted_objective_normalized_sd"],
                ),
                "delta_j_mean_percent": _finite(delta["mean_delta_j_percent"]),
                "delta_j_sd_percent": _finite(delta["sample_sd_delta_j_percent"]),
                "delta_j_valid_n": int(delta["paired_n"]),
                "delta_j_mean_sd": _mean_sd(
                    delta["mean_delta_j_percent"], delta["sample_sd_delta_j_percent"], 2
                ),
                "ppo_feasible_percent": _rate_percent(ppo["strict_feasible_rate"]),
                "ppo_feasible_sd_percent": None if ppo_feas.sd is None else 100.0 * ppo_feas.sd,
                "ppo_feasible_n": ppo_feas.valid_n,
                "ppo_runtime_mean_seconds": _finite(ppo["runtime_seconds_mean"]),
                "ppo_runtime_sd_seconds": _finite(ppo["runtime_seconds_sd"]),
                "ppo_runtime_n": expected_n if _finite(ppo["runtime_seconds_mean"]) is not None else 0,
                "ppo_feasible_runtime": (
                    f"{_percent(_rate_percent(ppo['strict_feasible_rate']))} / "
                    f"{_mean_sd(ppo['runtime_seconds_mean'], ppo['runtime_seconds_sd'], 4)}"
                ),
                "ga_feasible_percent": _rate_percent(ga["strict_feasible_rate"]),
                "ga_feasible_sd_percent": None if ga_feas.sd is None else 100.0 * ga_feas.sd,
                "ga_feasible_n": ga_feas.valid_n,
                "ga_runtime_mean_seconds": _finite(ga["runtime_seconds_mean"]),
                "ga_runtime_sd_seconds": _finite(ga["runtime_seconds_sd"]),
                "ga_runtime_n": expected_n if _finite(ga["runtime_seconds_mean"]) is not None else 0,
                "ga_feasible_runtime": (
                    f"{_percent(_rate_percent(ga['strict_feasible_rate']))} / "
                    f"{_mean_sd(ga['runtime_seconds_mean'], ga['runtime_seconds_sd'], 4)}"
                ),
                "ppo_training_seconds_once": training_seconds,
                "ppo_runtime_concurrency_scope": _concurrency_scope(manifest, "ppo"),
                "ga_runtime_concurrency_scope": _concurrency_scope(manifest, "ga"),
                "outcome_disclosure": (
                    f"PPO[{_outcome_text(ppo_outcomes)}]; GA[{_outcome_text(ga_outcomes)}]"
                ),
                **outcome_fields,
            }
        )
    frame = pd.DataFrame(rows)
    markdown_rows = [
        (
            row["preference"],
            f"{row['ppo_cost_mean_sd']} (n={row['ppo_cost_n']})",
            f"{row['ppo_risk_mean_sd']} (n={row['ppo_risk_n']})",
            f"{row['ppo_j_mean_sd']} (n={row['ppo_j_n']})",
            f"{row['ga_j_mean_sd']} (n={row['ga_j_n']})",
            f"{row['delta_j_mean_sd']} (n={row['delta_j_valid_n']})",
            row["ppo_feasible_runtime"],
            row["ga_feasible_runtime"],
            row["outcome_disclosure"],
        )
        for row in rows
    ]
    markdown = _markdown_table(
        title=f"表4  大规模{expected_n}个未见实例的算法性能比较",
        headers=(
            "偏好 (wC,wR)",
            "PPO成本",
            "PPO风险",
            "PPO J",
            "GA J",
            "ΔJ%",
            "PPO可行率/时间(s)",
            "GA可行率/时间(s)",
            "异常实例计数",
        ),
        rows=markdown_rows,
        notes=(
            f"统计单位为独立实例（计划n={expected_n}）；成本、风险、J、ΔJ与时间为均值±样本SD；异常计数依次为技术失败、纯求解不可行、缺失、质量不完整。",
            "ΔJ=(J_GA−J_PPO)/J_GA×100%，正值表示PPO目标较低。",
            (
                f"Train-L一次性训练耗时为{training_seconds:.3f}s，不计入单元推理时间。"
                if training_seconds is not None
                else "Train-L一次性训练耗时未记录，且训练耗时不计入单元推理时间。"
            ),
            "PPO与GA时间均为并发条件下的求解段观测值，不能据此作隔离运行下纯算法速度的因果比较；完整批次、workers与整批墙钟见附录运行表及manifest.evaluation_runs。",
        ),
    )
    return frame, markdown


def _table5_scale_design(protocol: Mapping[str, Any]) -> tuple[pd.DataFrame, str]:
    """Build the training/test scale design directly from the locked protocol."""

    scale_fields = protocol.get("scale_fields")
    if not isinstance(scale_fields, list) or tuple(scale_fields) != SCALE_FIELDS:
        raise PaperTableError(
            f"protocol scale_fields must equal {list(SCALE_FIELDS)}, found {scale_fields!r}"
        )
    scales = protocol.get("test_scales")
    models = protocol.get("models")
    if not isinstance(scales, Mapping) or not isinstance(models, list):
        raise PaperTableError("protocol has invalid models or test_scales for Table 5")

    rows: list[dict[str, Any]] = []
    trained_by_scale: dict[str, str] = {}
    for model in models:
        if not isinstance(model, Mapping):
            raise PaperTableError("protocol has an invalid model entry for Table 5")
        model_id = str(model.get("id"))
        source_scale = str(model.get("train_scale"))
        checkpoint = str(model.get("checkpoint"))
        scale_record = scales.get(source_scale)
        if not isinstance(scale_record, Mapping) or not isinstance(
            scale_record.get("scale"), Mapping
        ):
            raise PaperTableError(f"training model {model_id} refers to invalid scale {source_scale}")
        shape = scale_record["scale"]
        values = [_integer(shape.get(field), f"Table 5 {model_id}/{field}") for field in scale_fields]
        trained_by_scale[source_scale] = model_id
        rows.append(
            {
                "object_type": "training",
                "object_type_label": "训练规模",
                "design_id": model_id,
                "source_test_scale": source_scale,
                **dict(zip(scale_fields, values)),
                "scale_signature": "/".join(str(value) for value in values),
                "quantity": 1,
                "quantity_unit": "model",
                "quantity_display": "1份模型",
                "description": f"生成{checkpoint}",
            }
        )

    intermediate_index = 0
    for scale_id in _scales(protocol):
        scale_record = scales[scale_id]
        shape = scale_record["scale"]
        values = [_integer(shape.get(field), f"Table 5 {scale_id}/{field}") for field in scale_fields]
        quantity = _expected_instances(protocol, scale_id)
        matching_model = trained_by_scale.get(scale_id)
        if matching_model is None:
            intermediate_index += 1
            description = f"未见中间规模{intermediate_index}"
        else:
            description = f"与{matching_model}同规模"
        rows.append(
            {
                "object_type": "test",
                "object_type_label": "测试规模",
                "design_id": scale_id,
                "source_test_scale": scale_id,
                **dict(zip(scale_fields, values)),
                "scale_signature": "/".join(str(value) for value in values),
                "quantity": quantity,
                "quantity_unit": "instance",
                "quantity_display": f"{quantity}个实例",
                "description": description,
            }
        )

    frame = pd.DataFrame(rows)
    markdown = _markdown_table(
        title="表5  两种训练规模与四种测试规模设置",
        headers=(
            "类型",
            "编号",
            "产废节点/废物类型/处理处置节点/车辆/周期",
            "实例或模型数量",
            "说明",
        ),
        rows=(
            (
                row["object_type_label"],
                row["design_id"],
                row["scale_signature"],
                row["quantity_display"],
                row["description"],
            )
            for row in rows
        ),
        notes=(
            "规模向量依次为产废节点、废物类型、处理处置节点、车辆和周期。",
            "测试实例数量直接来自已验证且哈希锁定的protocol_config.json。",
            "四种测试规模共用同一锁定generator_profile，除规模字段外不改变生成参数。",
        ),
    )
    return frame, markdown


def _generalization_core(
    protocol: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], PrimaryStats]]:
    instance = tables["instance_level_results.csv"]
    paired = tables["paired_ppo_vs_ga.csv"]
    operational = tables["generalization_operational.csv"]
    source_matrix = tables["generalization_matrix.csv"]
    preference_ids = [str(value["id"]) for value in _preferences(protocol)]
    models = [str(value["id"]) for value in protocol.get("models", [])]
    scales = _scales(protocol)
    if set(models) != {"Train-S", "Train-L"}:
        raise PaperTableError(f"expected Train-S and Train-L models, found {models}")
    rows: list[dict[str, Any]] = []
    macro_stats: dict[tuple[str, str], PrimaryStats] = {}
    for model_id in models:
        for scale in scales:
            expected_n = _expected_instances(protocol, scale)
            ppo_group = instance[
                (instance["method"] == "ppo")
                & (instance["model_id"] == model_id)
                & (instance["test_scale"] == scale)
            ]
            expected_cells = expected_n * len(preference_ids)
            quality_cells = int(
                pd.to_numeric(ppo_group["weighted_objective_normalized"], errors="coerce")
                .notna()
                .sum()
            )
            macro = _primary_instance_macro(
                ppo_group,
                value_column="weighted_objective_normalized",
                expected_preferences=preference_ids,
                expected_instances=expected_n,
            )
            feasible = _primary_instance_macro(
                ppo_group,
                value_column="feasible_rate",
                expected_preferences=preference_ids,
                expected_instances=expected_n,
            )
            terminal = _primary_instance_macro(
                ppo_group,
                value_column="terminal_clear_rate",
                expected_preferences=preference_ids,
                expected_instances=expected_n,
            )
            runtime = _primary_instance_macro(
                ppo_group,
                value_column="runtime_seconds",
                expected_preferences=preference_ids,
                expected_instances=expected_n,
            )
            delta = _paired_macro_stats(
                paired,
                model_id=model_id,
                scale=scale,
                preference_ids=preference_ids,
                expected_instances=expected_n,
            )
            outcomes = _outcome_counts(ppo_group)
            source = _only_row(
                operational,
                "generalization_operational.csv",
                model_id=model_id,
                test_scale=scale,
            )
            _assert_close(f"{model_id}/{scale} macro J", macro.mean, source["macro_j"])
            _assert_close(
                f"{model_id}/{scale} feasible rate", feasible.mean, source["strict_feasible_rate"]
            )
            _assert_close(
                f"{model_id}/{scale} terminal-clear rate",
                terminal.mean,
                source["terminal_clear_rate"],
            )
            _assert_close(
                f"{model_id}/{scale} inference time",
                runtime.mean,
                source["mean_inference_seconds"],
            )
            _assert_close(
                f"{model_id}/{scale} relative GA improvement",
                delta.mean,
                source["relative_ga_improvement_percent"],
            )
            source_matrix_row = _only_row(
                source_matrix, "generalization_matrix.csv", model_id=model_id
            )
            _assert_close(
                f"{model_id}/{scale} matrix J",
                macro.mean,
                source_matrix_row[f"macro_j_{scale}"],
            )
            macro_stats[(model_id, scale)] = macro
            rows.append(
                {
                    "model_id": model_id,
                    "test_scale": scale,
                    "expected_instances": expected_n,
                    "valid_macro_j_instances": macro.valid_n,
                    "expected_instance_preference_cells": expected_cells,
                    "quality_cells": quality_cells,
                    "quality_missing_cells": expected_cells - quality_cells,
                    "incomplete_macro_instances": expected_n - macro.valid_n,
                    "macro_j_mean": macro.mean,
                    "macro_j_sd_by_instance": macro.sd,
                    "strict_feasible_rate_percent": (
                        None if feasible.mean is None else 100.0 * feasible.mean
                    ),
                    "strict_feasible_rate_sd_percent": (
                        None if feasible.sd is None else 100.0 * feasible.sd
                    ),
                    "terminal_clear_rate_percent": (
                        None if terminal.mean is None else 100.0 * terminal.mean
                    ),
                    "terminal_clear_rate_sd_percent": (
                        None if terminal.sd is None else 100.0 * terminal.sd
                    ),
                    "mean_inference_seconds": runtime.mean,
                    "sd_inference_seconds_by_instance": runtime.sd,
                    "relative_ga_improvement_percent": delta.mean,
                    "relative_ga_improvement_sd_percent_by_instance": delta.sd,
                    "relative_ga_valid_instances": delta.valid_n,
                    "expected_paired_instance_preference_cells": expected_cells,
                    "paired_instance_preference_cells": int(source["paired_ga_n"]),
                    "two_model_relative_gap_percent": _finite(
                        source["two_model_relative_gap_percent"]
                    ),
                    **outcomes,
                }
            )
    by_key = {(row["model_id"], row["test_scale"]): row for row in rows}
    for scale in scales:
        small = macro_stats[("Train-S", scale)].mean
        large = macro_stats[("Train-L", scale)].mean
        if small is None or large is None:
            expected_gap = {"Train-S": None, "Train-L": None}
        else:
            best = min(small, large)
            if abs(best) <= 1e-12:
                expected_gap = {"Train-S": None, "Train-L": None}
            else:
                expected_gap = {
                    "Train-S": (small - best) / abs(best) * 100.0,
                    "Train-L": (large - best) / abs(best) * 100.0,
                }
        for model_id in ("Train-S", "Train-L"):
            row = by_key[(model_id, scale)]
            _assert_close(
                f"{model_id}/{scale} two-model relative gap",
                row["two_model_relative_gap_percent"],
                expected_gap[model_id],
            )
            matrix_row = _only_row(source_matrix, "generalization_matrix.csv", model_id=model_id)
            _assert_close(
                f"{model_id}/{scale} matrix two-model relative gap",
                row["two_model_relative_gap_percent"],
                matrix_row[f"two_model_relative_gap_percent_{scale}"],
            )
    return rows, macro_stats


def _table6(
    protocol: Mapping[str, Any], core_rows: Sequence[Mapping[str, Any]]
) -> tuple[pd.DataFrame, str]:
    scales = _scales(protocol)
    models = [str(value["id"]) for value in protocol["models"]]
    by_key = {(row["model_id"], row["test_scale"]): row for row in core_rows}
    wide_rows: list[dict[str, Any]] = []
    markdown_rows: list[tuple[Any, ...]] = []
    for model_id in models:
        output: dict[str, Any] = {"model_id": model_id}
        display: list[Any] = [model_id]
        for scale in scales:
            source = by_key[(model_id, scale)]
            for field in (
                "macro_j_mean",
                "macro_j_sd_by_instance",
                "valid_macro_j_instances",
                "expected_instances",
                "quality_cells",
                "expected_instance_preference_cells",
                "quality_missing_cells",
                "two_model_relative_gap_percent",
            ):
                output[f"{field}_{scale}"] = source[field]
            j_text = _mean_sd(
                source["macro_j_mean"], source["macro_j_sd_by_instance"], 4
            )
            gap_text = _percent(source["two_model_relative_gap_percent"], 2)
            cell = (
                f"{j_text} [{gap_text}]; "
                f"n={source['valid_macro_j_instances']}/{source['expected_instances']}, "
                f"缺失质量单元={source['quality_missing_cells']}"
            )
            output[f"display_{scale}"] = cell
            display.append(cell)
        wide_rows.append(output)
        markdown_rows.append(tuple(display))
    frame = pd.DataFrame(wide_rows)
    markdown = _markdown_table(
        title="表6  两种训练规模下PPO-Transformer的任务规模泛化结果",
        headers=("训练模型", *scales),
        rows=markdown_rows,
        notes=(
            "每格为宏平均J±样本SD [两模型相对差距L]；先在实例内等权平均全部偏好，再以独立实例为统计单位。",
            "L仅比较同一测试规模下两份固定模型，不是严格的同规模反事实泛化损失。",
            "n为完整宏平均实例数/计划实例数；任何缺失或不可行质量单元均使对应实例的主宏平均为空，不用conditional结果替代。",
        ),
    )
    return frame, markdown


def _table7(
    manifest: Mapping[str, Any], core_rows: Sequence[Mapping[str, Any]]
) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    for source in core_rows:
        row = dict(source)
        row["strict_feasible_rate_mean_sd"] = _mean_sd(
            source["strict_feasible_rate_percent"],
            source["strict_feasible_rate_sd_percent"],
            2,
        )
        row["terminal_clear_rate_mean_sd"] = _mean_sd(
            source["terminal_clear_rate_percent"],
            source["terminal_clear_rate_sd_percent"],
            2,
        )
        row["inference_seconds_mean_sd"] = _mean_sd(
            source["mean_inference_seconds"],
            source["sd_inference_seconds_by_instance"],
            4,
        )
        row["relative_ga_improvement_mean_sd"] = _mean_sd(
            source["relative_ga_improvement_percent"],
            source["relative_ga_improvement_sd_percent_by_instance"],
            2,
        )
        row["training_seconds_once"] = _training_seconds(manifest, str(source["model_id"]))
        row["runtime_concurrency_scope"] = _concurrency_scope(manifest, "ppo")
        row["outcome_disclosure"] = (
            f"技术={source['technical_failure_instances']}; "
            f"不可行={source['infeasible_instances']}; "
            f"缺失={source['missing_instances']}; "
            f"宏不完整={source['incomplete_macro_instances']}"
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    markdown_rows = [
        (
            row["model_id"],
            row["test_scale"],
            row["expected_instances"],
            _mean_sd_percent(
                row["strict_feasible_rate_percent"],
                row["strict_feasible_rate_sd_percent"],
                2,
            ),
            _mean_sd_percent(
                row["terminal_clear_rate_percent"],
                row["terminal_clear_rate_sd_percent"],
                2,
            ),
            row["inference_seconds_mean_sd"],
            _mean_sd_percent(
                row["relative_ga_improvement_percent"],
                row["relative_ga_improvement_sd_percent_by_instance"],
                2,
            ),
            f"{row['quality_cells']}/{row['expected_instance_preference_cells']}",
            row["outcome_disclosure"],
        )
        for row in rows
    ]
    train_s = _training_seconds(manifest, "Train-S")
    train_l = _training_seconds(manifest, "Train-L")
    training_note = (
        f"一次性训练耗时：Train-S={train_s:.3f}s，Train-L={train_l:.3f}s。"
        if train_s is not None and train_l is not None
        else "至少一份模型的一次性训练耗时未记录。"
    )
    markdown = _markdown_table(
        title="表7  泛化实验的可行性与计算时间",
        headers=(
            "训练模型",
            "测试规模",
            "实例数",
            "严格可行率",
            "期末清零率",
            "推理时间(s)",
            "相对GA改善率",
            "质量单元n/计划n",
            "异常实例计数",
        ),
        rows=markdown_rows,
        notes=(
            "比率、推理时间与相对GA改善率均为独立实例层面的均值±样本SD，每个实例先等权平均全部偏好；异常计数依次为技术失败、纯求解不可行、缺失、宏平均不完整。",
            "正的相对GA改善率表示PPO的归一化目标较低。",
            training_note + " 训练耗时不计入推理时间。",
            "PPO时间为并发条件下的求解段观测值，不能据此作隔离运行下纯算法速度的因果比较；完整批次、workers与整批墙钟见附录运行表及manifest.evaluation_runs。",
        ),
    )
    return frame, markdown


def _status_counts_text(frame: pd.DataFrame) -> str:
    counts = Counter(frame["solver_status"].fillna("missing_status").astype(str))
    return "; ".join(f"{status}:n={counts[status]}" for status in sorted(counts))


def _main_method_selections() -> tuple[tuple[str, str, str, str, str], ...]:
    return (
        ("E1", "PPO-Transformer (Train-S)", "ppo", "Train-S", "Test-1"),
        ("E1", "GA", "ga", "GA", "Test-1"),
        ("E1", "启发式", "heuristic", "Heuristic", "Test-1"),
        ("E1", "MILP", "milp", "MILP", "Test-1"),
        ("E2", "PPO-Transformer (Train-L)", "ppo", "Train-L", "Test-4"),
        ("E2", "GA", "ga", "GA", "Test-4"),
        ("E2", "启发式", "heuristic", "Heuristic", "Test-4"),
    )


def _all_methods_appendix(
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    summary = tables["summary_by_preference.csv"]
    instance = tables["instance_level_results.csv"]
    gaps = tables["milp_gap_summary.csv"]
    rows: list[dict[str, Any]] = []
    for experiment, display_method, method, model_id, scale in _main_method_selections():
        expected_n = _expected_instances(protocol, scale)
        for preference in _preferences(protocol):
            preference_id = str(preference["id"])
            source = _summary_row(
                summary,
                method=method,
                model_id=model_id,
                scale=scale,
                preference_id=preference_id,
            )
            group = instance[
                (instance["method"] == method)
                & (instance["model_id"] == model_id)
                & (instance["test_scale"] == scale)
                & (instance["preference_id"] == preference_id)
            ]
            if len(group) != expected_n:
                raise PaperTableError(
                    f"{experiment}/{method}/{preference_id}: expected {expected_n} rows, "
                    f"found {len(group)}"
                )
            outcomes = _outcome_counts(group)
            terminal = _single_preference_stats(
                instance,
                method=method,
                model_id=model_id,
                scale=scale,
                preference_id=preference_id,
                value_column="terminal_clear_rate",
                expected_instances=expected_n,
            )
            feasible = _single_preference_stats(
                instance,
                method=method,
                model_id=model_id,
                scale=scale,
                preference_id=preference_id,
                value_column="feasible_rate",
                expected_instances=expected_n,
            )
            gap_mean = gap_sd = None
            gap_n = 0
            if experiment == "E1" and method in {"ppo", "ga", "heuristic"}:
                gap_row = _only_row(
                    gaps,
                    "milp_gap_summary.csv",
                    method=method,
                    model_id=model_id,
                    preference_id=preference_id,
                )
                gap_mean = _finite(gap_row["mean_gap_percent"])
                gap_sd = _finite(gap_row["sample_sd_gap_percent"])
                gap_n = int(gap_row["valid_n"])
            runtime_mean = _finite(source["runtime_seconds_mean"])
            runtime_sd = _finite(source["runtime_seconds_sd"])
            is_heuristic = method == "heuristic"
            row = {
                "experiment": experiment,
                "method": display_method,
                "method_id": method,
                "model_id": model_id,
                "test_scale": scale,
                "preference_id": preference_id,
                "preference": _preference_label(preference),
                "cost_mean": _finite(source["cost_mean"]),
                "cost_sd": _finite(source["cost_sd"]),
                "cost_mean_sd": _mean_sd(source["cost_mean"], source["cost_sd"], 2),
                "risk_mean": _finite(source["risk_mean"]),
                "risk_sd": _finite(source["risk_sd"]),
                "risk_mean_sd": _mean_sd(source["risk_mean"], source["risk_sd"], 3),
                "j_mean": _finite(source["weighted_objective_normalized_mean"]),
                "j_sd": _finite(source["weighted_objective_normalized_sd"]),
                "j_mean_sd": _mean_sd(
                    source["weighted_objective_normalized_mean"],
                    source["weighted_objective_normalized_sd"],
                    4,
                ),
                "strict_feasible_percent": _rate_percent(source["strict_feasible_rate"]),
                "strict_feasible_sd_percent": (
                    None if feasible.sd is None else 100.0 * feasible.sd
                ),
                "terminal_clear_percent": _rate_percent(source["terminal_clear_rate"]),
                "terminal_clear_sd_percent": (
                    None if terminal.sd is None else 100.0 * terminal.sd
                ),
                "time_measurement": (
                    "测试集锁定阶段初始解构造时间(initialization_seconds)"
                    if is_heuristic
                    else "并发条件下求解段观测时间(runtime_seconds)"
                ),
                "solve_segment_seconds_mean": None if is_heuristic else runtime_mean,
                "solve_segment_seconds_sd": None if is_heuristic else runtime_sd,
                "solve_segment_seconds_mean_sd": (
                    None if is_heuristic else _mean_sd(runtime_mean, runtime_sd, 4)
                ),
                "initialization_seconds_mean": runtime_mean if is_heuristic else None,
                "initialization_seconds_sd": runtime_sd if is_heuristic else None,
                "initialization_seconds_mean_sd": (
                    _mean_sd(runtime_mean, runtime_sd, 4) if is_heuristic else None
                ),
                "quality_complete_n": int(source["n_instances_quality"]),
                "expected_n": expected_n,
                "gap_percent_mean": gap_mean,
                "gap_percent_sd": gap_sd,
                "gap_valid_n": gap_n,
                "gap_mean_sd": _mean_sd(gap_mean, gap_sd, 2),
                "solver_status_scope": _status_counts_text(group) if method == "milp" else "",
                "solve_segment_concurrency_scope": (
                    "" if is_heuristic else _concurrency_scope(manifest, method)
                ),
                "outcome_disclosure": _outcome_text(outcomes),
                **outcomes,
            }
            rows.append(row)
    frame = pd.DataFrame(rows)
    markdown_rows = [
        (
            row["experiment"],
            row["method"],
            row["preference"],
            f"{row['cost_mean_sd']} (n={row['quality_complete_n']})",
            f"{row['risk_mean_sd']} (n={row['quality_complete_n']})",
            f"{row['j_mean_sd']} (n={row['quality_complete_n']})",
            _mean_sd_percent(
                row["strict_feasible_percent"], row["strict_feasible_sd_percent"], 2
            ),
            _mean_sd_percent(
                row["terminal_clear_percent"], row["terminal_clear_sd_percent"], 2
            ),
            (
                row["initialization_seconds_mean_sd"]
                if row["method_id"] == "heuristic"
                else row["solve_segment_seconds_mean_sd"]
            ),
            row["outcome_disclosure"],
            (
                f"{row['gap_mean_sd']} (n={row['gap_valid_n']})"
                if row["experiment"] == "E1" and row["method_id"] != "milp"
                else "—"
            ),
            row["solver_status_scope"] or "—",
        )
        for row in rows
    ]
    markdown = _markdown_table(
        title="附表  E1/E2全方法结果",
        headers=(
            "实验",
            "方法",
            "偏好",
            "成本",
            "风险",
            "J",
            "严格可行率",
            "期末清零率",
            "时间(s；PPO/GA/MILP求解段；启发式初始化)",
            "技术/不可行/缺失/质量不完整实例",
            "Gap%",
            "MILP状态范围",
        ),
        rows=markdown_rows,
        notes=(
            "成本、风险、J与所列时间为独立实例均值±样本SD；完整质量n单列在前三项括号中。",
            "异常计数中的不可行仅指无技术失败、无缺失但至少一个重启未通过严格可行性检验的实例。",
            "Test-1的Gap只按MILP证明optimal的逐实例基准计算；MILP自身及Test-4不报告Gap。",
            "PPO/GA/MILP时间为并发条件下求解段观测时间(runtime_seconds)，不含模型训练，并按各方法manifest批次的并发披露解释。",
            "启发式时间为测试集锁定阶段初始解构造时间(initialization_seconds)，不属于evaluation求解段且不绑定启发式evaluation并发批次。",
        ),
    )
    return frame, markdown


def _milp_status_appendix(
    protocol: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, str]:
    instance = tables["instance_level_results.csv"]
    rows: list[dict[str, Any]] = []
    for preference in _preferences(protocol):
        preference_id = str(preference["id"])
        group = instance[
            (instance["method"] == "milp")
            & (instance["model_id"] == "MILP")
            & (instance["test_scale"] == "Test-1")
            & (instance["preference_id"] == preference_id)
        ]
        expected_n = _expected_instances(protocol, "Test-1")
        if len(group) != expected_n:
            raise PaperTableError(
                f"MILP status appendix expected {expected_n} rows for {preference_id}, "
                f"found {len(group)}"
            )
        status_counts = Counter(group["solver_status"].fillna("missing_status").astype(str))
        for status in sorted(status_counts, key=lambda item: (item.casefold() != "optimal", item)):
            status_group = group[group["solver_status"].fillna("missing_status").astype(str) == status]
            incumbent = pd.to_numeric(
                status_group["weighted_objective_normalized"], errors="coerce"
            )
            complete = len(incumbent) > 0 and incumbent.notna().all()
            mean = float(incumbent.mean()) if complete else None
            sd = float(incumbent.std(ddof=1)) if complete and len(incumbent) >= 2 else None
            eligible = status.casefold() == "optimal"
            rows.append(
                {
                    "preference_id": preference_id,
                    "preference": _preference_label(preference),
                    "solver_status": status,
                    "status_n": int(len(status_group)),
                    "preference_total_n": expected_n,
                    "incumbent_j_mean": mean,
                    "incumbent_j_sd": sd,
                    "incumbent_j_valid_n": int(incumbent.notna().sum()),
                    "incumbent_j_mean_sd": _mean_sd(mean, sd, 4),
                    "eligible_as_j_star": eligible,
                    "reference_rule": (
                        "可作为逐实例J*基准（仍按实例配对，不使用本行均值替代）"
                        if eligible
                        else "仅为非optimal incumbent（若存在）；不得作为J*或Gap分母"
                    ),
                    "all_status_counts": "; ".join(
                        f"{key}:n={status_counts[key]}" for key in sorted(status_counts)
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    markdown_rows = [
        (
            row["preference"],
            row["solver_status"],
            row["status_n"],
            f"{row['incumbent_j_mean_sd']} (n={row['incumbent_j_valid_n']})",
            "是" if row["eligible_as_j_star"] else "否",
            row["reference_rule"],
        )
        for row in rows
    ]
    markdown = _markdown_table(
        title="附表  MILP状态与限时基准",
        headers=("偏好", "solver status", "状态n", "incumbent J", "可作J*", "基准规则"),
        rows=markdown_rows,
        notes=(
            "incumbent J按solver status分层报告均值±样本SD，但Gap必须使用同一实例上status=optimal的J*逐一配对。",
            "任何非optimal状态（包括限时或其他中止状态）即使给出可行incumbent，也不得称为全局最优或用于Gap分母。",
        ),
    )
    return frame, markdown


def _risk_appendix(
    protocol: Mapping[str, Any], tables: Mapping[str, pd.DataFrame]
) -> tuple[pd.DataFrame, str]:
    summary = tables["summary_by_preference.csv"]
    instance = tables["instance_level_results.csv"]
    selections = _main_method_selections()
    rows: list[dict[str, Any]] = []
    for experiment, display_method, method, model_id, scale in selections:
        for preference in _preferences(protocol):
            preference_id = str(preference["id"])
            source = _summary_row(
                summary,
                method=method,
                model_id=model_id,
                scale=scale,
                preference_id=preference_id,
            )
            total_mean = _finite(source["risk_mean"])
            total_sd = _finite(source["risk_sd"])
            milp_group = instance[
                (instance["method"] == "milp")
                & (instance["model_id"] == "MILP")
                & (instance["test_scale"] == "Test-1")
                & (instance["preference_id"] == preference_id)
            ]
            output: dict[str, Any] = {
                "experiment": experiment,
                "method": display_method,
                "method_id": method,
                "model_id": model_id,
                "test_scale": scale,
                "preference_id": preference_id,
                "preference": _preference_label(preference),
                "risk_mean": total_mean,
                "risk_sd": total_sd,
                "risk_n": int(source["n_instances_quality"]),
                "risk_mean_sd": _mean_sd(total_mean, total_sd, 3),
                "solver_status_scope": (
                    _status_counts_text(milp_group) if method == "milp" else ""
                ),
            }
            component_sum = 0.0
            all_components = total_mean is not None
            for component, field, _ in RISK_COMPONENTS:
                mean = _finite(source[f"{field}_mean"])
                sd = _finite(source[f"{field}_sd"])
                if mean is None:
                    all_components = False
                else:
                    component_sum += mean
                share = (
                    mean / total_mean * 100.0
                    if mean is not None and total_mean is not None and abs(total_mean) > 1e-12
                    else None
                )
                output[f"{component}_risk_mean"] = mean
                output[f"{component}_risk_sd"] = sd
                output[f"{component}_risk_pct_share"] = share
                output[f"{component}_risk_mean_sd_share"] = (
                    f"{_mean_sd(mean, sd, 3)} ({_percent(share, 1)})"
                )
            if all_components and total_mean is not None:
                _assert_close(
                    f"risk decomposition {method}/{scale}/{preference_id}",
                    component_sum,
                    total_mean,
                    tolerance=1e-7,
                )
            rows.append(output)
    frame = pd.DataFrame(rows)
    markdown_rows = [
        (
            row["experiment"],
            row["method"],
            row["preference"],
            row["risk_mean_sd"],
            row["transport_risk_mean_sd_share"],
            row["coload_risk_mean_sd_share"],
            row["producer_inventory_risk_mean_sd_share"],
            row["facility_inventory_risk_mean_sd_share"],
            row["risk_n"],
            row["solver_status_scope"] or "—",
        )
        for row in rows
    ]
    markdown = _markdown_table(
        title="附表  风险构成",
        headers=(
            "实验",
            "方法",
            "偏好 (wC,wR)",
            "总风险",
            "运输风险（占比）",
            "共载风险（占比）",
            "产废端库存风险（占比）",
            "处理处置端库存风险（占比）",
            "有效n",
            "MILP状态范围",
        ),
        rows=markdown_rows,
        notes=(
            "风险及各分量为独立实例的均值±样本SD；括号内为分量均值占总风险均值的比例。",
            "只展示E1/E2主比较：Test-1使用Train-S，Test-4使用Train-L；未使用任何conditional字段。",
            "主质量统计缺失时相应风险与构成保持空白。",
            "MILP风险行披露全部solver status；非optimal行对应的incumbent不得解释为J*。",
        ),
    )
    return frame, markdown


def _registry(
    manifest: Mapping[str, Any],
    table3: pd.DataFrame,
    table4: pd.DataFrame,
    table6: pd.DataFrame,
    table7: pd.DataFrame,
    risk: pd.DataFrame,
    all_methods: pd.DataFrame,
    milp_status: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        hypothesis_id: str,
        metric_name: str,
        value: Any,
        metric_se: Any,
        component: str,
        pct_share: Any,
        source_table: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "metric_name": metric_name,
                "metric_value": _finite(value),
                "metric_se": _finite(metric_se),
                "component": component,
                "pct_share": _finite(pct_share),
                "source_table": source_table,
                "source_figure": "",
                "script": SCRIPT_ID,
                "notes": notes,
            }
        )

    concurrency_ppo = _concurrency_scope(manifest, "ppo")
    concurrency_ga = _concurrency_scope(manifest, "ga")
    solve_segment_note = "并发条件下求解段观测时间(runtime_seconds)"
    for _, row in table3.iterrows():
        pref = str(row["preference_id"])
        base_note = f"preference={pref}; independent-instance unit; primary fields only"
        add("E1", f"ppo_j_mean.{pref}", row["ppo_j_mean"], _se(row["ppo_j_sd"], row["ppo_j_n"]), "PPO-Transformer", None, "table3_small_scale.csv", base_note)
        add("E1", f"ga_j_mean.{pref}", row["ga_j_mean"], _se(row["ga_j_sd"], row["ga_j_n"]), "GA", None, "table3_small_scale.csv", base_note)
        add("E1", f"ppo_gap_percent.{pref}", row["ppo_gap_mean_percent"], _se(row["ppo_gap_sd_percent"], row["ppo_gap_valid_n"]), "PPO-Transformer", None, "table3_small_scale.csv", base_note + "; MILP-optimal references only")
        add("E1", f"ga_gap_percent.{pref}", row["ga_gap_mean_percent"], _se(row["ga_gap_sd_percent"], row["ga_gap_valid_n"]), "GA", None, "table3_small_scale.csv", base_note + "; MILP-optimal references only")
        add("E1", f"delta_j_percent.{pref}", row["delta_j_mean_percent"], _se(row["delta_j_sd_percent"], row["delta_j_valid_n"]), "PPO vs GA", None, "table3_small_scale.csv", base_note + "; positive means lower PPO J")
        add("E1", f"ppo_feasible_percent.{pref}", row["ppo_feasible_percent"], _se(row["ppo_feasible_sd_percent"], row["ppo_feasible_n"]), "PPO-Transformer", None, "table3_small_scale.csv", base_note)
        add("E1", f"ga_feasible_percent.{pref}", row["ga_feasible_percent"], _se(row["ga_feasible_sd_percent"], row["ga_feasible_n"]), "GA", None, "table3_small_scale.csv", base_note)
        add("E1", f"ppo_runtime_seconds.{pref}", row["ppo_runtime_mean_seconds"], _se(row["ppo_runtime_sd_seconds"], row["ppo_runtime_n"]), "PPO-Transformer", None, "table3_small_scale.csv", base_note + "; " + solve_segment_note + "; " + concurrency_ppo)
        add("E1", f"ga_runtime_seconds.{pref}", row["ga_runtime_mean_seconds"], _se(row["ga_runtime_sd_seconds"], row["ga_runtime_n"]), "GA", None, "table3_small_scale.csv", base_note + "; " + solve_segment_note + "; " + concurrency_ga)
        add("E1", f"milp_runtime_seconds.{pref}", row["milp_runtime_mean_seconds"], _se(row["milp_runtime_sd_seconds"], row["milp_runtime_n"]), "MILP", None, "table3_small_scale.csv", base_note + "; " + solve_segment_note + "; " + _concurrency_scope(manifest, "milp"))
        add("E1", f"milp_optimal_percent.{pref}", row["milp_optimal_percent"], None, "MILP", None, "table3_small_scale.csv", base_note + f"; optimal_n={int(row['milp_optimal_n'])}/{int(row['milp_total_n'])}")
        for algorithm in ("ppo", "ga"):
            for outcome in (
                "technical_failure_instances",
                "infeasible_instances",
                "missing_instances",
                "incomplete_quality_instances",
            ):
                add(
                    "E1",
                    f"{algorithm}_{outcome}.{pref}",
                    row[f"{algorithm}_{outcome}"],
                    None,
                    "PPO-Transformer" if algorithm == "ppo" else "GA",
                    None,
                    "table3_small_scale.csv",
                    base_note + "; diagnostic instance count",
                )
    for _, row in table4.iterrows():
        pref = str(row["preference_id"])
        base_note = f"preference={pref}; independent-instance unit; primary fields only"
        for metric, label, component in (
            ("ppo_cost", "cost", "PPO-Transformer"),
            ("ppo_risk", "risk", "PPO-Transformer"),
            ("ppo_j", "weighted objective J", "PPO-Transformer"),
            ("ga_j", "weighted objective J", "GA"),
        ):
            add("E2", f"{metric}_mean.{pref}", row[f"{metric}_mean"], _se(row[f"{metric}_sd"], row[f"{metric}_n"]), component, None, "table4_large_scale.csv", base_note + f"; metric={label}")
        add("E2", f"delta_j_percent.{pref}", row["delta_j_mean_percent"], _se(row["delta_j_sd_percent"], row["delta_j_valid_n"]), "PPO vs GA", None, "table4_large_scale.csv", base_note + "; positive means lower PPO J")
        add("E2", f"ppo_feasible_percent.{pref}", row["ppo_feasible_percent"], _se(row["ppo_feasible_sd_percent"], row["ppo_feasible_n"]), "PPO-Transformer", None, "table4_large_scale.csv", base_note)
        add("E2", f"ga_feasible_percent.{pref}", row["ga_feasible_percent"], _se(row["ga_feasible_sd_percent"], row["ga_feasible_n"]), "GA", None, "table4_large_scale.csv", base_note)
        add("E2", f"ppo_runtime_seconds.{pref}", row["ppo_runtime_mean_seconds"], _se(row["ppo_runtime_sd_seconds"], row["ppo_runtime_n"]), "PPO-Transformer", None, "table4_large_scale.csv", base_note + "; " + solve_segment_note + "; " + concurrency_ppo)
        add("E2", f"ga_runtime_seconds.{pref}", row["ga_runtime_mean_seconds"], _se(row["ga_runtime_sd_seconds"], row["ga_runtime_n"]), "GA", None, "table4_large_scale.csv", base_note + "; " + solve_segment_note + "; " + concurrency_ga)
        for algorithm in ("ppo", "ga"):
            for outcome in (
                "technical_failure_instances",
                "infeasible_instances",
                "missing_instances",
                "incomplete_quality_instances",
            ):
                add(
                    "E2",
                    f"{algorithm}_{outcome}.{pref}",
                    row[f"{algorithm}_{outcome}"],
                    None,
                    "PPO-Transformer" if algorithm == "ppo" else "GA",
                    None,
                    "table4_large_scale.csv",
                    base_note + "; diagnostic instance count",
                )
    for _, row in table7.iterrows():
        model = str(row["model_id"])
        scale = str(row["test_scale"])
        key = f"{model}.{scale}"
        hypothesis_id = f"G-{model}"
        n = int(row["expected_instances"])
        note = "preferences averaged within each independent instance; primary fields only"
        add(hypothesis_id, f"macro_j.{key}", row["macro_j_mean"], _se(row["macro_j_sd_by_instance"], row["valid_macro_j_instances"]), model, None, "table6_generalization_matrix.csv", note)
        add(hypothesis_id, f"two_model_relative_gap_percent.{key}", row["two_model_relative_gap_percent"], None, model, None, "table6_generalization_matrix.csv", "descriptive comparison of two fixed models; not strict generalization loss")
        add(hypothesis_id, f"strict_feasible_percent.{key}", row["strict_feasible_rate_percent"], _se(row["strict_feasible_rate_sd_percent"], n), model, None, "table7_generalization_operational.csv", note)
        add(hypothesis_id, f"terminal_clear_percent.{key}", row["terminal_clear_rate_percent"], _se(row["terminal_clear_rate_sd_percent"], n), model, None, "table7_generalization_operational.csv", note)
        add(hypothesis_id, f"inference_seconds.{key}", row["mean_inference_seconds"], _se(row["sd_inference_seconds_by_instance"], n), model, None, "table7_generalization_operational.csv", note + "; " + solve_segment_note + "; " + concurrency_ppo)
        add(hypothesis_id, f"relative_ga_improvement_percent.{key}", row["relative_ga_improvement_percent"], _se(row["relative_ga_improvement_sd_percent_by_instance"], row["relative_ga_valid_instances"]), model, None, "table7_generalization_operational.csv", note + "; positive means lower PPO J")
        for outcome in (
            "technical_failure_instances",
            "infeasible_instances",
            "missing_instances",
            "incomplete_macro_instances",
        ):
            add(
                hypothesis_id,
                f"{outcome}.{key}",
                row[outcome],
                None,
                model,
                None,
                "table7_generalization_operational.csv",
                note + "; diagnostic instance count",
            )
    for model, source in (("Train-S", "table3_small_scale.csv"), ("Train-L", "table4_large_scale.csv")):
        add(f"G-{model}", f"one_time_training_seconds.{model}", _training_seconds(manifest, model), None, model, None, source, "one training run; excluded from per-instance inference time")
    for _, row in all_methods[
        all_methods["method_id"].isin(["heuristic", "milp"])
    ].iterrows():
        hypothesis_id = str(row["experiment"])
        identity = f"{row['model_id']}.{row['preference_id']}"
        n = int(row["quality_complete_n"])
        for metric, label in (("cost", "cost"), ("risk", "risk"), ("j", "weighted objective J")):
            add(
                hypothesis_id,
                f"{metric}_mean.{identity}",
                row[f"{metric}_mean"],
                _se(row[f"{metric}_sd"], n),
                str(row["method"]),
                None,
                "appendix_all_methods.csv",
                f"metric={label}; independent-instance unit; primary fields only",
            )
        add(
            hypothesis_id,
            f"strict_feasible_percent.{identity}",
            row["strict_feasible_percent"],
            _se(row["strict_feasible_sd_percent"], row["expected_n"]),
            str(row["method"]),
            None,
            "appendix_all_methods.csv",
            "independent-instance unit; primary fields only",
        )
        if row["method_id"] == "heuristic":
            add(
                hypothesis_id,
                f"initialization_seconds.{identity}",
                row["initialization_seconds_mean"],
                _se(row["initialization_seconds_sd"], row["expected_n"]),
                "启发式",
                None,
                "appendix_all_methods.csv",
                "测试集锁定阶段初始解构造时间(initialization_seconds)；"
                "不属于evaluation求解段且不绑定启发式evaluation并发批次",
            )
        if row["method_id"] == "heuristic" and row["experiment"] == "E1":
            add(
                "E1",
                f"heuristic_gap_percent.{row['preference_id']}",
                row["gap_percent_mean"],
                _se(row["gap_percent_sd"], row["gap_valid_n"]),
                "启发式",
                None,
                "appendix_all_methods.csv",
                "MILP-optimal references only",
            )
    for _, row in milp_status.iterrows():
        add(
            "E1",
            f"milp_status_count.{row['preference_id']}.{row['solver_status']}",
            row["status_n"],
            None,
            "MILP",
            None,
            "appendix_milp_status.csv",
            str(row["reference_rule"]),
        )
    for _, row in risk.iterrows():
        n = int(row["risk_n"])
        identity = f"{row['experiment']}.{row['model_id']}.{row['preference_id']}"
        add(str(row["experiment"]), f"total_risk.{identity}", row["risk_mean"], _se(row["risk_sd"], n), "total", None, "appendix_risk_components.csv", "independent-instance unit; primary fields only")
        for component, _, _ in RISK_COMPONENTS:
            add(str(row["experiment"]), f"{component}_risk.{identity}", row[f"{component}_risk_mean"], _se(row[f"{component}_risk_sd"], n), component, row[f"{component}_risk_pct_share"], "appendix_risk_components.csv", "component mean; pct_share is component mean / total-risk mean")
    return pd.DataFrame(
        rows,
        columns=(
            "hypothesis_id",
            "metric_name",
            "metric_value",
            "metric_se",
            "component",
            "pct_share",
            "source_table",
            "source_figure",
            "script",
            "notes",
        ),
    )


def _adjudication(
    protocol: Mapping[str, Any],
    tables: Mapping[str, pd.DataFrame],
    table3: pd.DataFrame,
    table7: pd.DataFrame,
) -> pd.DataFrame:
    paired = tables["paired_ppo_vs_ga.csv"]
    preference_ids = [str(value["id"]) for value in _preferences(protocol)]

    def delta_direction(value: float | None) -> str:
        if value is None:
            return "不可判定"
        if value > 1e-12:
            return "PPO的J较低"
        if value < -1e-12:
            return "PPO的J较高"
        return "报告精度下持平"

    rows: list[dict[str, str]] = []
    for hypothesis_id, model, scale, source in (
        ("E1", "Train-S", "Test-1", "table3_small_scale.csv"),
        ("E2", "Train-L", "Test-4", "table4_large_scale.csv"),
    ):
        stats = _paired_macro_stats(
            paired,
            model_id=model,
            scale=scale,
            preference_ids=preference_ids,
            expected_instances=_expected_instances(protocol, scale),
        )
        magnitude = (
            f"平均ΔJ={stats.mean:.3f}%，SD={stats.sd:.3f}%，n={stats.valid_n}"
            if stats.mean is not None and stats.sd is not None
            else f"平均ΔJ={stats.mean:.3f}%，n={stats.valid_n}"
            if stats.mean is not None
            else f"主ΔJ不可用；完整实例n={stats.valid_n}/{stats.expected_n}"
        )
        interpretation = (
            "先在每个实例内等权平均全部偏好后进行描述性比较；ΔJ为正表示PPO目标较低，"
            "不据此作统计显著性判断。"
        )
        if hypothesis_id == "E1":
            optimal_n = int(table3["milp_optimal_n"].sum())
            total_n = int(table3["milp_total_n"].sum())
            interpretation += (
                f" MILP在{optimal_n}/{total_n}个实例—偏好基准上证明最优；"
                "Gap结论仅限于这些基准。"
            )
        rows.append(
            {
                "hypothesis_id": hypothesis_id,
                "direction": delta_direction(stats.mean),
                "magnitude": magnitude,
                "interpretation": interpretation,
                "source": source,
            }
        )

    for model in ("Train-S", "Train-L"):
        selected = table7[table7["model_id"] == model]
        improvements = pd.to_numeric(
            selected["relative_ga_improvement_percent"], errors="coerce"
        )
        feasibility = pd.to_numeric(
            selected["strict_feasible_rate_percent"], errors="coerce"
        )
        if len(selected) != len(_scales(protocol)) or improvements.isna().any() or feasibility.isna().any():
            direction = "不可判定"
            magnitude = "至少一个模型—测试规模主单元不可用"
        else:
            has_positive = bool((improvements > 1e-12).any())
            has_negative = bool((improvements < -1e-12).any())
            if has_positive and has_negative:
                direction = "不同测试规模方向不一致"
            elif has_positive:
                direction = "各测试规模的PPO目标较低或持平"
            elif has_negative:
                direction = "各测试规模的PPO目标较高或持平"
            else:
                direction = "四个测试规模均在报告精度下持平"
            magnitude = (
                f"各规模平均ΔJ范围={improvements.min():.3f}%至"
                f"{improvements.max():.3f}%；最低严格可行率={feasibility.min():.2f}%"
            )
        rows.append(
            {
                "hypothesis_id": f"G-{model}",
                "direction": direction,
                "magnitude": magnitude,
                "interpretation": (
                    "冻结模型的描述性证据仅覆盖Test-1至Test-4。Test-2/Test-3的两模型相对差距"
                    "不是严格泛化损失，结果不能外推为适用于任意规模。"
                ),
                "source": "table6_generalization_matrix.csv; table7_generalization_operational.csv",
            }
        )
    return pd.DataFrame(
        rows,
        columns=("hypothesis_id", "direction", "magnitude", "interpretation", "source"),
    )


def _validate_result_ids(registry: pd.DataFrame, adjudication: pd.DataFrame) -> None:
    registry_ids = set(registry["hypothesis_id"].dropna().astype(str))
    adjudication_ids = set(adjudication["hypothesis_id"].dropna().astype(str))
    if registry_ids != ALLOWED_HYPOTHESIS_IDS:
        raise PaperTableError(
            f"results registry IDs must be {sorted(ALLOWED_HYPOTHESIS_IDS)}, "
            f"found {sorted(registry_ids)}"
        )
    if adjudication_ids != ALLOWED_HYPOTHESIS_IDS:
        raise PaperTableError(
            f"adjudication IDs must be {sorted(ALLOWED_HYPOTHESIS_IDS)}, "
            f"found {sorted(adjudication_ids)}"
        )
    if adjudication["hypothesis_id"].duplicated().any():
        raise PaperTableError("adjudication log must contain exactly one row per result ID")


def _portable_source_run_dir(run_dir: Path) -> tuple[str, str]:
    """Prefer a repository-relative run path without hiding external locations."""

    resolved = run_dir.resolve()
    repository_root = Path(__file__).resolve().parents[3]
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError:
        return str(resolved), "absolute-external"
    return relative.as_posix() or ".", "repository-relative"


def _write_inventory(
    path: Path,
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    output_hashes: Mapping[str, str],
    expected_counts: Mapping[str, int],
    verification: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    allow_smoke: bool,
) -> None:
    source_run_dir, source_run_dir_path_kind = _portable_source_run_dir(run_dir)
    inventory = {
        "schema_version": "paper-table-inventory-v1",
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": source_run_dir,
        "source_run_dir_path_kind": source_run_dir_path_kind,
        "mode": "smoke" if manifest["smoke"] else "formal",
        "allow_smoke_flag": bool(allow_smoke),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": manifest["protocol_sha256"],
        "build_script": SCRIPT_ID,
        "build_script_sha256": _sha256(Path(__file__).resolve()),
        "expected_cell_counts": dict(expected_counts),
        "observed_complete_counts": {
            key: int(verification["observed_complete_counts"][key])
            for key in ("ppo", "ga", "heuristic", "milp")
        },
        "design_counts": {
            "models": len(protocol["models"]),
            "test_scales": len(protocol["test_scales"]),
            "instances_by_scale": {
                scale: _expected_instances(protocol, scale) for scale in _scales(protocol)
            },
            "preferences": len(_preferences(protocol)),
            "ppo_restarts": int(protocol["methods"]["ppo"]["restarts"]),
            "ga_restarts": int(protocol["methods"]["ga"]["restarts"]),
            "heuristic_restarts": int(protocol["methods"]["heuristic"]["restarts"]),
            "milp_restarts": int(protocol["methods"]["milp"]["restarts"]),
        },
        "input_files_sha256": dict(sorted(input_hashes.items())),
        "output_files_sha256": dict(sorted(output_hashes.items())),
        "output_row_counts": {name: int(len(frame)) for name, frame in sorted(frames.items())},
        "result_ids": sorted(ALLOWED_HYPOTHESIS_IDS),
        "inventory_self_hash": None,
        "inventory_self_hash_note": (
            "Omitted to avoid recursive self-hashing; every other emitted artifact is hashed above."
        ),
    }
    _atomic_write_text(
        path,
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )


def build_paper_tables(
    run_dir: Path, output_dir: Path, *, allow_smoke: bool = False
) -> list[Path]:
    manifest, protocol, tables, input_hashes, expected_counts, verification = _load_inputs(
        run_dir, allow_smoke=allow_smoke
    )
    table3, table3_md = _table3(manifest, protocol, tables)
    table4, table4_md = _table4(manifest, protocol, tables)
    table5, table5_md = _table5_scale_design(protocol)
    core_rows, _ = _generalization_core(protocol, tables)
    table6, table6_md = _table6(protocol, core_rows)
    table7, table7_md = _table7(manifest, core_rows)
    risk, risk_md = _risk_appendix(protocol, tables)
    all_methods, all_methods_md = _all_methods_appendix(manifest, protocol, tables)
    milp_status, milp_status_md = _milp_status_appendix(protocol, tables)
    registry = _registry(
        manifest,
        table3,
        table4,
        table6,
        table7,
        risk,
        all_methods,
        milp_status,
    )
    adjudication = _adjudication(protocol, tables, table3, table7)
    _validate_result_ids(registry, adjudication)

    output_dir = output_dir.resolve()
    run_root = run_dir.resolve()
    try:
        output_dir.relative_to(run_root)
    except ValueError:
        pass
    else:
        raise PaperTableError("output directory must not be inside the immutable run directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".paper-tables-", dir=str(output_dir.parent))
    ).resolve()
    frames = {
        "table3_small_scale.csv": table3,
        "table4_large_scale.csv": table4,
        "table5_scale_design.csv": table5,
        "table6_generalization_matrix.csv": table6,
        "table7_generalization_operational.csv": table7,
        "appendix_risk_components.csv": risk,
        "appendix_all_methods.csv": all_methods,
        "appendix_milp_status.csv": milp_status,
        "results-registry.csv": registry,
        "adjudication-log.csv": adjudication,
    }
    markdowns = {
        "table3_small_scale.md": table3_md,
        "table4_large_scale.md": table4_md,
        "table5_scale_design.md": table5_md,
        "table6_generalization_matrix.md": table6_md,
        "table7_generalization_operational.md": table7_md,
        "appendix_risk_components.md": risk_md,
        "appendix_all_methods.md": all_methods_md,
        "appendix_milp_status.md": milp_status_md,
    }
    try:
        for name, frame in frames.items():
            _write_frame(staging / name, frame)
        for name, markdown in markdowns.items():
            _atomic_write_text(staging / name, markdown)
        output_hashes = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        inventory_name = "paper-table-inventory.json"
        _write_inventory(
            staging / inventory_name,
            run_dir=run_root,
            manifest=manifest,
            protocol=protocol,
            input_hashes=input_hashes,
            output_hashes=output_hashes,
            expected_counts=expected_counts,
            verification=verification,
            frames=frames,
            allow_smoke=allow_smoke,
        )
        staged_files = sorted(
            path
            for path in staging.iterdir()
            if path.is_file() and path.name != inventory_name
        )
        staged_inventory = staging / inventory_name
        output_dir.mkdir(parents=True, exist_ok=True)
        inventory_destination = output_dir / inventory_name
        # An inventory is the completion marker for one coherent publication
        # set.  Remove any marker from a previous build before replacing member
        # files so an interrupted rebuild cannot leave a stale marker visible.
        inventory_destination.unlink(missing_ok=True)
        outputs: list[Path] = []
        for staged_path in staged_files:
            destination = output_dir / staged_path.name
            os.replace(staged_path, destination)
            outputs.append(destination)
        os.replace(staged_inventory, inventory_destination)
        outputs.append(inventory_destination)
        return outputs
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create paper Tables 3, 4, 5, 6, 7, all-method/risk/MILP-status "
            "appendices, result registries, and a hash inventory from a verified "
            "supplementary-experiment run."
        )
    )
    parser.add_argument("run_dir", type=Path, help="verified formal run directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"destination directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--allow-smoke",
        action="store_true",
        help="explicitly permit the one-instance smoke protocol (testing only)",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        outputs = build_paper_tables(
            args.run_dir, args.output_dir, allow_smoke=args.allow_smoke
        )
    except PaperTableError as exc:
        raise SystemExit(f"paper-table build failed: {exc}") from exc
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
