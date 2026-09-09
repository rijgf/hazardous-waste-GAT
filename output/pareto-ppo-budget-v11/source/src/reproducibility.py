from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple


RouteKey = Tuple[str, int]
RoutePlan = Dict[RouteKey, list[str]]

_PLAN_SCHEMA = "route-plan-v1"
_SEED_SCHEMA = "sha256-seed-v1"
_DEFAULT_PACKAGES = ("numpy", "scipy", "pandas", "matplotlib", "torch")


def canonical_json_dumps(value: Any) -> str:
    """Return a stable UTF-8 JSON representation suitable for hashing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def json_sha256(value: Any) -> str:
    """Hash a JSON-compatible value after canonical serialization."""

    payload = canonical_json_dumps(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(master_seed: int | str, *namespace: Any, bits: int = 63) -> int:
    """Derive a call-order-independent integer seed using SHA-256.

    Namespace parts are encoded as canonical JSON rather than Python's unstable
    ``hash()``. The default 63-bit result works with Python, NumPy generators,
    and PyTorch while remaining a non-negative signed integer.
    """

    if not 1 <= bits <= 64:
        raise ValueError("bits must be between 1 and 64")
    payload = {
        "schema": _SEED_SCHEMA,
        "master_seed": master_seed,
        "namespace": list(namespace),
    }
    value = int.from_bytes(
        hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    return value & ((1 << bits) - 1)


def plan_to_canonical_data(plan: Mapping[RouteKey, Sequence[str]]) -> Dict[str, Any]:
    """Convert a route plan to a sorted, JSON-compatible representation."""

    routes = []
    for key, route in plan.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("plan keys must be (vehicle, period) tuples")
        vehicle, period = key
        if not isinstance(vehicle, str):
            raise TypeError("vehicle must be a string")
        if not isinstance(period, int) or isinstance(period, bool):
            raise TypeError("period must be an integer")
        if isinstance(route, (str, bytes)) or not isinstance(route, Sequence):
            raise TypeError("route must be a sequence of node strings")
        nodes = list(route)
        if not all(isinstance(node, str) for node in nodes):
            raise TypeError("route nodes must be strings")
        routes.append({"vehicle": vehicle, "period": period, "route": nodes})

    routes.sort(key=lambda item: (item["period"], item["vehicle"]))
    return {"schema": _PLAN_SCHEMA, "routes": routes}


def serialize_plan(plan: Mapping[RouteKey, Sequence[str]]) -> str:
    """Serialize a route plan to canonical JSON."""

    return canonical_json_dumps(plan_to_canonical_data(plan))


def deserialize_plan(payload: str | bytes | Mapping[str, Any]) -> RoutePlan:
    """Deserialize and validate a canonical route-plan payload."""

    if isinstance(payload, bytes):
        data = json.loads(payload.decode("utf-8"))
    elif isinstance(payload, str):
        data = json.loads(payload)
    elif isinstance(payload, Mapping):
        data = dict(payload)
    else:
        raise TypeError("payload must be JSON text, bytes, or a mapping")

    if data.get("schema") != _PLAN_SCHEMA:
        raise ValueError(f"unsupported plan schema: {data.get('schema')!r}")
    routes = data.get("routes")
    if not isinstance(routes, list):
        raise TypeError("routes must be a list")

    plan: RoutePlan = {}
    for item in routes:
        if not isinstance(item, Mapping):
            raise TypeError("each route entry must be a mapping")
        vehicle = item.get("vehicle")
        period = item.get("period")
        route = item.get("route")
        if not isinstance(vehicle, str):
            raise TypeError("vehicle must be a string")
        if not isinstance(period, int) or isinstance(period, bool):
            raise TypeError("period must be an integer")
        if not isinstance(route, list) or not all(isinstance(node, str) for node in route):
            raise TypeError("route must be a list of node strings")
        key = (vehicle, period)
        if key in plan:
            raise ValueError(f"duplicate route key: {key!r}")
        plan[key] = list(route)
    return plan


def plan_sha256(plan: Mapping[RouteKey, Sequence[str]]) -> str:
    """Return a stable digest independent of route-key insertion order."""

    return hashlib.sha256(serialize_plan(plan).encode("utf-8")).hexdigest()


def environment_snapshot(packages: Sequence[str] = _DEFAULT_PACKAGES) -> Dict[str, Any]:
    """Capture software, platform, and optional PyTorch runtime versions."""

    package_versions: Dict[str, str | None] = {}
    for package in sorted(set(packages), key=str.casefold):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    snapshot: Dict[str, Any] = {
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "description": platform.platform(),
        },
        "packages": package_versions,
    }

    if package_versions.get("torch") is not None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            snapshot["torch_runtime"] = {
                "cuda_available": cuda_available,
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "device_count": torch.cuda.device_count(),
                "devices": [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ] if cuda_available else [],
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
            }
        except Exception as exc:  # pragma: no cover - depends on local drivers
            snapshot["torch_runtime"] = {"error": f"{type(exc).__name__}: {exc}"}

    return snapshot


def _git_command(repo_path: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def git_status_snapshot(repo_path: str | Path = ".") -> Dict[str, Any]:
    """Capture commit, branch, porcelain status, and tracked-diff hashes."""

    path = Path(repo_path).resolve()
    try:
        root_result = _git_command(path, "rev-parse", "--show-toplevel")
    except OSError as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if root_result.returncode != 0:
        error = root_result.stderr.decode("utf-8", errors="replace").strip()
        return {"available": False, "error": error or "not a git repository"}

    root = Path(root_result.stdout.decode("utf-8", errors="replace").strip()).resolve()
    commit_result = _git_command(root, "rev-parse", "HEAD")
    branch_result = _git_command(root, "rev-parse", "--abbrev-ref", "HEAD")
    status_result = _git_command(root, "status", "--porcelain=v1", "--untracked-files=all")
    diff_result = _git_command(root, "diff", "--binary", "HEAD", "--")

    status_text = status_result.stdout.decode("utf-8", errors="replace")
    status_lines = status_text.splitlines()
    return {
        "available": True,
        "root": str(root),
        "commit": commit_result.stdout.decode("utf-8", errors="replace").strip()
        if commit_result.returncode == 0 else None,
        "branch": branch_result.stdout.decode("utf-8", errors="replace").strip()
        if branch_result.returncode == 0 else None,
        "dirty": bool(status_lines),
        "status": status_lines,
        "status_sha256": hashlib.sha256(status_result.stdout).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff_result.stdout).hexdigest()
        if diff_result.returncode == 0 else None,
    }
