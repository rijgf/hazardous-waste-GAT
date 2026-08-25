from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model_config(name: str) -> Dict[str, Any]:
    data = load_json(CONFIG_DIR / "model_config.json")
    return dict(data[name])


def load_algorithm_config() -> Dict[str, Any]:
    return load_json(CONFIG_DIR / "algorithm_config.json")


def load_network_config() -> Dict[str, Any]:
    return load_json(CONFIG_DIR / "network_config.json")


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output
