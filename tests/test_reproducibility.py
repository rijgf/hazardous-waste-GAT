from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReproducibilityTests(unittest.TestCase):
    def _initial_plan_json(self, hash_seed: str) -> str:
        code = (
            "import json; from dataclasses import replace; "
            "from sample_params import build_sample_params; "
            "from src.heuristics import build_greedy_initial_plan; "
            "params=build_sample_params(); "
            "params=replace(params, generation={k:1.0 for k in params.generation}); "
            "p=build_greedy_initial_plan(params, seed=42); "
            "print(json.dumps([{'vehicle':k[0],'period':k[1],'route':v} "
            "for k,v in sorted(p.items())], sort_keys=True))"
        )
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed
        return subprocess.check_output(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            text=True,
        ).strip()

    def test_initial_plan_is_stable_across_python_hash_seeds(self) -> None:
        self.assertEqual(self._initial_plan_json("1"), self._initial_plan_json("999"))


if __name__ == "__main__":
    unittest.main()
