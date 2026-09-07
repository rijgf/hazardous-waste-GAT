from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.supplementary_experiment import (
    DEFAULT_PROTOCOL_PATH,
    evaluate_run,
    export_replication_summary,
    initialize_run,
    lock_test_sets,
    prepare_schedule,
    summarize_run,
    train_models,
    verify_run,
)


def _comma_list(value: str | None) -> Sequence[str] | None:
    if not value:
        return None
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen-model multi-instance supplementary experiments."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Persistent run directory. The protocol default is used when omitted for init.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="Protocol JSON used only by init.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="Lock protocol and create training instances.")
    init.add_argument(
        "--smoke",
        action="store_true",
        help="Create a tiny, clearly non-publication integration run.",
    )

    train = subparsers.add_parser("train", help="Train each selected model exactly once.")
    train.add_argument(
        "--models", default="Train-S,Train-L", help="Comma-separated model ids."
    )

    subparsers.add_parser(
        "lock-tests", help="Generate held-out tests after both models are frozen."
    )
    subparsers.add_parser("schedule", help="Materialize the de-duplicated cell ledger.")

    evaluate = subparsers.add_parser("evaluate", help="Run or resume evaluation cells.")
    evaluate.add_argument(
        "--methods", help="Comma-separated subset: ppo,ga,heuristic,milp."
    )
    evaluate.add_argument("--models", help="Comma-separated model/baseline ids.")
    evaluate.add_argument("--scales", help="Comma-separated Test-1..Test-4 subset.")
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--max-cells", type=int)
    evaluate.add_argument("--retry-failed", action="store_true")
    evaluate.add_argument(
        "--allow-concurrent-ppo",
        action="store_true",
        help=(
            "Allow multiple PPO workers to share one GPU for throughput. "
            "Recorded per-cell times are then contention-affected."
        ),
    )

    subparsers.add_parser("summarize", help="Rebuild all tables from committed cells.")
    verify = subparsers.add_parser("verify", help="Audit hashes/counts and replay samples.")
    verify.add_argument("--replay-per-stratum", type=_positive_int, default=1)
    export = subparsers.add_parser("export", help="Copy compact results into a tracked folder.")
    export.add_argument("destination", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "init":
        result = initialize_run(
            args.config,
            args.run_dir,
            smoke=bool(args.smoke),
        )
    else:
        if args.run_dir is None:
            raise SystemExit("--run-dir is required after init")
        if args.command == "train":
            result = train_models(args.run_dir, _comma_list(args.models))
        elif args.command == "lock-tests":
            result = lock_test_sets(args.run_dir)
        elif args.command == "schedule":
            cells = prepare_schedule(args.run_dir)
            result = {"cell_count": len(cells)}
        elif args.command == "evaluate":
            result = evaluate_run(
                args.run_dir,
                methods=_comma_list(args.methods),
                models=_comma_list(args.models),
                scales=_comma_list(args.scales),
                workers=args.workers,
                max_cells=args.max_cells,
                retry_failed=args.retry_failed,
                allow_concurrent_ppo=args.allow_concurrent_ppo,
            )
        elif args.command == "summarize":
            result = summarize_run(args.run_dir)
        elif args.command == "verify":
            result = verify_run(
                args.run_dir, replay_per_stratum=args.replay_per_stratum
            )
            if not result["passed"]:
                print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
                raise SystemExit(2)
        elif args.command == "export":
            result = export_replication_summary(args.run_dir, args.destination)
        else:
            raise AssertionError(args.command)
    if isinstance(result, Path):
        print(result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
