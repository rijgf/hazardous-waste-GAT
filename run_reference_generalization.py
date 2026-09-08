"""Fixed-reference datasets, multi-instance PPO, and held-out scale transfer.

The v1 training hyperparameters are copied without modification. Existing
checkpoints keep their legacy input encoding; new checkpoints explicitly bind
instance_reference semantics. Every evaluation uses a persisted reference plan.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from sample_params import params_from_json_data, params_to_json_data
from src.heuristics import build_greedy_initial_plan
from src.instance_generator import generate_random_params
from src.reproducibility import (
    deserialize_plan, environment_snapshot, file_sha256, json_sha256,
    plan_sha256, plan_to_canonical_data,
)
from src.solution_utils import evaluate_solution, route_plan_to_solution

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets/reference_generalization_v2"
RUN = ROOT / "outputs/reference_generalization_v2"
BASE = ROOT / "configs/supplementary_experiment.json"
LEGACY = ROOT / "output/supplementary-experiments/replication/models"
SOURCES = ["run_reference_generalization.py", "src/ppo_improver.py", "src/operators.py",
           "src/heuristics.py", "src/solution_utils.py", "src/instance_generator.py",
           "src/reproducibility.py", "sample_params.py", "hazardous_waste_model.py"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def seed(*parts):
    digest = hashlib.sha256(json.dumps(["reference-generalization-v2", 20260908, *parts]).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def source_hashes():
    return {name: file_sha256(ROOT / name) for name in SOURCES}


def load_manifest():
    manifest = read(DATA / "manifest.json")
    if manifest["source_hashes"] != source_hashes():
        raise RuntimeError("Source changed after dataset lock; use a new experiment version")
    return manifest


def prepare():
    if (DATA / "manifest.json").exists():
        return load_manifest()
    cfg = read(BASE)
    records = []
    all_seeds, hashes = set(), set()
    for split, scales, count in [("train", ["Test-1", "Test-4"], 24),
                                  ("test", list(cfg["test_scales"]), 10)]:
        for scale in scales:
            generation = {**cfg["generator_profile"], **cfg["test_scales"][scale]["scale"]}
            for index in range(count):
                instance_seed, initial_seed = seed(split, scale, index, "instance"), seed(split, scale, index, "reference")
                assert instance_seed not in all_seeds and initial_seed not in all_seeds
                all_seeds.update((instance_seed, initial_seed))
                params = generate_random_params(generation, instance_seed)
                payload = params_to_json_data(params)
                instance_hash = json_sha256(payload)
                assert instance_hash not in hashes
                hashes.add(instance_hash)
                plan = build_greedy_initial_plan(params, initial_seed)
                metrics = evaluate_solution(params, route_plan_to_solution(params, plan), (.5, .5))
                if not metrics["feasible"] or min(metrics["cost"], metrics["risk"]) <= 0:
                    raise RuntimeError(f"Invalid reference for {split}/{scale}/{index}: {metrics}")
                entry = {
                    "instance_id": f"{split}-{scale}-{index:03d}", "split": split,
                    "scale": scale, "index": index, "instance_seed": instance_seed,
                    "reference_seed": initial_seed, "instance_sha256": instance_hash,
                    "params": payload, "reference_plan": plan_to_canonical_data(plan),
                    "reference_plan_sha256": plan_sha256(plan),
                    "b_C": metrics["cost"], "b_R": metrics["risk"],
                    "reference_metrics": metrics,
                }
                relative = f"{split}/{scale}/{index:03d}.json"
                path = DATA / relative
                if path.exists() and read(path) != entry:
                    raise RuntimeError(f"Refusing to overwrite changed dataset file: {path}")
                write(path, entry)
                records.append({"path": relative, "sha256": file_sha256(path),
                                **{k: entry[k] for k in ("instance_id", "split", "scale", "index", "b_C", "b_R", "instance_sha256")}})
    network = {**cfg["network"], "objective_normalization": "instance_reference"}
    manifest = {
        "protocol": "reference-generalization-v2", "training_instances_per_scale": 24,
        "test_instances_per_scale": 10, "training_scales": ["Test-1", "Test-4"],
        "algorithm": cfg["algorithm"], "network": network,
        "generator_profile": cfg["generator_profile"],
        "scales": {name: {"scale": item["scale"], "instances": 10} for name, item in cfg["test_scales"].items()},
        "preferences": cfg["preferences"], "restarts": 3,
        "base_config_sha256": file_sha256(BASE), "records": records,
        "source_hashes": source_hashes(), "environment": environment_snapshot(),
        "seed_count": len(all_seeds), "unique_instance_count": len(hashes),
        "train_test_hash_overlap": 0,
        "legacy_checkpoints": {n: file_sha256(LEGACY / f"{n}_model.pt") for n in ("small", "large")},
    }
    # The two datasets are locked before training; test outcomes never select a model.
    write(DATA / "manifest.json", manifest)
    print("DATASETS LOCKED: train=48, test=40; training hyperparameters unchanged", flush=True)
    return manifest


def load_instance(record):
    path = DATA / record["path"]
    if file_sha256(path) != record["sha256"]:
        raise RuntimeError(f"Dataset hash mismatch: {path}")
    data = read(path)
    params = params_from_json_data(data["params"])
    plan = deserialize_plan(data["reference_plan"])
    refs = (data["b_C"], data["b_R"])
    if plan_sha256(plan) != data["reference_plan_sha256"]:
        raise RuntimeError("Reference plan changed")
    return data, params, plan, refs


def train_models():
    import torch
    from src.ppo_improver import PPOImprover, PPOTrainingInstance
    manifest = load_manifest()
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for model, scale in [("small", "Test-1"), ("large", "Test-4")]:
        checkpoint = RUN / "models" / f"{model}_model.pt"
        record_path = RUN / "models" / f"{model}.json"
        if record_path.exists():
            record = read(record_path)
            assert file_sha256(checkpoint) == record["checkpoint_sha256"]
            assert record["dataset_manifest_sha256"] == file_sha256(DATA / "manifest.json")
            continue
        if checkpoint.exists():
            raise RuntimeError("Checkpoint exists without completion record")
        instances = []
        for rec in manifest["records"]:
            if rec["split"] == "train" and rec["scale"] == scale:
                data, params, plan, refs = load_instance(rec)
                instances.append(PPOTrainingInstance(data["instance_id"], params, plan, refs))
        first = instances[0]
        training_seed = seed("training", model)
        improver = PPOImprover(first.params, {"ppo": manifest["algorithm"]["ppo"]},
                               manifest["network"], seed=training_seed, objective_refs=first.objective_refs)
        progress_path = RUN / f"training/{model}_progress.json"
        started = time.perf_counter()
        def progress(iteration, metrics):
            elapsed = time.perf_counter() - started
            write(progress_path, {"iteration": iteration + 1, "elapsed_seconds": elapsed, **metrics})
            print(f"TRAIN {model} {iteration+1}/20 elapsed={elapsed:.1f}s {metrics}", flush=True)
        history = improver.train(checkpoint, training_instances=instances, progress_callback=progress)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        assert set(improver.training_instance_counts.values()) == {20}
        write(RUN / f"training/{model}_history.json", history)
        record = {"model": model, "training_seed": training_seed, "training_seconds": elapsed,
                  "checkpoint_sha256": file_sha256(checkpoint),
                  "dataset_manifest_sha256": file_sha256(DATA / "manifest.json"),
                  "instance_episode_counts": improver.training_instance_counts,
                  "algorithm": manifest["algorithm"]["ppo"], "network": manifest["network"]}
        write(record_path, record)
        del improver
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate_group(task):
    variant, rec = task
    import torch
    from src.ppo_improver import PPOImprover
    torch.set_num_threads(1)
    manifest = read(DATA / "manifest.json")
    data, params, plan, refs = load_instance(rec)
    checkpoint_hash = None
    improver = None
    if variant != "GA":
        model = "small" if variant.endswith("S") else "large"
        checkpoint = (LEGACY if variant.startswith("Old") else RUN / "models") / f"{model}_model.pt"
        checkpoint_hash = file_sha256(checkpoint)
        expected = (manifest["legacy_checkpoints"][model] if variant.startswith("Old")
                    else read(RUN / "models" / f"{model}.json")["checkpoint_sha256"])
        if checkpoint_hash != expected:
            raise RuntimeError("Checkpoint hash mismatch")
        improver = PPOImprover.from_frozen_checkpoint(params, checkpoint, objective_refs=refs)
        if variant.startswith("New"):
            assert improver.encoder.objective_refs == refs
        else:
            # Legacy input semantics remain unchanged; its search uses the same
            # reference initial plan, hence the same b values (all > 1 here).
            assert min(refs) > 1
    completed = 0
    for pref in manifest["preferences"]:
        preference = (pref["cost_weight"], pref["risk_weight"])
        for restart in range(3):
            cell = f"{variant}_{rec['scale']}_{rec['index']:03d}_{pref['id']}_{restart}"
            target = RUN / "cells" / f"{cell}.json"
            eval_seed = seed("evaluate", rec["scale"], rec["index"], pref["id"], restart)
            identity = {"variant": variant, "scale": rec["scale"], "index": rec["index"],
                        "preference": list(preference), "restart": restart, "seed": eval_seed,
                        "dataset_sha256": rec["sha256"], "checkpoint_sha256": checkpoint_hash,
                        "b_C": refs[0], "b_R": refs[1]}
            if target.exists():
                saved = read(target)
                if any(saved[k] != v for k, v in identity.items()):
                    raise RuntimeError(f"Changed cell identity: {cell}")
                completed += 1
                continue
            started = time.perf_counter()
            if variant == "GA":
                from dataclasses import replace
                import sys
                sys.path.insert(0, str(ROOT / "遗传算法"))
                from genetic_algorithm import ClassicGeneticAlgorithm, GAConfig
                from src.supplementary_experiment import _initial_chromosome
                gp = replace(params, cost_weight=preference[0]/refs[0], risk_weight=preference[1]/refs[1])
                result = ClassicGeneticAlgorithm(gp, GAConfig(random_seed=eval_seed, **manifest["algorithm"]["ga_by_scale"][rec["scale"]]),
                                                initial_chromosome=_initial_chromosome(params, plan),
                                                initial_solution=route_plan_to_solution(params, plan)).run()
                solution = result.best_solution
            else:
                result = improver.improve(preference, seed=eval_seed, initial_plan=plan)
                solution = result.solution
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            metrics = evaluate_solution(params, solution, preference, objective_refs=refs)
            if not metrics["feasible"]:
                raise RuntimeError(f"Infeasible evaluation retained as task failure: {cell}: {metrics['violations']}")
            record = {**identity, "seconds": seconds,
                      **{k: metrics[k] for k in ("cost", "risk", "weighted_objective_normalized", "transport_risk", "coload_risk", "producer_inventory_risk", "facility_inventory_risk", "feasible")}}
            if improver is not None:
                record["plan"] = plan_to_canonical_data(result.plan)
                record["plan_sha256"] = plan_sha256(result.plan)
                record["accepted_steps"] = sum(t["accepted"] for t in result.trace)
            write(target, record)
            completed += 1
    return variant, rec["scale"], rec["index"], completed


def evaluate(workers=8):
    manifest = load_manifest()
    for n in ("small", "large"):
        record = read(RUN / "models" / f"{n}.json")
        assert record["dataset_manifest_sha256"] == file_sha256(DATA / "manifest.json")
    tasks = [(v, r) for r in manifest["records"] if r["split"] == "test"
             for v in ("New-S", "New-L", "Old-S", "Old-L", "GA")]
    write(RUN / "evaluation_plan.json", {"workers": workers, "tasks": len(tasks), "expected_cells": 3000,
                                        "same_evaluation_seeds_across_models": True,
                                        "dataset_manifest_sha256": file_sha256(DATA / "manifest.json")})
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(evaluate_group, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(f"EVALUATE {index}/{len(tasks)} {result} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    write(RUN / "evaluation_complete.json", {"seconds": time.perf_counter()-started, "workers": workers,
                                            "cells": len(list((RUN / "cells").glob("*.json")))})


def summarize():
    import pandas as pd
    manifest = load_manifest()
    rows = [read(path) for path in sorted((RUN / "cells").glob("*.json"))]
    if len(rows) != 3000:
        raise RuntimeError(f"Incomplete experiment: {len(rows)}/3000 cells; no partial main table")
    for row in rows:
        if not row["feasible"]:
            raise RuntimeError("Cannot drop failed cells")
        row["risk_weight"] = row["preference"][1]
    df = pd.DataFrame(rows)
    assert not df.duplicated(["variant", "scale", "index", "risk_weight", "restart"]).any()
    per = df.groupby(["variant", "scale", "index", "risk_weight"])[["weighted_objective_normalized", "cost", "risk", "seconds"]].mean().reset_index()
    macro = per.groupby(["variant", "scale", "index"])[["weighted_objective_normalized", "seconds"]].mean().reset_index()
    stats = macro.groupby(["variant", "scale"])[["weighted_objective_normalized", "seconds"]].agg(["mean", "std", "count"])
    RUN.mkdir(parents=True, exist_ok=True)
    per.to_csv(RUN / "per_instance_preference.csv", index=False)
    macro.to_csv(RUN / "per_instance_macro.csv", index=False)
    stats.to_csv(RUN / "summary.csv")
    text = ["# 固定参考尺度与多实例训练：四规模泛化结果", "",
            "训练集：小、大规模各24个实例；测试集：四种规模各10个独立未见实例。训练参数与原配置一致（20轮×24个episode×24步，每轮24次更新），各实例参与20个episode。",
            "每个实例锁定参考解和b_C、b_R。新模型的目标输入、奖励与搜索采用同一参考尺度；旧模型保持其原始编码。每个实例、偏好、模型运行3个相同求解种子。", "",
            "新旧模型同时改变了训练数据与目标输入语义，因此本轮比较不能单独识别归一化修改的因果贡献。旧模型和新模型在完全相同的新测试集上比较；不与历史50实例均值混算。", "",
            "| 方法 | Test-1 J | Test-2 J | Test-3 J | Test-4 J |",
            "| --- | ---: | ---: | ---: | ---: |"]
    for v in ["New-S", "New-L", "Old-S", "Old-L", "GA"]:
        values = [f"{stats.loc[(v,s),('weighted_objective_normalized','mean')]:.6f} ± {stats.loc[(v,s),('weighted_objective_normalized','std')]:.6f}" for s in manifest["scales"]]
        text.append("| " + " | ".join([v, *values]) + " |")
    text += ["", "注：先平均3次重启，再在实例内等权平均5组偏好，最后报告10个实例的均值±样本标准差。J越小越好；不能跨测试规模直接比较J的绝对水平。", "",
             "| 方法 | 测试规模 | 计算时间（s） | 可行率 |",
             "| --- | --- | ---: | ---: |"]
    for v in ["New-S", "New-L", "Old-S", "Old-L", "GA"]:
        for s in manifest["scales"]:
            text.append(f"| {v} | {s} | {stats.loc[(v,s),('seconds','mean')]:.4f} ± {stats.loc[(v,s),('seconds','std')]:.4f} | 100% |")
    text += ["", "时间为并发条件下平均单次求解段耗时，不含模型训练，不能解释为隔离运行速度。", "", "## 同规模优势检验", ""]
    for s, owner, other in [("Test-1", "New-S", "New-L"), ("Test-4", "New-L", "New-S")]:
        a = stats.loc[(owner,s),("weighted_objective_normalized","mean")]
        b = stats.loc[(other,s),("weighted_objective_normalized","mean")]
        text.append(f"- {s}：{owner}={a:.6f}，{other}={b:.6f}；同规模模型均值{'较低' if a < b else '未更低'}。")
    for new, old in [("New-S", "Old-S"), ("New-L", "Old-L")]:
        for s in manifest["scales"]:
            a = stats.loc[(new,s),("weighted_objective_normalized","mean")]
            b = stats.loc[(old,s),("weighted_objective_normalized","mean")]
            text.append(f"- {s}：{new}相对{old}的宏平均目标降幅={(b-a)/b*100:.3f}%。")
    for n in ("small", "large"):
        rec = read(RUN / "models" / f"{n}.json")
        text.append(f"- {n}模型训练耗时：{rec['training_seconds']:.3f}s；checkpoint SHA256：{rec['checkpoint_sha256']}。")
    (RUN / "results.md").write_text("\n".join(text)+"\n", encoding="utf-8")
    write(RUN / "verification.json", {"complete_cells": len(rows), "expected_cells": 3000,
                                      "feasible_cells": int(df.feasible.sum()), "passed": True,
                                      "dataset_manifest_sha256": file_sha256(DATA / "manifest.json"),
                                      "source_hashes": source_hashes()})
    print("REPORT", RUN / "results.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "report", "all"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.command in {"prepare", "all"}: prepare()
    if args.command in {"train", "all"}: train_models()
    if args.command in {"evaluate", "all"}: evaluate(args.workers)
    if args.command in {"report", "all"}: summarize()
