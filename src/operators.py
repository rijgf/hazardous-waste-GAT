from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import List, Tuple

from hazardous_waste_model import ModelParams, normalize_pair, pickup_type
from src.heuristics import _best_facility_for_type, _can_add
from src.solution_utils import RoutePlan, make_route, terminal_inventory


OPERATORS = ["relocate", "swap", "two_opt", "change_facility", "delay_or_advance_period"]


@dataclass(frozen=True)
class OperatorAction:
    operator_id: int
    object_1: int
    object_2: int
    object_3: int = 0


def clone_plan(plan: RoutePlan) -> RoutePlan:
    return {key: list(route) for key, route in plan.items()}


def route_tasks(route: List[str]) -> List[str]:
    return list(route[1:-1]) if len(route) >= 3 else []


def flatten_tasks(plan: RoutePlan) -> List[str]:
    tasks: List[str] = []
    for key in sorted(plan.keys(), key=lambda x: (x[1], x[0])):
        tasks.extend(route_tasks(plan[key]))
    return tasks


def repair_plan(params: ModelParams, plan: RoutePlan, seed: int = 0) -> Tuple[RoutePlan, bool]:
    ordered: List[str] = []
    seen = set()
    for node in flatten_tasks(plan):
        if node in params.pickup_nodes and node not in seen:
            ordered.append(node)
            seen.add(node)
    for node in params.pickup_nodes:
        if node not in seen:
            ordered.append(node)
            seen.add(node)

    repaired: RoutePlan = {}
    remaining = list(ordered)
    last_period = params.periods[-1]
    for vehicle in params.vehicles:
        if not remaining:
            break
        first = remaining[0]
        current_facility = _best_facility_for_type(params, pickup_type(first))
        current_nodes: List[str] = []
        current_load = 0.0

        while True:
            candidates = [
                node
                for node in remaining
                if params.technology[current_facility, pickup_type(node)] == 1
                and _can_add(params, current_nodes, node, current_load)
            ]
            if not candidates:
                break
            node = candidates[0]
            current_nodes.append(node)
            current_load += terminal_inventory(params, node)
            remaining.remove(node)

        if current_nodes:
            repaired[vehicle, last_period] = make_route(current_facility, current_nodes)

    return repaired, not remaining


def apply_operator(params: ModelParams, plan: RoutePlan, action: OperatorAction, seed: int = 0) -> Tuple[RoutePlan, bool, str]:
    rng = random.Random(seed)
    candidate = clone_plan(plan)
    keys = sorted(candidate.keys(), key=lambda x: (x[1], x[0]))
    all_tasks = flatten_tasks(candidate)
    if not keys or not all_tasks:
        return candidate, False, "empty_plan"

    op = OPERATORS[action.operator_id % len(OPERATORS)]
    changed = False

    if op == "relocate":
        task = all_tasks[action.object_1 % len(all_tasks)]
        source_key = next((key for key in keys if task in candidate[key]), None)
        target_key = keys[action.object_2 % len(keys)]
        if source_key is not None:
            candidate[source_key].remove(task)
            insert_pos = 1 + (action.object_3 % max(1, len(candidate[target_key]) - 1))
            candidate[target_key].insert(insert_pos, task)
            changed = True

    elif op == "swap":
        if len(all_tasks) >= 2:
            t1 = all_tasks[action.object_1 % len(all_tasks)]
            t2 = all_tasks[action.object_2 % len(all_tasks)]
            if t1 != t2:
                k1 = next(key for key in keys if t1 in candidate[key])
                k2 = next(key for key in keys if t2 in candidate[key])
                i1 = candidate[k1].index(t1)
                i2 = candidate[k2].index(t2)
                candidate[k1][i1], candidate[k2][i2] = candidate[k2][i2], candidate[k1][i1]
                changed = True

    elif op == "two_opt":
        key = keys[action.object_1 % len(keys)]
        tasks = route_tasks(candidate[key])
        if len(tasks) >= 3:
            a = action.object_2 % len(tasks)
            b = action.object_3 % len(tasks)
            if a > b:
                a, b = b, a
            if a < b:
                new_tasks = tasks[:a] + list(reversed(tasks[a : b + 1])) + tasks[b + 1 :]
                candidate[key] = make_route(candidate[key][0], new_tasks)
                changed = True

    elif op == "change_facility":
        key = keys[action.object_1 % len(keys)]
        facility = params.facilities[action.object_2 % len(params.facilities)]
        tasks = route_tasks(candidate[key])
        if tasks and all(params.technology[facility, pickup_type(node)] == 1 for node in tasks):
            candidate[key] = make_route(facility, tasks)
            changed = True

    elif op == "delay_or_advance_period":
        key = keys[action.object_1 % len(keys)]
        target_period = params.periods[action.object_2 % len(params.periods)]
        vehicle, _ = key
        if (vehicle, target_period) not in candidate:
            candidate[vehicle, target_period] = candidate.pop(key)
            changed = True

    if not changed:
        return clone_plan(plan), False, f"{op}_no_change"
    repaired, ok = repair_plan(params, candidate, seed=rng.randint(0, 1_000_000))
    if not ok:
        return clone_plan(plan), False, f"{op}_repair_failed"
    return repaired, True, op
