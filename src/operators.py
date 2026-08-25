from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from hazardous_waste_model import ModelParams, normalize_pair, pickup_owner, pickup_type
from src.solution_utils import RoutePlan, enhanced_check_solution, make_route, route_plan_to_solution


OPERATORS = [
    "relocate",
    "swap",
    "two_opt",
    "change_facility",
    "move_visit_period",
    "add_early_service",
    "split_by_waste_type",
    "split_route",
]


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


def _sorted_keys(plan: RoutePlan) -> List[Tuple[str, int]]:
    return sorted(plan, key=lambda key: (key[1], key[0]))


def _visits(plan: RoutePlan) -> List[Tuple[Tuple[str, int], str]]:
    return [(key, node) for key in _sorted_keys(plan) for node in route_tasks(plan[key])]


def _available_quantities(
    params: ModelParams,
    visits_by_period: Dict[int, set[str]],
) -> Dict[Tuple[str, int], float]:
    inventory = {
        node: float(params.initial_producer_inventory[pickup_owner(node), pickup_type(node)])
        for node in params.pickup_nodes
    }
    available: Dict[Tuple[str, int], float] = {}
    for period in params.periods:
        for node in params.pickup_nodes:
            owner = pickup_owner(node)
            waste = pickup_type(node)
            inventory[node] += float(params.generation[owner, waste, period])
            available[node, period] = inventory[node]
            if node in visits_by_period.get(period, set()):
                inventory[node] = 0.0
    return available


def _compatible(params: ModelParams, tasks: Iterable[str], node: str) -> bool:
    waste = pickup_type(node)
    return all(
        params.compatibility[normalize_pair(pickup_type(existing), waste)] == 1
        for existing in tasks
    )


def _can_append(
    params: ModelParams,
    facility: str,
    tasks: List[str],
    load: float,
    node: str,
    quantity: float,
) -> bool:
    if params.technology[facility, pickup_type(node)] != 1:
        return False
    if load + quantity > params.vehicle_capacity + 1e-9:
        return False
    if not _compatible(params, tasks, node):
        return False
    previous = tasks[-1] if tasks else facility
    return (previous, node) in params.distance and (node, facility) in params.distance


def _facility_candidates(params: ModelParams, tasks: List[str], preferred: str | None) -> List[str]:
    candidates = []
    if preferred in params.facilities:
        candidates.append(preferred)
    candidates.extend(facility for facility in params.facilities if facility not in candidates)
    return [
        facility
        for facility in candidates
        if all(params.technology[facility, pickup_type(node)] == 1 for node in tasks)
    ]


def repair_plan(params: ModelParams, plan: RoutePlan, seed: int = 0) -> Tuple[RoutePlan, bool]:
    """Repair a plan without erasing valid cross-period visits.

    A visit is identified by ``(pickup node, period)``. Duplicate visits inside one
    period are removed, while the same pickup node may be serviced in multiple
    periods. Every pickup node is guaranteed a terminal-period visit so terminal
    inventory can be cleared.
    """

    del seed  # Repair is deliberately deterministic for local reproducibility.
    ordered_by_period: Dict[int, List[Tuple[str, str, str]]] = {
        period: [] for period in params.periods
    }
    seen: set[Tuple[str, int]] = set()
    for (vehicle, period) in _sorted_keys(plan):
        if vehicle not in params.vehicles or period not in params.periods:
            continue
        route = plan[vehicle, period]
        preferred_facility = route[0] if route and route[0] in params.facilities else params.facilities[0]
        for node in route_tasks(route):
            visit = (node, period)
            if node in params.pickup_nodes and visit not in seen:
                ordered_by_period[period].append((node, vehicle, preferred_facility))
                seen.add(visit)

    last_period = params.periods[-1]
    terminal_seen = {node for node, _, _ in ordered_by_period[last_period]}
    for node in params.pickup_nodes:
        if node not in terminal_seen:
            ordered_by_period[last_period].append((node, params.vehicles[0], params.facilities[0]))

    visit_sets = {
        period: {node for node, _, _ in entries}
        for period, entries in ordered_by_period.items()
    }
    quantities = _available_quantities(params, visit_sets)
    repaired: RoutePlan = {}

    for period in params.periods:
        loads = {vehicle: 0.0 for vehicle in params.vehicles}
        tasks_by_vehicle: Dict[str, List[str]] = {vehicle: [] for vehicle in params.vehicles}
        facility_by_vehicle: Dict[str, str] = {}
        for node, preferred_vehicle, preferred_facility in ordered_by_period[period]:
            quantity = quantities[node, period]
            vehicle_order = [preferred_vehicle] + [
                vehicle for vehicle in params.vehicles if vehicle != preferred_vehicle
            ]
            placed = False
            for vehicle in vehicle_order:
                tasks = tasks_by_vehicle[vehicle]
                if tasks:
                    facility = facility_by_vehicle[vehicle]
                    if _can_append(params, facility, tasks, loads[vehicle], node, quantity):
                        tasks.append(node)
                        loads[vehicle] += quantity
                        placed = True
                        break
                    continue
                for facility in _facility_candidates(params, [node], preferred_facility):
                    if _can_append(params, facility, [], 0.0, node, quantity):
                        facility_by_vehicle[vehicle] = facility
                        tasks.append(node)
                        loads[vehicle] = quantity
                        placed = True
                        break
                if placed:
                    break
            if not placed:
                return clone_plan(plan), False

        for vehicle in params.vehicles:
            tasks = tasks_by_vehicle[vehicle]
            if tasks:
                repaired[vehicle, period] = make_route(facility_by_vehicle[vehicle], tasks)

    solution = route_plan_to_solution(params, repaired)
    if enhanced_check_solution(params, solution):
        return clone_plan(plan), False
    return repaired, True


def _target_route(
    params: ModelParams,
    candidate: RoutePlan,
    vehicle: str,
    period: int,
    node: str,
) -> Tuple[str, int]:
    key = (vehicle, period)
    if key not in candidate:
        facilities = _facility_candidates(params, [node], None)
        candidate[key] = make_route(facilities[0] if facilities else params.facilities[0], [])
    return key


def apply_operator(
    params: ModelParams,
    plan: RoutePlan,
    action: OperatorAction,
    seed: int = 0,
) -> Tuple[RoutePlan, bool, str]:
    rng = random.Random(seed)
    candidate = clone_plan(plan)
    keys = _sorted_keys(candidate)
    visits = _visits(candidate)
    if not keys or not visits:
        return candidate, False, "empty_plan"

    op = OPERATORS[action.operator_id % len(OPERATORS)]
    changed = False

    if op == "relocate":
        (source_key, task) = visits[action.object_1 % len(visits)]
        target_key = keys[action.object_2 % len(keys)]
        candidate[source_key].remove(task)
        insert_at = 1 + action.object_3 % max(1, len(candidate[target_key]) - 1)
        candidate[target_key].insert(insert_at, task)
        changed = source_key != target_key or candidate[target_key].index(task) != plan[target_key].index(task)

    elif op == "swap" and len(visits) >= 2:
        key1, task1 = visits[action.object_1 % len(visits)]
        key2, task2 = visits[action.object_2 % len(visits)]
        if (key1, task1) != (key2, task2):
            index1 = candidate[key1].index(task1)
            index2 = candidate[key2].index(task2)
            candidate[key1][index1], candidate[key2][index2] = task2, task1
            changed = True

    elif op == "two_opt":
        key = keys[action.object_1 % len(keys)]
        tasks = route_tasks(candidate[key])
        if len(tasks) >= 2:
            start = action.object_2 % len(tasks)
            stop = action.object_3 % len(tasks)
            start, stop = sorted((start, stop))
            if start < stop:
                tasks[start : stop + 1] = reversed(tasks[start : stop + 1])
                candidate[key] = make_route(candidate[key][0], tasks)
                changed = True

    elif op == "change_facility":
        key = keys[action.object_1 % len(keys)]
        facility = params.facilities[action.object_2 % len(params.facilities)]
        tasks = route_tasks(candidate[key])
        if facility != candidate[key][0] and all(
            params.technology[facility, pickup_type(node)] == 1 for node in tasks
        ):
            candidate[key] = make_route(facility, tasks)
            changed = True

    elif op == "move_visit_period":
        source_key, task = visits[action.object_1 % len(visits)]
        target_period = params.periods[action.object_2 % len(params.periods)]
        target_vehicle = params.vehicles[action.object_3 % len(params.vehicles)]
        target_key = _target_route(params, candidate, target_vehicle, target_period, task)
        if source_key != target_key and task not in route_tasks(candidate[target_key]):
            candidate[source_key].remove(task)
            candidate[target_key].insert(-1, task)
            changed = True

    elif op == "add_early_service":
        task = sorted(params.pickup_nodes)[action.object_1 % len(params.pickup_nodes)]
        early_periods = params.periods[:-1]
        if early_periods:
            target_period = early_periods[action.object_2 % len(early_periods)]
            target_vehicle = params.vehicles[action.object_3 % len(params.vehicles)]
            target_key = _target_route(params, candidate, target_vehicle, target_period, task)
            if task not in route_tasks(candidate[target_key]):
                candidate[target_key].insert(-1, task)
                changed = True

    elif op == "split_by_waste_type":
        source_key = keys[action.object_1 % len(keys)]
        tasks = route_tasks(candidate[source_key])
        waste = params.waste_types[action.object_2 % len(params.waste_types)]
        moving = [node for node in tasks if pickup_type(node) == waste]
        target_vehicle = params.vehicles[action.object_3 % len(params.vehicles)]
        target_key = _target_route(params, candidate, target_vehicle, source_key[1], moving[0]) if moving else source_key
        if moving and target_key != source_key:
            for node in moving:
                candidate[source_key].remove(node)
                if node not in route_tasks(candidate[target_key]):
                    candidate[target_key].insert(-1, node)
            changed = True

    elif op == "split_route":
        source_key = keys[action.object_1 % len(keys)]
        tasks = route_tasks(candidate[source_key])
        if len(tasks) >= 2:
            task = tasks[action.object_2 % len(tasks)]
            target_vehicle = params.vehicles[action.object_3 % len(params.vehicles)]
            target_key = _target_route(params, candidate, target_vehicle, source_key[1], task)
            if target_key != source_key:
                candidate[source_key].remove(task)
                candidate[target_key].insert(-1, task)
                changed = True

    if not changed:
        return clone_plan(plan), False, f"{op}_no_change"

    repaired, ok = repair_plan(params, candidate, seed=rng.randrange(1_000_000))
    if not ok:
        return clone_plan(plan), False, f"{op}_repair_failed"
    return repaired, True, op
