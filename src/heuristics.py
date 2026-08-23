from __future__ import annotations

import random
from typing import Dict, Iterable, List, Tuple

from hazardous_waste_model import ModelParams, normalize_pair, pickup_type
from src.solution_utils import RoutePlan, make_route, route_plan_to_solution, terminal_inventory


def _best_facility_for_type(params: ModelParams, waste_type: str) -> str:
    candidates = [j for j in params.facilities if params.technology[j, waste_type] == 1]
    return candidates[0] if candidates else params.facilities[0]


def _can_add(params: ModelParams, route_nodes: List[str], node: str, load: float) -> bool:
    qty = terminal_inventory(params, node)
    if load + qty > params.vehicle_capacity + 1e-9:
        return False
    previous = route_nodes[-1] if route_nodes else None
    if previous is not None and (previous, node) not in params.distance:
        return False
    node_type = pickup_type(node)
    for existing in route_nodes:
        if params.compatibility[normalize_pair(pickup_type(existing), node_type)] == 0:
            return False
    return True


def build_greedy_initial_plan(params: ModelParams, seed: int = 0) -> RoutePlan:
    rng = random.Random(seed)
    routes: RoutePlan = {}
    remaining = set(params.pickup_nodes)
    last_period = params.periods[-1]

    for vehicle in params.vehicles:
        if not remaining:
            break
        first = max(remaining, key=lambda n: (terminal_inventory(params, n), rng.random()))
        facility = _best_facility_for_type(params, pickup_type(first))
        current_nodes: List[str] = []
        current_load = 0.0
        while remaining:
            candidates = [
                node
                for node in remaining
                if params.technology[facility, pickup_type(node)] == 1 and _can_add(params, current_nodes, node, current_load)
            ]
            if not candidates:
                break
            def score(candidate: str) -> tuple[float, float]:
                previous = current_nodes[-1] if current_nodes else facility
                return terminal_inventory(params, candidate), -params.distance.get((previous, candidate), 0.0)

            node = max(candidates, key=score)
            current_nodes.append(node)
            current_load += terminal_inventory(params, node)
            remaining.remove(node)
        if current_nodes:
            routes[vehicle, last_period] = make_route(facility, current_nodes)
    return routes


def build_greedy_initial_solution(params: ModelParams, seed: int = 0) -> Dict[str, object]:
    return route_plan_to_solution(params, build_greedy_initial_plan(params, seed=seed))
