from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isclose
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from hazardous_waste_model import (
    ModelParams,
    check_solution,
    display_node,
    normalize_pair,
    pickup_node,
    pickup_owner,
    pickup_type,
    summarize_solution,
)


RouteKey = Tuple[str, int]
RoutePlan = Dict[RouteKey, List[str]]


@dataclass
class ObjectiveBreakdown:
    fixed_cost: float
    distance_cost: float
    processing_cost: float
    transport_risk: float
    coload_risk: float
    producer_inventory_risk: float
    facility_inventory_risk: float

    @property
    def cost(self) -> float:
        return self.fixed_cost + self.distance_cost + self.processing_cost

    @property
    def risk(self) -> float:
        return self.transport_risk + self.coload_risk + self.producer_inventory_risk + self.facility_inventory_risk


def terminal_inventory(params: ModelParams, node: str) -> float:
    i = pickup_owner(node)
    s = pickup_type(node)
    return params.initial_producer_inventory[i, s] + sum(params.generation[i, s, t] for t in params.periods)


def make_route(facility: str, pickups: Iterable[str]) -> List[str]:
    return [facility, *list(pickups), facility]


def route_plan_to_solution(params: ModelParams, plan: RoutePlan) -> Dict[str, Any]:
    raw: Dict[Tuple[Any, ...], float] = {}
    pickups_by_period = {(n, t): [] for n in params.pickup_nodes for t in params.periods}
    received = {(j, s, t): 0.0 for j in params.facilities for s in params.waste_types for t in params.periods}

    for (vehicle, period), route in plan.items():
        if vehicle not in params.vehicles or period not in params.periods or len(route) < 3:
            continue
        for a, b in zip(route, route[1:]):
            if (a, b) in params.distance:
                raw["x", a, b, vehicle, period] = 1.0
        for node in route[1:-1]:
            if node in params.pickup_nodes:
                pickups_by_period[node, period].append(vehicle)

    for t in params.periods:
        prev_t = params.periods[params.periods.index(t) - 1] if params.periods.index(t) > 0 else None
        for node in params.pickup_nodes:
            i = pickup_owner(node)
            s = pickup_type(node)
            prev = params.initial_producer_inventory[i, s] if prev_t is None else raw.get(("IG", node, prev_t), 0.0)
            bg = prev + params.generation[i, s, t]
            raw["BG", node, t] = bg
            vehicles = pickups_by_period[node, t]
            collected = bg if vehicles else 0.0
            if vehicles:
                raw["q", node, vehicles[0], t] = collected
            raw["IG", node, t] = max(0.0, bg - collected)

        for (vehicle, period), route in plan.items():
            if period != t or vehicle not in params.vehicles or len(route) < 3:
                continue
            load_by_type = {s: 0.0 for s in params.waste_types}
            for a, b in zip(route, route[1:]):
                if a in params.pickup_nodes:
                    st = pickup_type(a)
                    load_by_type[st] += raw.get(("q", a, vehicle, t), 0.0)
                for s, load in load_by_type.items():
                    if load > 1e-9 and (a, b) in params.distance:
                        raw["F", a, b, s, vehicle, t] = load
                if b in params.facilities:
                    for s, load in load_by_type.items():
                        received[b, s, t] += load

        for j in params.facilities:
            for s in params.waste_types:
                r = received[j, s, t]
                raw["R", j, s, t] = r
                prev_id = params.initial_facility_inventory[j, s] if prev_t is None else raw.get(("ID", j, s, prev_t), 0.0)
                bd = prev_id + r
                cap = params.processing_capacity[j, s, t] * params.technology[j, s]
                processed = min(bd, cap)
                raw["BD", j, s, t] = bd
                raw["p", j, s, t] = processed
                raw["ID", j, s, t] = bd - processed

    routes = {
        key: [display_node(node) for node in route]
        for key, route in plan.items()
        if len(route) >= 3
    }
    return {"raw": raw, "routes": routes, "summary": summarize_solution(params, raw), "plan": plan}


def objective_breakdown(params: ModelParams, solution: Mapping[str, Any]) -> ObjectiveBreakdown:
    raw = solution.get("raw", solution)
    used_vehicles = set()
    distance_cost = 0.0
    processing_cost = 0.0
    transport_risk = 0.0
    coload_risk_value = 0.0
    producer_inventory_risk = 0.0
    facility_inventory_risk = 0.0

    arc_loads: Dict[Tuple[str, str, str, int], Dict[str, float]] = {}
    for key, value in raw.items():
        if value <= 1e-9:
            continue
        if key[0] == "x":
            _, a, b, vehicle, period = key
            distance_cost += params.distance[a, b] * params.distance_cost
            used_vehicles.add((vehicle, period))
        elif key[0] == "p":
            _, j, s, _ = key
            processing_cost += params.processing_cost[j, s] * value
        elif key[0] == "F":
            _, a, b, s, vehicle, period = key
            transport_risk += params.accident_probability[a, b] * params.distance[a, b] * params.waste_consequence[s] * value
            arc_loads.setdefault((a, b, vehicle, period), {})[s] = value
        elif key[0] == "IG":
            _, node, _ = key
            producer_inventory_risk += params.producer_inventory_risk[pickup_owner(node), pickup_type(node)] * value
        elif key[0] == "ID":
            _, j, s, _ = key
            facility_inventory_risk += params.facility_inventory_risk[j, s] * value

    for (a, b, _, _), loads in arc_loads.items():
        for s1, s2 in combinations(params.waste_types, 2):
            if loads.get(s1, 0.0) > 1e-9 and loads.get(s2, 0.0) > 1e-9:
                gamma = params.coload_risk.get(normalize_pair(s1, s2), 0.0)
                coload_risk_value += params.accident_probability[a, b] * params.distance[a, b] * gamma * (loads[s1] + loads[s2])

    return ObjectiveBreakdown(
        fixed_cost=len(used_vehicles) * params.vehicle_fixed_cost,
        distance_cost=distance_cost,
        processing_cost=processing_cost,
        transport_risk=transport_risk,
        coload_risk=coload_risk_value,
        producer_inventory_risk=producer_inventory_risk,
        facility_inventory_risk=facility_inventory_risk,
    )


def evaluate_solution(
    params: ModelParams,
    solution: Mapping[str, Any],
    preference: Tuple[float, float] | None = None,
    objective_refs: Tuple[float, float] | None = None,
    check_constraints: bool = True,
) -> Dict[str, Any]:
    pref = preference if preference is not None else (params.cost_weight, params.risk_weight)
    breakdown = objective_breakdown(params, solution)
    violations = check_solution(params, solution) if check_constraints else []
    weighted_raw = pref[0] * breakdown.cost + pref[1] * breakdown.risk
    cost_ref, risk_ref = objective_refs if objective_refs is not None else (1.0, 1.0)
    normalized_cost = breakdown.cost / max(cost_ref, 1e-9)
    normalized_risk = breakdown.risk / max(risk_ref, 1e-9)
    weighted_normalized = pref[0] * normalized_cost + pref[1] * normalized_risk
    return {
        "cost": breakdown.cost,
        "risk": breakdown.risk,
        "normalized_cost": normalized_cost,
        "normalized_risk": normalized_risk,
        "weighted_objective_raw": weighted_raw,
        "weighted_objective_normalized": weighted_normalized,
        "weighted_objective": weighted_normalized if objective_refs is not None else weighted_raw,
        "cost_ref": cost_ref,
        "risk_ref": risk_ref,
        "fixed_cost": breakdown.fixed_cost,
        "distance_cost": breakdown.distance_cost,
        "processing_cost": breakdown.processing_cost,
        "transport_risk": breakdown.transport_risk,
        "coload_risk": breakdown.coload_risk,
        "producer_inventory_risk": breakdown.producer_inventory_risk,
        "facility_inventory_risk": breakdown.facility_inventory_risk,
        "violations": violations,
        "feasible": not violations,
    }


def enhanced_check_solution(params: ModelParams, solution: Mapping[str, Any], tol: float = 1e-5) -> List[str]:
    raw = solution.get("raw", solution)
    violations = list(check_solution(params, solution, tol=tol))
    for t in params.periods:
        for j in params.facilities:
            for s in params.waste_types:
                bd = raw.get(("BD", j, s, t), 0.0)
                processed = raw.get(("p", j, s, t), 0.0)
                cap = params.processing_capacity[j, s, t] * params.technology[j, s]
                if processed > cap + tol:
                    violations.append(f"processing capacity exceeded: facility={j}, waste={s}, t={t}")
                if processed > bd + tol:
                    violations.append(f"processed more than available: facility={j}, waste={s}, t={t}")
        for node in params.pickup_nodes:
            visits = sum(raw.get(("x", a, node, k, t), 0.0) for k in params.vehicles for a, b in params.arcs if b == node)
            if visits > 0.5:
                i, s = pickup_owner(node), pickup_type(node)
                feasible_facilities = []
                for k in params.vehicles:
                    for a, b in params.arcs:
                        if b == node and raw.get(("x", a, b, k, t), 0.0) > 0.5:
                            starts = [j for j in params.facilities for n in params.pickup_nodes if raw.get(("x", j, n, k, t), 0.0) > 0.5]
                            feasible_facilities.extend(starts)
                if feasible_facilities and not any(params.technology[j, s] == 1 for j in feasible_facilities):
                    violations.append(f"facility technology mismatch for {node}, t={t}")
            qty = sum(raw.get(("q", node, k, t), 0.0) for k in params.vehicles)
            if qty > tol and visits < 0.5:
                violations.append(f"pickup quantity without visit: {display_node(node)}, t={t}")
    if params.require_terminal_clear:
        last = params.periods[-1]
        for node in params.pickup_nodes:
            if not isclose(raw.get(("IG", node, last), 0.0), 0.0, abs_tol=tol):
                violations.append(f"terminal producer inventory not clear: {display_node(node)}")
        for j in params.facilities:
            for s in params.waste_types:
                if not isclose(raw.get(("ID", j, s, last), 0.0), 0.0, abs_tol=tol):
                    violations.append(f"terminal facility inventory not clear: {j},{s}")
    return violations
