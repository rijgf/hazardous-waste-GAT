from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isclose, isfinite
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
    violations = validate_solution(params, solution) if check_constraints else []
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


def validate_solution(params: ModelParams, solution: Mapping[str, Any], tol: float = 1e-5) -> List[str]:
    """Validate a complete solution, including the conservation equations.

    The legacy checker focused on upper bounds and treated every missing value as
    zero. This validator rejects an empty raw solution and checks producer and
    facility mass balance. Sparse zero variables emitted by the MILP remain valid.
    """

    raw = solution.get("raw", solution)
    if not raw:
        return ["missing raw solution variables"]

    violations = list(check_solution(params, solution, tol=tol))
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            if not isfinite(float(value)):
                violations.append(f"non-finite solution value: {key}")
            elif float(value) < -tol:
                violations.append(f"negative solution value: {key}={value}")

    previous_period: int | None = None
    for t in params.periods:
        # The MILP bounds both before-service and ending inventories. Checking
        # only IG/ID misses overflow that is cleared within the same period.
        for producer in params.producers:
            before = sum(raw.get(("BG", pickup_node(producer, waste), t), 0.0)
                         for waste in params.waste_types)
            if before > params.producer_capacity[producer] + tol:
                violations.append(f"producer before-pickup capacity exceeded: {producer}, t={t}")
        for facility in params.facilities:
            before = sum(raw.get(("BD", facility, waste, t), 0.0)
                         for waste in params.waste_types)
            if before > params.facility_capacity[facility] + tol:
                violations.append(f"facility before-processing capacity exceeded: {facility}, t={t}")
        for node in params.pickup_nodes:
            owner = pickup_owner(node)
            waste = pickup_type(node)
            previous_inventory = (
                params.initial_producer_inventory[owner, waste]
                if previous_period is None
                else raw.get(("IG", node, previous_period), 0.0)
            )
            expected_before_pickup = previous_inventory + params.generation[owner, waste, t]
            before_pickup = raw.get(("BG", node, t), 0.0)
            collected = sum(raw.get(("q", node, vehicle, t), 0.0) for vehicle in params.vehicles)
            ending = raw.get(("IG", node, t), 0.0)
            if not isclose(before_pickup, expected_before_pickup, abs_tol=tol):
                violations.append(
                    f"producer inventory balance before pickup: {display_node(node)}, t={t}, "
                    f"BG={before_pickup}, expected={expected_before_pickup}"
                )
            if not isclose(before_pickup, collected + ending, abs_tol=tol):
                violations.append(
                    f"producer inventory balance after pickup: {display_node(node)}, t={t}, "
                    f"BG={before_pickup}, q+IG={collected + ending}"
                )

        for j in params.facilities:
            for s in params.waste_types:
                previous_inventory = (
                    params.initial_facility_inventory[j, s]
                    if previous_period is None
                    else raw.get(("ID", j, s, previous_period), 0.0)
                )
                received = raw.get(("R", j, s, t), 0.0)
                bd = raw.get(("BD", j, s, t), 0.0)
                processed = raw.get(("p", j, s, t), 0.0)
                ending = raw.get(("ID", j, s, t), 0.0)
                cap = params.processing_capacity[j, s, t] * params.technology[j, s]
                if not isclose(bd, previous_inventory + received, abs_tol=tol):
                    violations.append(
                        f"facility inventory balance before processing: facility={j}, waste={s}, t={t}"
                    )
                if not isclose(bd, processed + ending, abs_tol=tol):
                    violations.append(
                        f"facility inventory balance after processing: facility={j}, waste={s}, t={t}"
                    )
                if processed > cap + tol:
                    violations.append(f"processing capacity exceeded: facility={j}, waste={s}, t={t}")
                if processed > bd + tol:
                    violations.append(f"processed more than available: facility={j}, waste={s}, t={t}")
                if received > tol and params.technology[j, s] != 1:
                    violations.append(f"facility technology mismatch: facility={j}, waste={s}, t={t}")

        for waste in params.waste_types:
            collected = sum(
                raw.get(("q", node, vehicle, t), 0.0)
                for node in params.pickup_nodes
                if pickup_type(node) == waste
                for vehicle in params.vehicles
            )
            received = sum(raw.get(("R", facility, waste, t), 0.0) for facility in params.facilities)
            if not isclose(collected, received, abs_tol=tol):
                violations.append(
                    f"pickup-to-facility flow balance: waste={waste}, t={t}, q={collected}, R={received}"
                )

        previous_period = t

    return violations


def enhanced_check_solution(params: ModelParams, solution: Mapping[str, Any], tol: float = 1e-5) -> List[str]:
    """Backward-compatible name for the complete validator."""

    return validate_solution(params, solution, tol=tol)
