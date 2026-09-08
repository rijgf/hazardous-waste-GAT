from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import isclose
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


Number = float
Node = str
Arc = Tuple[Node, Node]


@dataclass(frozen=True)
class ModelParams:
    producers: List[str]
    waste_types: List[str]
    facilities: List[str]
    vehicles: List[str]
    periods: List[int]
    generation: Dict[Tuple[str, str, int], Number]
    initial_producer_inventory: Dict[Tuple[str, str], Number]
    initial_facility_inventory: Dict[Tuple[str, str], Number]
    producer_capacity: Dict[str, Number]
    facility_capacity: Dict[str, Number]
    vehicle_capacity: Number
    distance: Dict[Arc, Number]
    accident_probability: Dict[Arc, Number]
    vehicle_fixed_cost: Number
    distance_cost: Number
    processing_cost: Dict[Tuple[str, str], Number]
    technology: Dict[Tuple[str, str], int]
    processing_capacity: Dict[Tuple[str, str, int], Number]
    compatibility: Dict[Tuple[str, str], int]
    coload_risk: Dict[Tuple[str, str], Number]
    waste_consequence: Dict[str, Number]
    producer_inventory_risk: Dict[Tuple[str, str], Number]
    facility_inventory_risk: Dict[Tuple[str, str], Number]
    min_pickup: Number = 0.001
    cost_weight: Number = 1.0
    risk_weight: Number = 1.0
    require_terminal_clear: bool = True

    @property
    def pickup_nodes(self) -> List[str]:
        return [pickup_node(i, s) for i in self.producers for s in self.waste_types]

    @property
    def nodes(self) -> List[str]:
        return self.facilities + self.pickup_nodes

    @property
    def arcs(self) -> List[Arc]:
        arcs: List[Arc] = []
        for a in self.nodes:
            for b in self.nodes:
                if a == b:
                    continue
                if a in self.facilities and b in self.facilities:
                    continue
                if a in self.pickup_nodes and b in self.pickup_nodes and pickup_owner(a) == pickup_owner(b):
                    continue
                arcs.append((a, b))
        return arcs


@dataclass
class SolveResult:
    status: str
    objective: Optional[float]
    solution: Dict[str, Any]
    diagnostics: List[str]


def pickup_node(producer: str, waste_type: str) -> str:
    return f"n::{producer}::{waste_type}"


def pickup_owner(node: str) -> str:
    return node.split("::", 2)[1]


def pickup_type(node: str) -> str:
    return node.split("::", 2)[2]


def normalize_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


class HazardousWasteMILP:
    def __init__(self, params: ModelParams):
        self.p = params
        self.vars: Dict[Tuple[Any, ...], int] = {}
        self.names: List[Tuple[Any, ...]] = []
        self.costs: List[float] = []
        self.lbs: List[float] = []
        self.ubs: List[float] = []
        self.integrality: List[int] = []
        self.rows: List[Dict[int, float]] = []
        self.row_lbs: List[float] = []
        self.row_ubs: List[float] = []
        self.diagnostics: List[str] = []

    def solve(self, time_limit: Optional[float] = 30.0) -> SolveResult:
        self._build()
        matrix = lil_matrix((len(self.rows), len(self.names)), dtype=float)
        for r, coeffs in enumerate(self.rows):
            for c, value in coeffs.items():
                matrix[r, c] = value
        options = {"time_limit": time_limit, "mip_rel_gap": 0.0} if time_limit else {"mip_rel_gap": 0.0}
        result = milp(
            c=np.array(self.costs, dtype=float),
            integrality=np.array(self.integrality, dtype=int),
            bounds=Bounds(np.array(self.lbs), np.array(self.ubs)),
            constraints=LinearConstraint(matrix.tocsr(), np.array(self.row_lbs), np.array(self.row_ubs)),
            options=options,
        )
        status = "optimal" if result.success else f"solver_status_{result.status}"
        if result.x is None:
            return SolveResult(status, None, {}, self.diagnostics + [result.message])
        solution = self._decode(result.x)
        checks = check_solution(self.p, solution)
        if checks:
            status = f"{status}_with_check_violations"
        return SolveResult(status, float(result.fun), solution, self.diagnostics + checks)

    def _add_var(self, key: Tuple[Any, ...], lb: float = 0.0, ub: float = np.inf, integer: bool = False, cost: float = 0.0) -> int:
        idx = len(self.names)
        self.vars[key] = idx
        self.names.append(key)
        self.lbs.append(lb)
        self.ubs.append(ub)
        self.integrality.append(1 if integer else 0)
        self.costs.append(cost)
        return idx

    def _v(self, *key: Any) -> int:
        return self.vars[tuple(key)]

    def _row(self, coeffs: Mapping[int, float], lb: float = -np.inf, ub: float = np.inf) -> None:
        self.rows.append(dict(coeffs))
        self.row_lbs.append(lb)
        self.row_ubs.append(ub)

    def _eq(self, coeffs: Mapping[int, float], rhs: float) -> None:
        self._row(coeffs, rhs, rhs)

    def _le(self, coeffs: Mapping[int, float], rhs: float) -> None:
        self._row(coeffs, -np.inf, rhs)

    def _ge(self, coeffs: Mapping[int, float], rhs: float) -> None:
        self._row(coeffs, rhs, np.inf)

    def _build(self) -> None:
        p = self.p
        arcs = p.arcs
        nodes = p.pickup_nodes
        q_cap = p.vehicle_capacity
        load_m = 2.0 * q_cap
        max_prod_cap = max(p.producer_capacity.values())
        facility_m = {j: max(p.facility_capacity[j], max(p.processing_capacity[j, s, t] for s in p.waste_types for t in p.periods)) for j in p.facilities}

        for t in p.periods:
            for k in p.vehicles:
                self._add_var(("e", k, t), 0, 1, True, p.cost_weight * p.vehicle_fixed_cost)
                for a, b in arcs:
                    self._add_var(("x", a, b, k, t), 0, 1, True, p.cost_weight * p.distance_cost * p.distance[a, b])
                    for s in p.waste_types:
                        risk_cost = p.risk_weight * p.accident_probability[a, b] * p.distance[a, b] * p.waste_consequence[s]
                        self._add_var(("F", a, b, s, k, t), 0, q_cap, False, risk_cost)
                        self._add_var(("H", a, b, s, k, t), 0, 1, True)
                    for s1, s2 in combinations(p.waste_types, 2):
                        gamma = p.coload_risk.get(normalize_pair(s1, s2), 0.0)
                        risk_cost = p.risk_weight * p.accident_probability[a, b] * p.distance[a, b] * gamma
                        self._add_var(("W", a, b, s1, s2, k, t), 0, 1, True)
                        self._add_var(("G", a, b, s1, s2, k, t), 0, 2 * q_cap, False, risk_cost)
                for n in nodes:
                    self._add_var(("q", n, k, t), 0, max_prod_cap)
                    for s in p.waste_types:
                        self._add_var(("l", n, s, k, t), 0, q_cap)
            for n in nodes:
                i, s = pickup_owner(n), pickup_type(n)
                self._add_var(("BG", n, t), 0, p.producer_capacity[i])
                self._add_var(("IG", n, t), 0, p.producer_capacity[i], False, p.risk_weight * p.producer_inventory_risk[i, s])
            for j in p.facilities:
                for s in p.waste_types:
                    self._add_var(("R", j, s, t), 0, p.facility_capacity[j])
                    self._add_var(("BD", j, s, t), 0, p.facility_capacity[j])
                    self._add_var(("p", j, s, t), 0, p.facility_capacity[j], False, p.cost_weight * p.processing_cost[j, s])
                    self._add_var(("ID", j, s, t), 0, p.facility_capacity[j], False, p.risk_weight * p.facility_inventory_risk[j, s])
                    self._add_var(("omega", j, s, t), 0, 1, True)

        self._constraints(arcs, nodes, q_cap, load_m, max_prod_cap, facility_m)

    def _constraints(self, arcs: List[Arc], nodes: List[str], q_cap: float, load_m: float, max_prod_cap: float, facility_m: Dict[str, float]) -> None:
        p = self.p
        incoming = {n: [(a, n) for a, b in arcs if b == n] for n in p.nodes}
        outgoing = {n: [(n, b) for a, b in arcs if a == n] for n in p.nodes}

        for t in p.periods:
            for k in p.vehicles:
                start_coeffs = {self._v("x", j, n, k, t): 1.0 for j in p.facilities for n in nodes if (j, n) in arcs}
                start_coeffs[self._v("e", k, t)] = -1.0
                self._eq(start_coeffs, 0.0)
                for n in nodes:
                    self._eq(
                        {**{self._v("x", a, b, k, t): 1.0 for a, b in incoming[n]},
                         **{self._v("x", a, b, k, t): -1.0 for a, b in outgoing[n]}},
                        0.0,
                    )
                for j in p.facilities:
                    coeffs: Dict[int, float] = {}
                    for n in nodes:
                        if (n, j) in arcs:
                            coeffs[self._v("x", n, j, k, t)] = coeffs.get(self._v("x", n, j, k, t), 0.0) + 1.0
                        if (j, n) in arcs:
                            coeffs[self._v("x", j, n, k, t)] = coeffs.get(self._v("x", j, n, k, t), 0.0) - 1.0
                    self._eq(coeffs, 0.0)
                self._le({self._v("q", n, k, t): 1.0 for n in nodes} | {self._v("e", k, t): -q_cap}, 0.0)

                for n in nodes:
                    visit = {self._v("x", a, b, k, t): 1.0 for a, b in incoming[n]}
                    self._le({self._v("q", n, k, t): 1.0, **{idx: -max_prod_cap * c for idx, c in visit.items()}}, 0.0)
                    self._ge({self._v("q", n, k, t): 1.0, **{idx: -p.min_pickup * c for idx, c in visit.items()}}, 0.0)
                    self._ge({self._v("q", n, k, t): 1.0, self._v("BG", n, t): -1.0, **{idx: -max_prod_cap * c for idx, c in visit.items()}}, -max_prod_cap)
                    self._le({self._v("l", n, s, k, t): 1.0 for s in p.waste_types} | {idx: -q_cap * c for idx, c in visit.items()}, 0.0)

                for j in p.facilities:
                    for n in nodes:
                        if (j, n) not in arcs:
                            continue
                        ns = pickup_type(n)
                        for s in p.waste_types:
                            alpha = 1.0 if s == ns else 0.0
                            self._le({self._v("l", n, s, k, t): 1.0, self._v("q", n, k, t): -alpha, self._v("x", j, n, k, t): load_m}, load_m)
                            self._le({self._v("l", n, s, k, t): -1.0, self._v("q", n, k, t): alpha, self._v("x", j, n, k, t): load_m}, load_m)
                for n in nodes:
                    for m in nodes:
                        if n == m or (n, m) not in arcs:
                            continue
                        ms = pickup_type(m)
                        for s in p.waste_types:
                            alpha = 1.0 if s == ms else 0.0
                            self._le({self._v("l", m, s, k, t): 1.0, self._v("l", n, s, k, t): -1.0, self._v("q", m, k, t): -alpha, self._v("x", n, m, k, t): load_m}, load_m)
                            self._le({self._v("l", n, s, k, t): 1.0, self._v("q", m, k, t): alpha, self._v("l", m, s, k, t): -1.0, self._v("x", n, m, k, t): load_m}, load_m)

                for a, b in arcs:
                    for s in p.waste_types:
                        if a in p.facilities:
                            self._eq({self._v("F", a, b, s, k, t): 1.0}, 0.0)
                        else:
                            self._le({self._v("F", a, b, s, k, t): 1.0, self._v("x", a, b, k, t): -q_cap}, 0.0)
                            self._le({self._v("F", a, b, s, k, t): 1.0, self._v("l", a, s, k, t): -1.0}, 0.0)
                            self._ge({self._v("F", a, b, s, k, t): 1.0, self._v("l", a, s, k, t): -1.0, self._v("x", a, b, k, t): -q_cap}, -q_cap)
                        self._le({self._v("F", a, b, s, k, t): 1.0, self._v("H", a, b, s, k, t): -q_cap}, 0.0)
                        self._ge({self._v("F", a, b, s, k, t): 1.0, self._v("H", a, b, s, k, t): -p.min_pickup}, 0.0)
                        self._le({self._v("H", a, b, s, k, t): 1.0, self._v("x", a, b, k, t): -1.0}, 0.0)
                    for s1, s2 in combinations(p.waste_types, 2):
                        self._le({self._v("W", a, b, s1, s2, k, t): 1.0, self._v("H", a, b, s1, k, t): -1.0}, 0.0)
                        self._le({self._v("W", a, b, s1, s2, k, t): 1.0, self._v("H", a, b, s2, k, t): -1.0}, 0.0)
                        self._ge({self._v("W", a, b, s1, s2, k, t): 1.0, self._v("H", a, b, s1, k, t): -1.0, self._v("H", a, b, s2, k, t): -1.0}, -1.0)
                        self._le({self._v("G", a, b, s1, s2, k, t): 1.0, self._v("F", a, b, s1, k, t): -1.0, self._v("F", a, b, s2, k, t): -1.0}, 0.0)
                        self._le({self._v("G", a, b, s1, s2, k, t): 1.0, self._v("W", a, b, s1, s2, k, t): -2.0 * q_cap}, 0.0)
                        self._ge({self._v("G", a, b, s1, s2, k, t): 1.0, self._v("F", a, b, s1, k, t): -1.0, self._v("F", a, b, s2, k, t): -1.0, self._v("W", a, b, s1, s2, k, t): -2.0 * q_cap}, -2.0 * q_cap)

                for j in p.facilities:
                    start = {self._v("x", j, m, k, t): 1.0 for m in nodes if (j, m) in arcs}
                    for n in nodes:
                        visit = {self._v("x", a, n, k, t): 1.0 for a, _ in incoming[n]}
                        self._le({**start, **visit}, 1.0 + p.technology[j, pickup_type(n)])
                for n, m in combinations(nodes, 2):
                    visit_n = {self._v("x", a, n, k, t): 1.0 for a, _ in incoming[n]}
                    visit_m = {self._v("x", a, m, k, t): 1.0 for a, _ in incoming[m]}
                    chi = p.compatibility[normalize_pair(pickup_type(n), pickup_type(m))]
                    self._le({**visit_n, **visit_m}, 1.0 + chi)

            for n in nodes:
                self._le({self._v("x", a, n, k, t): 1.0 for k in p.vehicles for a, _ in incoming[n]}, 1.0)

            for n in nodes:
                i, s = pickup_owner(n), pickup_type(n)
                prev = p.initial_producer_inventory[i, s] if t == p.periods[0] else None
                if prev is None:
                    prev_coeff = {self._v("IG", n, p.periods[p.periods.index(t) - 1]): -1.0}
                    self._eq({self._v("BG", n, t): 1.0, **prev_coeff}, p.generation[i, s, t])
                else:
                    self._eq({self._v("BG", n, t): 1.0}, prev + p.generation[i, s, t])
                self._eq({self._v("IG", n, t): 1.0, self._v("BG", n, t): -1.0, **{self._v("q", n, k, t): 1.0 for k in p.vehicles}}, 0.0)

            for i in p.producers:
                self._le({self._v("BG", pickup_node(i, s), t): 1.0 for s in p.waste_types}, p.producer_capacity[i])
                self._le({self._v("IG", pickup_node(i, s), t): 1.0 for s in p.waste_types}, p.producer_capacity[i])

            for j in p.facilities:
                for s in p.waste_types:
                    self._eq({self._v("R", j, s, t): 1.0, **{self._v("F", n, j, s, k, t): -1.0 for n in nodes if (n, j) in arcs for k in p.vehicles}}, 0.0)
                    if t == p.periods[0]:
                        self._eq({self._v("BD", j, s, t): 1.0, self._v("R", j, s, t): -1.0}, p.initial_facility_inventory[j, s])
                    else:
                        prev_t = p.periods[p.periods.index(t) - 1]
                        self._eq({self._v("BD", j, s, t): 1.0, self._v("ID", j, s, prev_t): -1.0, self._v("R", j, s, t): -1.0}, 0.0)
                    cap = p.processing_capacity[j, s, t] * p.technology[j, s]
                    self._le({self._v("p", j, s, t): 1.0, self._v("BD", j, s, t): -1.0}, 0.0)
                    self._le({self._v("p", j, s, t): 1.0}, cap)
                    self._ge({self._v("p", j, s, t): 1.0, self._v("BD", j, s, t): -1.0, self._v("omega", j, s, t): -facility_m[j]}, -facility_m[j])
                    self._ge({self._v("p", j, s, t): 1.0, self._v("omega", j, s, t): facility_m[j]}, cap)
                    self._eq({self._v("ID", j, s, t): 1.0, self._v("BD", j, s, t): -1.0, self._v("p", j, s, t): 1.0}, 0.0)
                self._le({self._v("BD", j, s, t): 1.0 for s in p.waste_types}, p.facility_capacity[j])
                self._le({self._v("ID", j, s, t): 1.0 for s in p.waste_types}, p.facility_capacity[j])

        if p.require_terminal_clear:
            last = p.periods[-1]
            for n in nodes:
                self._eq({self._v("IG", n, last): 1.0}, 0.0)
            for j in p.facilities:
                for s in p.waste_types:
                    self._eq({self._v("ID", j, s, last): 1.0}, 0.0)

    def _decode(self, x: np.ndarray) -> Dict[str, Any]:
        raw = {self.names[i]: float(x[i]) for i in range(len(self.names)) if abs(x[i]) > 1e-7}
        routes: Dict[Tuple[str, int], List[str]] = {}
        plan: Dict[Tuple[str, int], List[str]] = {}
        for t in self.p.periods:
            for k in self.p.vehicles:
                used = {(a, b): raw.get(("x", a, b, k, t), 0.0) for a, b in self.p.arcs if raw.get(("x", a, b, k, t), 0.0) > 0.5}
                if not used:
                    continue
                start = next((a, b) for a, b in used if a in self.p.facilities)
                path = [start[0], start[1]]
                while path[-1] not in self.p.facilities:
                    nxt = next((b for a, b in used if a == path[-1]), None)
                    if nxt is None:
                        break
                    if nxt in path and nxt not in self.p.facilities:
                        break
                    path.append(nxt)
                plan[k, t] = path
                routes[k, t] = [display_node(n) for n in path]
        return {
            "raw": raw,
            "routes": routes,
            "summary": summarize_solution(self.p, raw),
            "plan": plan,
        }


def display_node(node: str) -> str:
    return f"{pickup_owner(node)}:{pickup_type(node)}" if node.startswith("n::") else node


def summarize_solution(params: ModelParams, raw: Mapping[Tuple[Any, ...], float]) -> Dict[str, Any]:
    pickup = []
    processing = []
    inventory = []
    for t in params.periods:
        for n in params.pickup_nodes:
            qty = sum(raw.get(("q", n, k, t), 0.0) for k in params.vehicles)
            if qty > 1e-6:
                pickup.append({"period": t, "node": display_node(n), "quantity": round(qty, 6)})
            rem = raw.get(("IG", n, t), 0.0)
            if rem > 1e-6:
                inventory.append({"period": t, "node": display_node(n), "producer_inventory": round(rem, 6)})
        for j in params.facilities:
            for s in params.waste_types:
                qty = raw.get(("p", j, s, t), 0.0)
                if qty > 1e-6:
                    processing.append({"period": t, "facility": j, "waste_type": s, "quantity": round(qty, 6)})
                rem = raw.get(("ID", j, s, t), 0.0)
                if rem > 1e-6:
                    inventory.append({"period": t, "facility": j, "waste_type": s, "facility_inventory": round(rem, 6)})
    return {"pickup": pickup, "processing": processing, "ending_inventory": inventory}


def solve_with_milp(params: ModelParams, time_limit: Optional[float] = 30.0) -> SolveResult:
    return HazardousWasteMILP(params).solve(time_limit=time_limit)


def check_solution(params: ModelParams, solution: Mapping[str, Any], tol: float = 1e-5) -> List[str]:
    raw = solution.get("raw", solution)
    violations: List[str] = []
    # Build visit counts from the sparse chosen arcs. Iterating the full arc set
    # for every node made large-instance validation dominate training time.
    incoming_value = {
        (n, k, t): 0.0
        for n in params.nodes
        for k in params.vehicles
        for t in params.periods
    }
    outgoing_value = {
        (n, k, t): 0.0
        for n in params.nodes
        for k in params.vehicles
        for t in params.periods
    }
    used_arcs: Dict[Tuple[str, int], List[Arc]] = {}
    for key, value in raw.items():
        if key[0] != "x" or abs(value) <= tol:
            continue
        _, a, b, vehicle, period = key
        if (a, b) not in params.distance:
            violations.append(f"route uses forbidden arc: {a}->{b}, k={vehicle}, t={period}")
            continue
        if vehicle not in params.vehicles or period not in params.periods:
            violations.append(f"route uses unknown vehicle/period: k={vehicle}, t={period}")
            continue
        if not isclose(value, 1.0, abs_tol=tol):
            violations.append(f"route arc must be binary: {a}->{b}, k={vehicle}, t={period}, x={value}")
        incoming_value[b, vehicle, period] = incoming_value.get((b, vehicle, period), 0.0) + value
        outgoing_value[a, vehicle, period] = outgoing_value.get((a, vehicle, period), 0.0) + value
        used_arcs.setdefault((vehicle, period), []).append((a, b))

    flow_in: Dict[Tuple[str, str, str, int], float] = {}
    flow_out: Dict[Tuple[str, str, str, int], float] = {}
    for key, value in raw.items():
        if key[0] != "F" or abs(value) <= tol:
            continue
        _, a, b, waste, vehicle, period = key
        if (
            (a, b) not in params.distance
            or waste not in params.waste_types
            or vehicle not in params.vehicles
            or period not in params.periods
        ):
            violations.append(
                f"load flow uses unknown arc/type/vehicle/period: {key}"
            )
            continue
        if raw.get(("x", a, b, vehicle, period), 0.0) < 0.5:
            violations.append(
                f"load flow without route arc: {a}->{b}, waste={waste}, "
                f"k={vehicle}, t={period}"
            )
        if a in params.facilities:
            violations.append(
                f"vehicle leaves facility with waste load: facility={a}, "
                f"waste={waste}, k={vehicle}, t={period}"
            )
        flow_out[a, waste, vehicle, period] = (
            flow_out.get((a, waste, vehicle, period), 0.0) + value
        )
        flow_in[b, waste, vehicle, period] = (
            flow_in.get((b, waste, vehicle, period), 0.0) + value
        )
    for (j, s), can_process in params.technology.items():
        if can_process not in (0, 1):
            violations.append(f"technology[{j},{s}] must be 0 or 1")
    for pair, chi in params.compatibility.items():
        if chi not in (0, 1):
            violations.append(f"compatibility{pair} must be 0 or 1")
    for t in params.periods:
        for k in params.vehicles:
            for node in params.nodes:
                incoming = incoming_value[node, k, t]
                outgoing = outgoing_value[node, k, t]
                if not isclose(incoming, outgoing, abs_tol=tol):
                    violations.append(
                        f"route flow conservation violated: node={display_node(node)}, "
                        f"k={k}, t={t}, incoming={incoming}, outgoing={outgoing}"
                    )
            departures = sum(
                outgoing_value[facility, k, t]
                for facility in params.facilities
            )
            if departures > 1.0 + tol:
                violations.append(
                    f"vehicle starts more than one route: k={k}, t={t}, departures={departures}"
                )
            arcs = used_arcs.get((k, t), [])
            if arcs:
                adjacency: Dict[str, List[str]] = {}
                route_nodes = set()
                for a, b in arcs:
                    adjacency.setdefault(a, []).append(b)
                    route_nodes.update((a, b))
                frontier = [
                    facility
                    for facility in params.facilities
                    if outgoing_value[facility, k, t] > tol
                ]
                reachable = set(frontier)
                while frontier:
                    node = frontier.pop()
                    for successor in adjacency.get(node, []):
                        if successor not in reachable:
                            reachable.add(successor)
                            frontier.append(successor)
                disconnected = sorted(route_nodes - reachable)
                if disconnected:
                    violations.append(
                        f"route contains disconnected nodes: k={k}, t={t}, "
                        f"nodes={[display_node(node) for node in disconnected]}"
                    )
            load = sum(raw.get(("q", n, k, t), 0.0) for n in params.pickup_nodes)
            if load > params.vehicle_capacity + tol:
                violations.append(f"vehicle capacity exceeded: k={k}, t={t}, load={load}")
            served_types = [
                pickup_type(n)
                for n in params.pickup_nodes
                if incoming_value[n, k, t] > 0.5
            ]
            for s1, s2 in combinations(served_types, 2):
                if params.compatibility[normalize_pair(s1, s2)] == 0:
                    violations.append(f"incompatible coload on k={k}, t={t}: {s1}, {s2}")
            for n in params.pickup_nodes:
                node_waste = pickup_type(n)
                pickup_quantity = raw.get(("q", n, k, t), 0.0)
                for waste in params.waste_types:
                    incoming_load = flow_in.get((n, waste, k, t), 0.0)
                    outgoing_load = flow_out.get((n, waste, k, t), 0.0)
                    expected_increase = pickup_quantity if waste == node_waste else 0.0
                    if not isclose(
                        outgoing_load - incoming_load,
                        expected_increase,
                        abs_tol=tol,
                    ):
                        violations.append(
                            f"waste load conservation violated: node={display_node(n)}, "
                            f"waste={waste}, k={k}, t={t}"
                        )
        for n in params.pickup_nodes:
            visits = sum(incoming_value[n, k, t] for k in params.vehicles)
            qty = sum(raw.get(("q", n, k, t), 0.0) for k in params.vehicles)
            bg = raw.get(("BG", n, t), 0.0)
            if visits > 1.0 + tol:
                violations.append(f"pickup node served more than once: {display_node(n)}, t={t}")
            if qty > tol and visits < 0.5:
                violations.append(f"pickup quantity without visit: {display_node(n)}, t={t}")
            if visits > 0.5 and not isclose(qty, bg, abs_tol=tol):
                violations.append(f"visited node not fully collected: {display_node(n)}, t={t}, q={qty}, BG={bg}")
        for i in params.producers:
            inv = sum(raw.get(("IG", pickup_node(i, s), t), 0.0) for s in params.waste_types)
            if inv > params.producer_capacity[i] + tol:
                violations.append(f"producer inventory capacity exceeded: {i}, t={t}")
        for j in params.facilities:
            inv = sum(raw.get(("ID", j, s, t), 0.0) for s in params.waste_types)
            if inv > params.facility_capacity[j] + tol:
                violations.append(f"facility inventory capacity exceeded: {j}, t={t}")
            for s in params.waste_types:
                received_flow = sum(
                    flow_in.get((j, s, k, t), 0.0)
                    for k in params.vehicles
                )
                recorded_received = raw.get(("R", j, s, t), 0.0)
                if not isclose(received_flow, recorded_received, abs_tol=tol):
                    violations.append(
                        f"facility receipt flow mismatch: facility={j}, waste={s}, "
                        f"t={t}, flow={received_flow}, R={recorded_received}"
                    )
                bd = raw.get(("BD", j, s, t), 0.0)
                processed = raw.get(("p", j, s, t), 0.0)
                cap = params.processing_capacity[j, s, t] * params.technology[j, s]
                if processed > cap + tol:
                    violations.append(f"processing capacity exceeded: facility={j}, waste={s}, t={t}")
                if processed > bd + tol:
                    violations.append(f"processed more than available: facility={j}, waste={s}, t={t}")
    if params.require_terminal_clear:
        last = params.periods[-1]
        for n in params.pickup_nodes:
            if abs(raw.get(("IG", n, last), 0.0)) > tol:
                violations.append(f"terminal producer inventory not clear: {display_node(n)}")
        for j in params.facilities:
            for s in params.waste_types:
                if abs(raw.get(("ID", j, s, last), 0.0)) > tol:
                    violations.append(f"terminal facility inventory not clear: {j},{s}")
    return violations


class HazardousWasteHeuristicEnv:
    """Small RL/heuristic adapter around the same feasibility vocabulary.

    State is a dictionary of remaining producer inventory and facility inventory.
    An action is {"period": t, "vehicle": k, "facility": j, "route": [pickup_node, ...]}.
    """

    def __init__(self, params: ModelParams):
        self.params = params

    def initial_state(self) -> Dict[str, Any]:
        return {
            "producer_inventory": dict(self.params.initial_producer_inventory),
            "facility_inventory": dict(self.params.initial_facility_inventory),
            "period": self.params.periods[0],
        }

    def legal_pickups(self, state: Mapping[str, Any], facility: str, current_load: float = 0.0) -> List[str]:
        legal: List[str] = []
        inv = state["producer_inventory"]
        for i in self.params.producers:
            for s in self.params.waste_types:
                qty = inv.get((i, s), 0.0)
                if qty >= self.params.min_pickup and current_load + qty <= self.params.vehicle_capacity and self.params.technology[facility, s] == 1:
                    legal.append(pickup_node(i, s))
        return legal

    def action_to_solution_skeleton(self, actions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return {"actions": list(actions)}

    def reward(self, solution: Mapping[str, Any]) -> float:
        violations = check_solution(self.params, solution)
        if violations:
            return -1e6 - 1000.0 * len(violations)
        raw = solution.get("raw", {})
        total_distance = 0.0
        for key, value in raw.items():
            if len(key) >= 6 and key[0] == "x" and value > 0.5:
                total_distance += self.params.distance[key[1], key[2]]
        return -total_distance
