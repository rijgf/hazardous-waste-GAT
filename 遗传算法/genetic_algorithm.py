from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hazardous_waste_model import (
    ModelParams,
    check_solution,
    display_node,
    normalize_pair,
    pickup_owner,
    pickup_type,
    summarize_solution,
)
from src.solution_utils import evaluate_solution


Chromosome = List[str]


@dataclass
class GAConfig:
    population_size: int = 60
    generations: int = 120
    crossover_rate: float = 0.85
    mutation_rate: float = 0.2
    elite_size: int = 2
    tournament_size: int = 3
    random_seed: int = 42


@dataclass
class GAResult:
    best_objective: float
    best_chromosome: Chromosome
    best_solution: Dict[str, Any]
    feasible: bool
    history: List[float]


class ClassicGeneticAlgorithm:
    """Classic permutation GA for the hazardous-waste routing subproblem.

    The chromosome is a permutation of pickup nodes. The decoder postpones
    collection to the last period, then splits the permutation into vehicle
    routes while checking capacity, waste compatibility, disposal technology,
    and allowed arcs.
    """

    def __init__(self, params: ModelParams, config: Optional[GAConfig] = None):
        self.params = params
        self.config = config or GAConfig()
        self.rng = random.Random(self.config.random_seed)
        self.last_period = params.periods[-1]
        self.facility = params.facilities[0]

    def run(self) -> GAResult:
        population = self._initial_population()
        history: List[float] = []
        best_chromosome = population[0]
        best_objective, best_solution = self.evaluate(best_chromosome)

        for _ in range(self.config.generations):
            scored = [(self.evaluate(ch)[0], ch) for ch in population]
            scored.sort(key=lambda item: item[0])
            if scored[0][0] < best_objective:
                best_objective = scored[0][0]
                best_chromosome = scored[0][1][:]
                best_solution = self.evaluate(best_chromosome)[1]
            history.append(best_objective)

            new_population = [ch[:] for _, ch in scored[: self.config.elite_size]]
            while len(new_population) < self.config.population_size:
                p1 = self._tournament(scored)
                p2 = self._tournament(scored)
                if self.rng.random() < self.config.crossover_rate:
                    c1, c2 = self._order_crossover(p1, p2)
                else:
                    c1, c2 = p1[:], p2[:]
                self._mutate(c1)
                self._mutate(c2)
                new_population.append(c1)
                if len(new_population) < self.config.population_size:
                    new_population.append(c2)
            population = new_population

        feasible = not self._fast_search_violations(best_solution)
        return GAResult(best_objective, best_chromosome, best_solution, feasible, history)

    def evaluate(self, chromosome: Chromosome) -> Tuple[float, Dict[str, Any]]:
        solution = self.decode(chromosome)
        violations = self._fast_search_violations(solution)
        objective = evaluate_solution(self.params, solution, (self.params.cost_weight, self.params.risk_weight), check_constraints=False)["weighted_objective"]
        if violations:
            objective += 1_000_000.0 + 10_000.0 * len(violations)
        return objective, solution

    def _fast_search_violations(self, solution: Mapping[str, Any]) -> List[str]:
        raw = solution.get("raw", solution)
        violations: List[str] = []
        last = self.params.periods[-1]
        for node in self.params.pickup_nodes:
            if raw.get(("IG", node, last), 0.0) > 1e-6:
                violations.append(f"terminal inventory remains: {display_node(node)}")
        for t in self.params.periods:
            for k in self.params.vehicles:
                load = sum(raw.get(("q", n, k, t), 0.0) for n in self.params.pickup_nodes)
                if load > self.params.vehicle_capacity + 1e-6:
                    violations.append(f"vehicle capacity exceeded: {k},{t}")
        return violations

    def decode(self, chromosome: Chromosome) -> Dict[str, Any]:
        raw: Dict[Tuple[Any, ...], float] = {}
        routes = self._split_routes(chromosome)

        self._fill_inventory_and_processing(raw, routes)
        for vehicle, route in routes:
            if len(route) < 3:
                continue
            for a, b in zip(route, route[1:]):
                raw[("x", a, b, vehicle, self.last_period)] = 1.0
            for node in route[1:-1]:
                qty = raw.get(("BG", node, self.last_period), 0.0)
                raw[("q", node, vehicle, self.last_period)] = qty

            load_by_type = {s: 0.0 for s in self.params.waste_types}
            for a, b in zip(route, route[1:]):
                if a.startswith("n::"):
                    load_by_type[pickup_type(a)] += raw.get(("q", a, vehicle, self.last_period), 0.0)
                for s, load in load_by_type.items():
                    if load > 1e-9:
                        raw[("F", a, b, s, vehicle, self.last_period)] = load

        solution = {
            "raw": raw,
            "routes": {
                (vehicle, self.last_period): [display_node(node) for node in route]
                for vehicle, route in routes
                if len(route) >= 3
            },
        }
        solution["summary"] = summarize_solution(self.params, raw)
        return solution

    def _split_routes(self, chromosome: Chromosome) -> List[Tuple[str, List[str]]]:
        routes: List[Tuple[str, List[str]]] = []
        vehicle_index = 0
        current = [self.facility]
        load = 0.0
        types: List[str] = []

        for node in chromosome:
            qty = self._terminal_available_inventory(node)
            if qty <= 1e-9:
                continue
            if vehicle_index >= len(self.params.vehicles):
                routes.append(("__infeasible_extra_vehicle__", [self.facility, node, self.facility]))
                continue
            if not self._can_append(current, node, load, qty, types):
                current.append(self.facility)
                routes.append((self.params.vehicles[vehicle_index], current))
                vehicle_index += 1
                current = [self.facility]
                load = 0.0
                types = []
            if vehicle_index >= len(self.params.vehicles) or not self._can_append(current, node, load, qty, types):
                routes.append(("__infeasible_extra_vehicle__", [self.facility, node, self.facility]))
                continue
            current.append(node)
            load += qty
            types.append(pickup_type(node))

        if vehicle_index < len(self.params.vehicles) and len(current) > 1:
            current.append(self.facility)
            routes.append((self.params.vehicles[vehicle_index], current))
        return routes

    def _can_append(self, route: Sequence[str], node: str, load: float, qty: float, types: Sequence[str]) -> bool:
        if (route[-1], node) not in self.params.arcs or (node, self.facility) not in self.params.arcs:
            return False
        if load + qty > self.params.vehicle_capacity + 1e-9:
            return False
        node_type = pickup_type(node)
        if self.params.technology[self.facility, node_type] != 1:
            return False
        for s in types:
            if self.params.compatibility[normalize_pair(s, node_type)] == 0:
                return False
        return True

    def _terminal_available_inventory(self, node: str) -> float:
        i = pickup_owner(node)
        s = pickup_type(node)
        return self.params.initial_producer_inventory[i, s] + sum(
            self.params.generation[i, s, t] for t in self.params.periods
        )

    def _fill_inventory_and_processing(self, raw: Dict[Tuple[Any, ...], float], routes: Sequence[Tuple[str, List[str]]]) -> None:
        collected = {node: 0.0 for node in self.params.pickup_nodes}
        for vehicle, route in routes:
            if vehicle not in self.params.vehicles:
                continue
            for node in route[1:-1]:
                collected[node] = self._terminal_available_inventory(node)

        for t in self.params.periods:
            for node in self.params.pickup_nodes:
                i = pickup_owner(node)
                s = pickup_type(node)
                prev = self.params.initial_producer_inventory[i, s] if t == self.params.periods[0] else raw[("IG", node, self.params.periods[self.params.periods.index(t) - 1])]
                bg = prev + self.params.generation[i, s, t]
                raw[("BG", node, t)] = bg
                taken = collected[node] if t == self.last_period else 0.0
                raw[("IG", node, t)] = max(0.0, bg - taken)

            for j in self.params.facilities:
                for s in self.params.waste_types:
                    received = sum(
                        collected[node]
                        for node in self.params.pickup_nodes
                        if pickup_type(node) == s and any(route[-1] == j and node in route for vehicle, route in routes if vehicle in self.params.vehicles)
                    ) if t == self.last_period else 0.0
                    raw[("R", j, s, t)] = received
                    prev_id = self.params.initial_facility_inventory[j, s] if t == self.params.periods[0] else raw[("ID", j, s, self.params.periods[self.params.periods.index(t) - 1])]
                    bd = prev_id + received
                    cap = self.params.processing_capacity[j, s, t] * self.params.technology[j, s]
                    processed = min(bd, cap)
                    raw[("BD", j, s, t)] = bd
                    raw[("p", j, s, t)] = processed
                    raw[("ID", j, s, t)] = bd - processed

    def _objective(self, raw: Mapping[Tuple[Any, ...], float]) -> float:
        distance_cost = 0.0
        fixed_cost = 0.0
        processing_cost = 0.0
        inventory_risk = 0.0

        used_vehicles = set()
        for key, value in raw.items():
            if value <= 1e-9:
                continue
            if key[0] == "x":
                _, a, b, vehicle, period = key
                distance_cost += self.params.distance[a, b] * self.params.distance_cost
                used_vehicles.add((vehicle, period))
            elif key[0] == "p":
                _, j, s, _ = key
                processing_cost += self.params.processing_cost[j, s] * value
            elif key[0] == "IG":
                _, node, _ = key
                inventory_risk += self.params.producer_inventory_risk[pickup_owner(node), pickup_type(node)] * value
            elif key[0] == "ID":
                _, j, s, _ = key
                inventory_risk += self.params.facility_inventory_risk[j, s] * value
        fixed_cost = len(used_vehicles) * self.params.vehicle_fixed_cost
        return distance_cost + fixed_cost + processing_cost + inventory_risk

    def _initial_population(self) -> List[Chromosome]:
        base = self.params.pickup_nodes
        population = []
        for _ in range(self.config.population_size):
            ch = base[:]
            self.rng.shuffle(ch)
            population.append(ch)
        return population

    def _tournament(self, scored: Sequence[Tuple[float, Chromosome]]) -> Chromosome:
        candidates = self.rng.sample(list(scored), self.config.tournament_size)
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1][:]

    def _order_crossover(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        a, b = sorted(self.rng.sample(range(len(p1)), 2))
        return self._make_order_child(p1, p2, a, b), self._make_order_child(p2, p1, a, b)

    @staticmethod
    def _make_order_child(p1: Chromosome, p2: Chromosome, a: int, b: int) -> Chromosome:
        child: List[Optional[str]] = [None] * len(p1)
        child[a:b] = p1[a:b]
        fill = [gene for gene in p2 if gene not in child]
        fill_iter = iter(fill)
        for idx, gene in enumerate(child):
            if gene is None:
                child[idx] = next(fill_iter)
        return [gene for gene in child if gene is not None]

    def _mutate(self, chromosome: Chromosome) -> None:
        if self.rng.random() >= self.config.mutation_rate or len(chromosome) < 2:
            return
        a, b = self.rng.sample(range(len(chromosome)), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
