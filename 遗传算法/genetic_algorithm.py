from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from hazardous_waste_model import ModelParams
from src.operators import repair_plan
from src.solution_utils import (
    evaluate_solution,
    make_route,
    route_plan_to_solution,
    validate_solution,
)


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
    best_chromosome: Optional[Chromosome]
    best_solution: Dict[str, Any]
    feasible: bool
    history: List[float]
    best_source: str


class ClassicGeneticAlgorithm:
    """Classic permutation GA for the hazardous-waste routing subproblem.

    The chromosome is a permutation of pickup nodes. The decoder seeds a
    terminal-period visit order, then delegates vehicle/facility assignment and
    all feasibility repair to the shared deterministic route repairer.
    """

    def __init__(
        self,
        params: ModelParams,
        config: Optional[GAConfig] = None,
        initial_chromosome: Optional[Sequence[str]] = None,
        initial_solution: Optional[Mapping[str, Any]] = None,
    ):
        self.params = params
        self.config = config or GAConfig()
        self.rng = random.Random(self.config.random_seed)
        self.last_period = params.periods[-1]
        self.initial_chromosome = (
            list(initial_chromosome) if initial_chromosome is not None else None
        )
        if self.initial_chromosome is not None:
            expected = sorted(params.pickup_nodes)
            if sorted(self.initial_chromosome) != expected:
                raise ValueError(
                    "initial_chromosome must contain every pickup node exactly once"
                )
        self.initial_solution = dict(initial_solution) if initial_solution is not None else None
        if self.initial_solution is not None and self._violations(self.initial_solution):
            raise ValueError("initial_solution must be strictly feasible")

    def run(self) -> GAResult:
        population = self._initial_population()
        history: List[float] = []
        best_chromosome = population[0]
        best_objective, best_solution = self.evaluate(best_chromosome)
        best_source = "genetic"
        if self.initial_solution is not None:
            locked_objective = evaluate_solution(
                self.params,
                self.initial_solution,
                (self.params.cost_weight, self.params.risk_weight),
                check_constraints=False,
            )["weighted_objective"]
            if locked_objective <= best_objective:
                best_objective = float(locked_objective)
                best_solution = dict(self.initial_solution)
                best_chromosome = None
                best_source = "locked_initial_solution"

        for _ in range(self.config.generations):
            scored = [(self.evaluate(ch)[0], ch) for ch in population]
            scored.sort(key=lambda item: item[0])
            if scored[0][0] < best_objective:
                best_objective = scored[0][0]
                best_chromosome = scored[0][1][:]
                best_solution = self.evaluate(best_chromosome)[1]
                best_source = "genetic"
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

        feasible = not self._violations(best_solution)
        return GAResult(
            best_objective,
            best_chromosome,
            best_solution,
            feasible,
            history,
            best_source,
        )

    def evaluate(self, chromosome: Chromosome) -> Tuple[float, Dict[str, Any]]:
        solution = self.decode(chromosome)
        violations = self._violations(solution)
        objective = evaluate_solution(self.params, solution, (self.params.cost_weight, self.params.risk_weight), check_constraints=False)["weighted_objective"]
        if violations:
            objective += 1_000_000.0 + 10_000.0 * len(violations)
        return objective, solution

    def _violations(self, solution: Dict[str, Any]) -> List[str]:
        return validate_solution(self.params, solution)

    def decode(self, chromosome: Chromosome) -> Dict[str, Any]:
        # Delegate capacity, technology, compatibility, allowed-arc, inventory,
        # and processing repair to the same deterministic implementation used by
        # PPO.  The legacy decoder fixed every route to facilities[0], making all
        # chromosomes infeasible whenever another facility was required.
        seed_plan = {
            (self.params.vehicles[0], self.last_period): make_route(
                self.params.facilities[0],
                chromosome,
            )
        }
        repaired_plan, repaired = repair_plan(self.params, seed_plan)
        if repaired:
            return route_plan_to_solution(self.params, repaired_plan)
        return {}

    def _initial_population(self) -> List[Chromosome]:
        base = self.params.pickup_nodes
        population = []
        if self.initial_chromosome is not None:
            population.append(self.initial_chromosome[:])
        while len(population) < self.config.population_size:
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
