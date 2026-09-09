"""Versioned Pareto instrumentation; the frozen single-preference search is untouched.

Only use one observer per process. Observing adds no random draws, changes no
candidate, and does not feed the external archive back into either algorithm.
"""
from __future__ import annotations

from contextlib import contextmanager
from collections import Counter, OrderedDict
from dataclasses import dataclass
import math
import random
import time

import numpy as np

from src.operators import clone_plan, repair_plan
from src.reproducibility import plan_sha256, plan_to_canonical_data
from src.solution_utils import evaluate_solution, route_plan_to_solution

TOLERANCE = 1e-8


class ParetoArchive:
    def __init__(self, refs, tolerance=TOLERANCE):
        if len(refs) != 2 or any(not math.isfinite(x) or x <= 0 for x in refs):
            raise ValueError('Positive finite instance references required')
        self.refs = tuple(refs)
        self.tolerance = tolerance
        self.points = []

    def weak(self, a, b):
        return all((a[k] - b[k]) / scale <= self.tolerance
                   for k, scale in zip(('cost', 'risk'), self.refs))

    def equal(self, a, b):
        return self.weak(a, b) and self.weak(b, a)

    def strict(self, a, b):
        return self.weak(a, b) and not self.weak(b, a)

    def add(self, point):
        if not all(math.isfinite(point[k]) for k in ('cost', 'risk')):
            raise ValueError('Nonfinite objective')
        if any(self.strict(existing, point) for existing in self.points):
            return False
        for index, existing in enumerate(self.points):
            if self.equal(existing, point):
                if point['solution_id'] < existing['solution_id']:
                    self.points = [other for other in self.points
                                   if not self.equal(other, point) and not self.strict(point, other)]
                    self.points.append(point)
                    return True
                return False
        self.points = [existing for existing in self.points if not self.strict(point, existing)]
        self.points.append(point)
        return True

    def sorted_points(self):
        points = sorted(self.points, key=lambda p: (p['cost'], p['risk'], p['solution_id']))
        for index, a in enumerate(points):
            for b in points[index+1:]:
                if self.strict(a,b) or self.strict(b,a) or self.equal(a,b):
                    raise RuntimeError('Archive must contain only distinct nondominated objectives')
        return points


def coverage_pair(points_a, points_b, refs, tolerance=TOLERANCE):
    a, b = ParetoArchive(refs, tolerance), ParetoArchive(refs, tolerance)
    for points, archive in ((points_a, a), (points_b, b)):
        for point in sorted(points, key=lambda p: p['solution_id']):
            archive.add(point)
    result = {'n_A': len(a.points), 'n_B': len(b.points), 'tolerance': tolerance,
              'status': 'valid' if a.points and b.points else 'failed_empty_front',
              'A_covers_B_count': None, 'B_covers_A_count': None,
              'A_covers_B': None, 'B_covers_A': None}
    if a.points and b.points:
        ab = sum(any(a.weak(x, y) for x in a.points) for y in b.points)
        ba = sum(any(b.weak(y, x) for y in b.points) for x in a.points)
        result.update(A_covers_B_count=ab, B_covers_A_count=ba,
                      A_covers_B=ab / len(b.points), B_covers_A=ba / len(a.points))
    return result


class CandidateLedger:
    """One complete-front run's strict evaluator, cache and external archive."""
    def __init__(self, params, refs, reference_plan):
        self.params, self.refs = params, tuple(refs)
        self.archive = ParetoArchive(refs)
        self.cache = OrderedDict()
        self.counts = Counter(candidate_attempts=0, candidate_objective_evaluations=0,
                              cache_hits=0, invalid_candidates=0, operator_failures=0,
                              invalid_candidate_attempts=0,
                              ppo_internal_objective_evaluations=0)
        started = time.perf_counter()
        point = self.evaluate(reference_plan, {'source': 'common_reference'}, preparation=True)
        if not point or not point['feasible']:
            raise ValueError('Reference plan is not strictly feasible')
        self.preparation_seconds = time.perf_counter() - started

    def evaluate(self, plan, provenance, preparation=False):
        if not preparation:
            self.counts['candidate_attempts'] += 1
        key = tuple(sorted((v, t, tuple(route)) for (v, t), route in plan.items()))
        if key in self.cache:
            self.counts['cache_hits'] += 1
            self.cache.move_to_end(key)
            if not self.cache[key]['feasible']:
                self.counts['invalid_candidate_attempts'] += 1
            return self.cache[key]
        metrics = evaluate_solution(self.params, route_plan_to_solution(self.params, plan),
                                    (.5, .5), objective_refs=self.refs)
        if not preparation:
            self.counts['candidate_objective_evaluations'] += 1
        point = {**metrics, 'solution_id': plan_sha256(plan), 'provenance': dict(provenance)}
        self.cache[key] = point
        if len(self.cache) > 2048:
            self.cache.popitem(last=False)
        if metrics['feasible']:
            archived = dict(point)
            if self.archive.add(archived):
                archived['plan'] = plan_to_canonical_data(plan)
        else:
            self.counts['invalid_candidates'] += 1
            self.counts['invalid_candidate_attempts'] += 1
        return point

    @contextmanager
    def observe_ppo(self, preference, evaluation_seed):
        import src.ppo_improver as module
        original_operator, original_evaluate = module.apply_operator, module.evaluate_solution
        position = 0

        def observing_operator(params, plan, action, seed=0):
            nonlocal position
            candidate, ok, reason = original_operator(params, plan, action, seed=seed)
            if not ok:
                self.counts['operator_failures'] += 1
            self.evaluate(candidate, {'source': 'PPO_candidate', 'preference': list(preference),
                                      'evaluation_seed': evaluation_seed, 'operator_seed': seed,
                                      'candidate_index': position, 'operator_ok': ok,
                                      'operator_reason': reason})
            position += 1
            return candidate, ok, reason

        def observing_evaluate(*args, **kwargs):
            self.counts['ppo_internal_objective_evaluations'] += 1
            return original_evaluate(*args, **kwargs)

        module.apply_operator, module.evaluate_solution = observing_operator, observing_evaluate
        try:
            yield
        finally:
            module.apply_operator, module.evaluate_solution = original_operator, original_evaluate


@dataclass
class Individual:
    genome: list
    point: dict
    rank: int = 0
    crowding: float = 0.


class MultiPeriodEncoding:
    """Gene per (pickup, period): serve, vehicle, facility, random-key order.

    Terminal visits are mandatory, earlier visits can be toggled independently;
    all vehicles, facilities and within-route orders are available. Deterministic
    common repair enforces loading, technology, compatibility and inventories.
    """
    def __init__(self, params):
        self.params = params
        self.slots = [(node, period) for period in params.periods for node in params.pickup_nodes]

    def encode(self, plan):
        present = {}
        for (vehicle, period), route in plan.items():
            for index, node in enumerate(route[1:-1]):
                present[node, period] = (1, self.params.vehicles.index(vehicle),
                                        self.params.facilities.index(route[0]), index / max(1, len(route)))
        return [present.get(slot, (0, 0, 0, 0.5)) for slot in self.slots]

    def decode(self, genome):
        groups = {}
        for (node, period), (serve, vehicle, facility, order) in zip(self.slots, genome):
            if serve or period == self.params.periods[-1]:
                groups.setdefault((self.params.vehicles[vehicle], period), []).append((order, node, facility))
        plan = {}
        for key, entries in groups.items():
            entries.sort()
            facility = self.params.facilities[entries[0][2]]
            plan[key] = [facility, *[entry[1] for entry in entries], facility]
        return repair_plan(self.params, plan)

    def random_genome(self, rng):
        return [(int(period == self.params.periods[-1] or rng.random() < 0.5),
                 rng.randrange(len(self.params.vehicles)), rng.randrange(len(self.params.facilities)), rng.random())
                for _, period in self.slots]

    def child(self, left, right, rng, crossover=0.90, mutation=0.25):
        genes = list(left)
        if rng.random() < crossover:
            genes = [a if rng.random() < .5 else b for a, b in zip(left, right)]
        if rng.random() < mutation:
            # One of four independent decision components; order swaps are
            # represented by changing random keys, not by a fixed permutation.
            index = rng.randrange(len(genes))
            item = list(genes[index])
            component = rng.randrange(4)
            if component == 0:
                item[0] = 1 - item[0]
            elif component == 1:
                item[1] = rng.randrange(len(self.params.vehicles))
            elif component == 2:
                item[2] = rng.randrange(len(self.params.facilities))
            else:
                item[3] = rng.random()
            if self.slots[index][1] == self.params.periods[-1]:
                item[0] = 1
            genes[index] = tuple(item)
        return genes


def rank_and_crowding(population, refs):
    """Deb fast nondominated sort plus per-front two-objective crowding."""
    n = len(population)
    if not n:
        return []
    values = np.array([[p.point['cost'] / refs[0], p.point['risk'] / refs[1]] for p in population])
    feasible = np.array([p.point['feasible'] for p in population])
    violation = np.array([len(p.point.get('violations', [])) if not p.point['feasible'] else 0 for p in population])
    weak = (values[:, None, :] <= values[None, :, :] + TOLERANCE).all(axis=2)
    better = (values[:, None, :] < values[None, :, :] - TOLERANCE).any(axis=2)
    dominates = ((weak & better & feasible[:, None] & feasible[None, :]) |
                 (feasible[:, None] & ~feasible[None, :]) |
                 (~feasible[:, None] & ~feasible[None, :] & (violation[:, None] < violation[None, :])))
    np.fill_diagonal(dominates, False)
    counts = dominates.sum(axis=0)
    front = np.flatnonzero(counts == 0).tolist()
    fronts = []
    rank = 0
    while front:
        fronts.append(front)
        next_front = []
        for index in front:
            population[index].rank, population[index].crowding = rank, 0.
            for other in np.flatnonzero(dominates[index]):
                counts[other] -= 1
                if counts[other] == 0:
                    next_front.append(int(other))
        for objective in range(2):
            ordered = sorted(front, key=lambda index: (values[index, objective], index))
            span = values[ordered[-1], objective] - values[ordered[0], objective]
            if span > 0:
                population[ordered[0]].crowding = population[ordered[-1]].crowding = math.inf
                for location in range(1, len(ordered) - 1):
                    population[ordered[location]].crowding += (values[ordered[location+1], objective] - values[ordered[location-1], objective]) / span
        front, rank = next_front, rank + 1
    if sum(map(len, fronts)) != n:
        raise RuntimeError('Numerical tolerance created a dominance cycle')
    return fronts


def nsga_fronts(params, refs, reference_plan, seed, budgets=(40320,), population_size=100,
                checkpoint=None):
    rng = random.Random(seed)
    encoding = MultiPeriodEncoding(params)
    ledger = CandidateLedger(params, refs, reference_plan)
    reference = next(iter(ledger.cache.values()))
    population = [Individual(encoding.encode(reference_plan), reference)]
    started = time.perf_counter()
    snapshots = {}
    generation = 0

    def snapshot(partial):
        attempts = ledger.counts['candidate_attempts']
        if attempts in budgets:
            value = {'budget': attempts, 'counts': dict(ledger.counts),
                     'seconds': time.perf_counter() - started,
                     'preparation_seconds': ledger.preparation_seconds,
                     'last_complete_generation': generation, 'partial_generation_candidates': partial,
                     'points': ledger.archive.sorted_points(), 'stop_reason': 'candidate_budget_reached'}
            snapshots[attempts] = value
            if checkpoint:
                checkpoint(value)

    def generate(genome, source):
        plan, repaired = encoding.decode(genome)
        if not repaired:
            ledger.counts['operator_failures'] += 1
        point = ledger.evaluate(plan, source)
        return Individual(genome, point)

    for index in range(1, population_size):
        if ledger.counts['candidate_attempts'] >= max(budgets):
            break
        population.append(generate(encoding.random_genome(rng), {'source': 'NSGA_initial', 'index': index}))
        snapshot(index)
    rank_and_crowding(population, refs)

    def tournament():
        a, b = rng.sample(population, 2)
        if a.rank != b.rank:
            return a if a.rank < b.rank else b
        if a.crowding != b.crowding:
            return a if a.crowding > b.crowding else b
        return a if rng.random() < .5 else b

    while ledger.counts['candidate_attempts'] < max(budgets):
        children = []
        for index in range(population_size):
            if ledger.counts['candidate_attempts'] >= max(budgets):
                break
            genome = encoding.child(tournament().genome, tournament().genome, rng)
            children.append(generate(genome, {'source': 'NSGA_offspring', 'generation': generation + 1, 'index': index}))
            snapshot(len(children))
        combined = population + children
        fronts = rank_and_crowding(combined, refs)
        selected = []
        for front in fronts:
            candidates = [combined[index] for index in front]
            candidates.sort(key=lambda p: -p.crowding)
            selected.extend(candidates[:population_size - len(selected)])
            if len(selected) == population_size:
                break
        population = selected
        if len(children) == population_size:
            generation += 1
    return snapshots
