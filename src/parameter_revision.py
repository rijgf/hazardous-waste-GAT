"""Parameter-only revision and common feasible reference preparation.

This module never changes the frozen policy, operators, objective, or budgets.
The assignment MILP is used only to bind a common reference before experiments;
it is not the optimization benchmark and its time is recorded separately.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from itertools import combinations
import random
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from hazardous_waste_model import normalize_pair, pickup_owner, pickup_type
from src.solution_utils import evaluate_solution, route_plan_to_solution


def revised_parameters(params, seed=0, capacity_factor=2.0, total_ratio_range=(3., 6.)):
    """Draw pair-specific strengths; the worst pair is not forced to a value."""
    if capacity_factor <= 0 or not 1 < total_ratio_range[0] < total_ratio_range[1]:
        raise ValueError('Positive capacity and an ordered total-risk range above one required')
    capacities = {}
    capacity_rows = []
    for waste in params.waste_types:
        average = sum(params.generation[i, waste, t] for i in params.producers
                      for t in params.periods) / len(params.periods)
        eligible = [j for j in params.facilities if params.technology[j, waste]]
        if not eligible:
            raise ValueError('No technology-compatible facility')
        share = capacity_factor * average / len(eligible)
        for facility in params.facilities:
            for period in params.periods:
                # Forbidden cells retain a positive nominal value for existing
                # encoders; technology=0 makes their effective capacity zero.
                capacities[facility, waste, period] = share
        capacity_rows.append({'waste': waste, 'average_period_generation': average,
                              'eligible_facilities': eligible, 'capacity_per_eligible_facility': share,
                              'effective_capacity_each_period': share * len(eligible)})
    pairs = [normalize_pair(a, b) for a, b in combinations(params.waste_types, 2)
             if params.compatibility[normalize_pair(a, b)] and params.coload_risk[normalize_pair(a, b)] > 0]
    if not pairs:
        raise ValueError('No positive compatible pair to calibrate')
    rng = random.Random(seed)
    gamma = dict(params.coload_risk)
    for pair in pairs:
        additional_strength = rng.uniform(total_ratio_range[0] - 1, total_ratio_range[1] - 1)
        gamma[pair] = additional_strength * (params.waste_consequence[pair[0]] + params.waste_consequence[pair[1]]) / 2
    pair_rows = [{'pair': list(pair), 'old_gamma': params.coload_risk[pair], 'new_gamma': gamma[pair],
                  'equal_load_total_transport_risk_ratio': 1 + 2 * gamma[pair] /
                  (params.waste_consequence[pair[0]] + params.waste_consequence[pair[1]])}
                 for pair in pairs]
    return replace(params, processing_capacity=capacities, coload_risk=gamma), {
        'capacity_factor': capacity_factor, 'capacity_definition': 'sum over technology-compatible facilities / mean global generation per period',
        'capacity_allocation': 'equal effective capacity among technology-compatible facilities; no rounding',
        'risk_total_ratio_range': list(total_ratio_range), 'risk_parameter_seed': seed,
        'risk_calibration': 'same arc, two equal positive loads; baseline transport plus additional coload risk; excludes inventory risk',
        'risk_generation': 'independent pair strengths Uniform(2,5); gamma=strength*(h_s+h_t)/2; no extremum calibration',
        'pairs': pair_rows,
        'capacity_by_waste': capacity_rows,
        'empirical_status': 'user-specified synthetic stress scenario, not a chemical safety calibration',
    }


def _order_nodes(params, nodes, facility):
    remaining = set(nodes)
    route = [facility]
    while remaining:
        counts = Counter(pickup_owner(node) for node in remaining)
        candidates = [node for node in remaining if (route[-1], node) in params.distance]
        if not candidates:
            raise ValueError('Reference assignment has no valid alternating route')
        node = min(candidates, key=lambda n: (-counts[pickup_owner(n)], n))
        route.append(node)
        remaining.remove(node)
    route.append(facility)
    return route


def construct_reference(params):
    """Deterministic all-period service with joint vehicle/facility assignment."""
    started = time.perf_counter()
    plan, evidence = {}, []
    nodes, vehicles, facilities = sorted(params.pickup_nodes), params.vehicles, params.facilities
    for period in params.periods:
        quantity = {n: params.generation[pickup_owner(n), pickup_type(n), period]
                    + (params.initial_producer_inventory[pickup_owner(n), pickup_type(n)]
                       if period == params.periods[0] else 0.) for n in nodes}
        assignments = [(n, v, j) for n in nodes for v in vehicles for j in facilities
                       if params.technology[j, pickup_type(n)]]
        active = [(v, j) for v in vehicles for j in facilities]
        x = {key: k for k, key in enumerate(assignments)}
        y = {key: len(x) + k for k, key in enumerate(active)}
        rows, lower, upper = [], [], []

        def add(entries, lb=-np.inf, ub=np.inf):
            rows.append(entries); lower.append(lb); upper.append(ub)

        for node in nodes:
            add({idx: 1. for (n, v, j), idx in x.items() if n == node}, 1., 1.)
        for vehicle in vehicles:
            add({y[vehicle, j]: 1. for j in facilities}, ub=1.)
            add({idx: quantity[n] for (n, v, j), idx in x.items() if v == vehicle}, ub=params.vehicle_capacity)
            # Same-producer pickup nodes have no connecting arc. This bound
            # guarantees an ordering with no adjacent equal producer labels.
            for producer in params.producers:
                add({idx: (1. if pickup_owner(n) == producer else -1.)
                     for (n, v, j), idx in x.items() if v == vehicle}, ub=1.)
        for (node, vehicle, facility), idx in x.items():
            add({idx: 1., y[vehicle, facility]: -1.}, ub=0.)
        for facility in facilities:
            for waste in params.waste_types:
                add({idx: quantity[n] for (n, v, j), idx in x.items()
                     if j == facility and pickup_type(n) == waste},
                    ub=params.processing_capacity[facility, waste, period] * params.technology[facility, waste])
        matrix = lil_matrix((len(rows), len(x) + len(y)), dtype=float)
        for row, entries in enumerate(rows):
            for col, value in entries.items():
                matrix[row, col] = value
        objective = np.zeros(len(x) + len(y))
        for (vehicle, facility), idx in y.items():
            objective[idx] = 1. + (vehicles.index(vehicle) * len(facilities) + facilities.index(facility)) * 1e-5
        result = milp(objective, integrality=np.ones(len(objective)), bounds=Bounds(0., 1.),
                      constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                      options={'time_limit': 60., 'mip_rel_gap': 0., 'threads': 1, 'random_seed': 0})
        if result.x is None:
            raise RuntimeError(f'Reference assignment failed in period {period}: {result.message}; not proof of full model infeasibility')
        for vehicle, facility in active:
            assigned = [node for (node, v, j), idx in x.items()
                        if v == vehicle and j == facility and result.x[idx] > .5]
            if assigned:
                plan[vehicle, period] = _order_nodes(params, assigned, facility)
        evidence.append({'period': period, 'status': int(result.status), 'message': result.message,
                         'assignment_variables': len(objective), 'assignment_constraints': len(rows)})
    metrics = evaluate_solution(params, route_plan_to_solution(params, plan))
    if not metrics['feasible']:
        raise RuntimeError(f'Constructed reference is not strictly feasible: {metrics["violations"]}')
    return plan, {'method': 'all-period joint vehicle-facility assignment followed by alternating-producer routing',
                  'solver_seed': 0, 'time_limit_per_period': 60., 'seconds': time.perf_counter() - started,
                  'purpose': 'common reference only, no cost-risk optimization or benchmark warm start',
                  'periods': evidence}
