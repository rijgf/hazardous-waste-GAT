from __future__ import annotations

import random
from itertools import product
from typing import Any, Dict, Tuple

from hazardous_waste_model import ModelParams, normalize_pair, pickup_node


def _uniform(rng: random.Random, bounds: Tuple[float, float]) -> float:
    return rng.uniform(float(bounds[0]), float(bounds[1]))


def generate_random_params(config: Dict[str, Any], seed: int, cost_weight: float = 1.0, risk_weight: float = 1.0) -> ModelParams:
    rng = random.Random(seed)
    producers = [f"G{i + 1}" for i in range(int(config["producers_count"]))]
    waste_types = [f"S{i + 1}" for i in range(int(config["waste_types_count"]))]
    facilities = [f"D{i + 1}" for i in range(int(config["facilities_count"]))]
    vehicles = [f"K{i + 1}" for i in range(int(config["vehicles_count"]))]
    periods = list(range(1, int(config["periods_count"]) + 1))

    generation = {
        (i, s, t): round(_uniform(rng, tuple(config["generation_range"])), 3)
        for i, s, t in product(producers, waste_types, periods)
    }
    total_generation = sum(generation.values())
    per_producer_generation = {
        i: sum(generation[i, s, t] for s in waste_types for t in periods)
        for i in producers
    }
    per_facility_generation = total_generation / max(1, len(facilities))
    vehicle_capacity = round(total_generation / max(1, len(vehicles)) * float(config["vehicle_capacity_factor"]), 3)

    initial_producer_inventory = {(i, s): 0.0 for i, s in product(producers, waste_types)}
    initial_facility_inventory = {(j, s): 0.0 for j, s in product(facilities, waste_types)}
    producer_capacity = {
        i: round(max(per_producer_generation[i] * float(config["producer_capacity_factor"]), vehicle_capacity), 3)
        for i in producers
    }
    facility_capacity = {
        j: round(max(per_facility_generation * float(config["facility_capacity_factor"]), vehicle_capacity), 3)
        for j in facilities
    }

    nodes = facilities + [pickup_node(i, s) for i, s in product(producers, waste_types)]
    lo, hi = tuple(config["coordinate_range"])
    coords = {node: (rng.uniform(lo, hi), rng.uniform(lo, hi)) for node in nodes}
    distance = {}
    accident_probability = {}
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            if a in facilities and b in facilities:
                continue
            if a.startswith("n::") and b.startswith("n::") and a.split("::")[1] == b.split("::")[1]:
                continue
            ax, ay = coords[a]
            bx, by = coords[b]
            distance[a, b] = round(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5, 3)
            accident_probability[a, b] = round(_uniform(rng, tuple(config["accident_probability_range"])), 6)

    technology = {}
    for j, s in product(facilities, waste_types):
        technology[j, s] = 1 if rng.random() <= float(config["technology_probability"]) else 0
    for s in waste_types:
        if not any(technology[j, s] for j in facilities):
            technology[rng.choice(facilities), s] = 1

    per_type_generation = {
        s: sum(generation[i, s, t] for i in producers for t in periods)
        for s in waste_types
    }
    processing_capacity = {}
    for j, s, t in product(facilities, waste_types, periods):
        base = per_type_generation[s]
        processing_capacity[j, s, t] = round(base * float(config["processing_capacity_factor"]), 3)

    compatibility = {}
    for s1 in waste_types:
        for s2 in waste_types:
            pair = normalize_pair(s1, s2)
            compatibility[pair] = 1 if s1 == s2 or rng.random() <= float(config["compatibility_probability"]) else 0

    coload_risk = {}
    for s1 in waste_types:
        for s2 in waste_types:
            pair = normalize_pair(s1, s2)
            coload_risk[pair] = 0.0 if s1 == s2 else round(_uniform(rng, tuple(config["coload_risk_range"])), 3)

    return ModelParams(
        producers=producers,
        waste_types=waste_types,
        facilities=facilities,
        vehicles=vehicles,
        periods=periods,
        generation=generation,
        initial_producer_inventory=initial_producer_inventory,
        initial_facility_inventory=initial_facility_inventory,
        producer_capacity=producer_capacity,
        facility_capacity=facility_capacity,
        vehicle_capacity=vehicle_capacity,
        distance=distance,
        accident_probability=accident_probability,
        vehicle_fixed_cost=float(config["vehicle_fixed_cost"]),
        distance_cost=float(config["distance_cost"]),
        processing_cost={(j, s): round(_uniform(rng, tuple(config["processing_cost_range"])), 3) for j, s in product(facilities, waste_types)},
        technology=technology,
        processing_capacity=processing_capacity,
        compatibility=compatibility,
        coload_risk=coload_risk,
        waste_consequence={s: round(_uniform(rng, tuple(config["waste_consequence_range"])), 3) for s in waste_types},
        producer_inventory_risk={(i, s): round(_uniform(rng, tuple(config["producer_inventory_risk_range"])), 3) for i, s in product(producers, waste_types)},
        facility_inventory_risk={(j, s): round(_uniform(rng, tuple(config["facility_inventory_risk_range"])), 3) for j, s in product(facilities, waste_types)},
        min_pickup=0.001,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
        require_terminal_clear=bool(config.get("require_terminal_clear", True)),
    )
