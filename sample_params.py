from __future__ import annotations

import json
from pathlib import Path
from itertools import product
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from hazardous_waste_model import ModelParams, normalize_pair, pickup_node


PARAMS_JSON = Path(__file__).with_name("sample_params.json")


def build_sample_params() -> ModelParams:
    producers = ["G1", "G2"]
    waste_types = ["S1", "S2"]
    facilities = ["D1"]
    vehicles = ["K1", "K2"]
    periods = [1, 2]

    generation = {
        ("G1", "S1", 1): 2.0,
        ("G1", "S2", 1): 1.0,
        ("G2", "S1", 1): 1.5,
        ("G2", "S2", 1): 1.0,
        ("G1", "S1", 2): 1.0,
        ("G1", "S2", 2): 1.0,
        ("G2", "S1", 2): 1.0,
        ("G2", "S2", 2): 1.0,
    }
    initial_producer_inventory = {(i, s): 0.0 for i, s in product(producers, waste_types)}
    initial_facility_inventory = {(j, s): 0.0 for j, s in product(facilities, waste_types)}
    producer_capacity = {"G1": 10.0, "G2": 10.0}
    facility_capacity = {"D1": 30.0}
    vehicle_capacity = 10.0

    nodes = facilities + [pickup_node(i, s) for i, s in product(producers, waste_types)]
    coord = {
        "D1": (0.0, 0.0),
        pickup_node("G1", "S1"): (2.0, 1.0),
        pickup_node("G1", "S2"): (2.0, 1.2),
        pickup_node("G2", "S1"): (5.0, 1.5),
        pickup_node("G2", "S2"): (5.0, 1.7),
    }
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
            ax, ay = coord[a]
            bx, by = coord[b]
            d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            distance[(a, b)] = round(d, 3)
            accident_probability[(a, b)] = 0.001

    technology = {(j, s): 1 for j, s in product(facilities, waste_types)}
    processing_capacity = {(j, s, t): 20.0 for j, s, t in product(facilities, waste_types, periods)}
    compatibility = {normalize_pair(s1, s2): 1 for s1 in waste_types for s2 in waste_types}
    coload_risk = {normalize_pair("S1", "S1"): 0.0, normalize_pair("S1", "S2"): 0.2, normalize_pair("S2", "S2"): 0.0}

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
        vehicle_fixed_cost=50.0,
        distance_cost=3.0,
        processing_cost={(j, s): 8.0 for j, s in product(facilities, waste_types)},
        technology=technology,
        processing_capacity=processing_capacity,
        compatibility=compatibility,
        coload_risk=coload_risk,
        waste_consequence={"S1": 1.0, "S2": 1.5},
        producer_inventory_risk={(i, s): 0.4 for i, s in product(producers, waste_types)},
        facility_inventory_risk={(j, s): 0.2 for j, s in product(facilities, waste_types)},
        min_pickup=0.001,
        cost_weight=1.0,
        risk_weight=1.0,
        require_terminal_clear=True,
    )


def _encode_tuple_map(values: Mapping[Tuple[Any, ...], Any]) -> list[dict[str, Any]]:
    return [{"key": list(key), "value": value} for key, value in values.items()]


def _decode_tuple_map(items: Iterable[Mapping[str, Any]], key_types: Sequence[type] | None = None) -> Dict[Tuple[Any, ...], Any]:
    decoded: Dict[Tuple[Any, ...], Any] = {}
    for item in items:
        key = list(item["key"])
        if key_types is not None:
            key = [key_type(value) for key_type, value in zip(key_types, key)]
        decoded[tuple(key)] = item["value"]
    return decoded


def params_to_json_data(params: ModelParams) -> Dict[str, Any]:
    return {
        "producers": params.producers,
        "waste_types": params.waste_types,
        "facilities": params.facilities,
        "vehicles": params.vehicles,
        "periods": params.periods,
        "generation": _encode_tuple_map(params.generation),
        "initial_producer_inventory": _encode_tuple_map(params.initial_producer_inventory),
        "initial_facility_inventory": _encode_tuple_map(params.initial_facility_inventory),
        "producer_capacity": params.producer_capacity,
        "facility_capacity": params.facility_capacity,
        "vehicle_capacity": params.vehicle_capacity,
        "distance": _encode_tuple_map(params.distance),
        "accident_probability": _encode_tuple_map(params.accident_probability),
        "vehicle_fixed_cost": params.vehicle_fixed_cost,
        "distance_cost": params.distance_cost,
        "processing_cost": _encode_tuple_map(params.processing_cost),
        "technology": _encode_tuple_map(params.technology),
        "processing_capacity": _encode_tuple_map(params.processing_capacity),
        "compatibility": _encode_tuple_map(params.compatibility),
        "coload_risk": _encode_tuple_map(params.coload_risk),
        "waste_consequence": params.waste_consequence,
        "producer_inventory_risk": _encode_tuple_map(params.producer_inventory_risk),
        "facility_inventory_risk": _encode_tuple_map(params.facility_inventory_risk),
        "min_pickup": params.min_pickup,
        "cost_weight": params.cost_weight,
        "risk_weight": params.risk_weight,
        "require_terminal_clear": params.require_terminal_clear,
    }


def params_from_json_data(data: Mapping[str, Any]) -> ModelParams:
    return ModelParams(
        producers=list(data["producers"]),
        waste_types=list(data["waste_types"]),
        facilities=list(data["facilities"]),
        vehicles=list(data["vehicles"]),
        periods=[int(period) for period in data["periods"]],
        generation=_decode_tuple_map(data["generation"], (str, str, int)),
        initial_producer_inventory=_decode_tuple_map(data["initial_producer_inventory"], (str, str)),
        initial_facility_inventory=_decode_tuple_map(data["initial_facility_inventory"], (str, str)),
        producer_capacity=dict(data["producer_capacity"]),
        facility_capacity=dict(data["facility_capacity"]),
        vehicle_capacity=data["vehicle_capacity"],
        distance=_decode_tuple_map(data["distance"], (str, str)),
        accident_probability=_decode_tuple_map(data["accident_probability"], (str, str)),
        vehicle_fixed_cost=data["vehicle_fixed_cost"],
        distance_cost=data["distance_cost"],
        processing_cost=_decode_tuple_map(data["processing_cost"], (str, str)),
        technology=_decode_tuple_map(data["technology"], (str, str)),
        processing_capacity=_decode_tuple_map(data["processing_capacity"], (str, str, int)),
        compatibility=_decode_tuple_map(data["compatibility"], (str, str)),
        coload_risk=_decode_tuple_map(data["coload_risk"], (str, str)),
        waste_consequence=dict(data["waste_consequence"]),
        producer_inventory_risk=_decode_tuple_map(data["producer_inventory_risk"], (str, str)),
        facility_inventory_risk=_decode_tuple_map(data["facility_inventory_risk"], (str, str)),
        min_pickup=data.get("min_pickup", 0.001),
        cost_weight=data.get("cost_weight", 1.0),
        risk_weight=data.get("risk_weight", 1.0),
        require_terminal_clear=data.get("require_terminal_clear", True),
    )


def write_sample_params_json(path: Path | str = PARAMS_JSON) -> Path:
    output_path = Path(path)
    data = params_to_json_data(build_sample_params())
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(output_path)
    return output_path


def load_params_from_json(path: Path | str = PARAMS_JSON) -> ModelParams:
    input_path = Path(path)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    return params_from_json_data(data)


def ensure_sample_params_json(path: Path | str = PARAMS_JSON) -> Path:
    output_path = Path(path)
    if not output_path.exists():
        write_sample_params_json(output_path)
    return output_path


def main() -> None:
    output_path = write_sample_params_json()
    print(f"written: {output_path}")


if __name__ == "__main__":
    main()
