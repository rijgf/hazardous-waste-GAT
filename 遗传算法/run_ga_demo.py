from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genetic_algorithm import ClassicGeneticAlgorithm, GAConfig
from hazardous_waste_model import check_solution, display_node
from sample_params import ensure_sample_params_json, load_params_from_json


def main() -> None:
    params_path = ensure_sample_params_json()
    params = load_params_from_json(params_path)
    ga = ClassicGeneticAlgorithm(
        params,
        GAConfig(
            population_size=80,
            generations=150,
            crossover_rate=0.9,
            mutation_rate=0.25,
            elite_size=4,
            tournament_size=3,
            random_seed=7,
        ),
    )
    result = ga.run()

    print(f"feasible: {result.feasible}")
    print(f"best_objective: {result.best_objective:.6f}")
    print("best_chromosome:")
    print("  " + " -> ".join(display_node(node) for node in result.best_chromosome))
    print("routes:")
    for (vehicle, period), route in sorted(result.best_solution.get("routes", {}).items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"  period {period}, vehicle {vehicle}: {' -> '.join(route)}")
    print("summary:")
    print(json.dumps(result.best_solution.get("summary", {}), ensure_ascii=False, indent=2))
    print("feasibility_check:")
    violations = check_solution(params, result.best_solution)
    if violations:
        for item in violations:
            print(f"  VIOLATION: {item}")
    else:
        print("  feasible")
    print("history_tail:")
    print("  " + ", ".join(f"{value:.3f}" for value in result.history[-10:]))


if __name__ == "__main__":
    main()
