from __future__ import annotations

import json

from hazardous_waste_model import check_solution, solve_with_milp
from sample_params import ensure_sample_params_json, load_params_from_json


def main() -> None:
    params_path = ensure_sample_params_json()
    params = load_params_from_json(params_path)
    result = solve_with_milp(params, time_limit=30.0)
    print(f"status: {result.status}")
    print(f"objective: {result.objective}")
    print("routes:")
    for (vehicle, period), route in sorted(result.solution.get("routes", {}).items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"  period {period}, vehicle {vehicle}: {' -> '.join(route)}")
    print("summary:")
    print(json.dumps(result.solution.get("summary", {}), ensure_ascii=False, indent=2))
    violations = check_solution(params, result.solution)
    print("feasibility:")
    if violations:
        for item in violations:
            print(f"  VIOLATION: {item}")
    else:
        print("  feasible")
    if result.diagnostics:
        print("diagnostics:")
        for item in result.diagnostics:
            print(f"  {item}")


if __name__ == "__main__":
    main()
