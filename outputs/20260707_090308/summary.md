# Experiment Summary

## Method Summary

| scale | method | weighted_objective_normalized_mean | weighted_objective_raw_mean | cost_mean | risk_mean | runtime_mean | feasible_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| large | GA | 0.95846 | 16460.9 | 31357.2 | 1564.65 | 25.4723 | 1 |
| large | Heuristic | 1 | 16062.1 | 30353.9 | 1770.23 | 1.138e-05 | 1 |
| large | PPO | 0.590689 | 11215.8 | 23407.6 | 926.964 | 2.38934 | 1 |
| small | GA | 0.771624 | 782.907 | 1591.15 | 12.9294 | 8.80977 | 1 |
| small | Heuristic | 1 | 1032.19 | 2048.89 | 15.4837 | 5.48e-06 | 1 |
| small | MILP | 0.638367 | 688.853 | 1790.88 | 10.2255 | 60.0032 | 1 |
| small | PPO | 0.740878 | 815.959 | 1765.06 | 11.8723 | 0.439347 | 1 |

## Small-Scale Gap To MILP

| method | gap_to_milp_mean |
| --- | --- |
| GA | 10.0729 |
| Heuristic | 49.7421 |
| MILP | 0 |
| PPO | 10.0729 |

## Large-Scale Gap To PPO

| method | gap_to_ppo_mean |
| --- | --- |
| GA | 64.5202 |
| Heuristic | 73.0226 |
| PPO | 0 |

## Output Files

- `results.csv`: raw method metrics.
- `results_with_gap.csv`: metrics plus small-scale gap to MILP.
- `training_history.csv`: PPO training curves.
- `models/`: saved PPO model checkpoints.
- `figures/`: generated result figures.