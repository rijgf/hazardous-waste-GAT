# Experiment Summary

## Method Summary

| scale | method | weighted_objective_normalized_mean | weighted_objective_raw_mean | cost_mean | risk_mean | runtime_mean | feasible_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| large | GA | 0.95846 | 16460.9 | 31357.2 | 1564.65 | 23.151 | 1 |
| large | Heuristic | 1 | 16062.1 | 30353.9 | 1770.23 | 5.86e-06 | 1 |
| large | PPO | 0.590689 | 11215.8 | 23407.6 | 926.964 | 2.04264 | 1 |
| small | GA | 1.33356 | 1017.01 | 2094.76 | 9.52517 | 3.03052 | 0 |
| small | Heuristic | 1 | 765.216 | 1523.39 | 7.04355 | 3.26002e-06 | 1 |
| small | MILP | 0.681042 | 635.911 | 1552.95 | 4.5675 | 31.9224 | 1 |
| small | PPO | 0.808075 | 673.871 | 1373.3 | 5.38216 | 0.327551 | 1 |

## Small-Scale Gap To MILP

| method | gap_to_milp_mean |
| --- | --- |
| GA | 66.4145 |
| Heuristic | 23.8831 |
| MILP | 0 |
| PPO | 2.44078 |

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