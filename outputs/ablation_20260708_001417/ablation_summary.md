# Ablation Experiment Summary

## Variant Summary

| scale | variant | method | weighted_objective_normalized_mean | weighted_objective_raw_mean | cost_mean | risk_mean | feasible_rate | runtime_mean | train_time_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| large | full_transformer | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 2.06329 | 605.916 |
| large | heuristic | Heuristic | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 0 | nan |
| large | hybrid_transformer_fc | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 2.1024 | 611.209 |
| large | no_preference_token | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 2.08064 | 630.315 |
| large | no_route_arc_tokens | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 2.09609 | 560.901 |
| small | full_transformer | PPO | 0.90992 | 709.785 | 1474.98 | 11.1413 | 1 | 1.47954 | 415.585 |
| small | heuristic | Heuristic | 1 | 815.464 | 1619.85 | 11.0752 | 1 | 0 | nan |
| small | hybrid_transformer_fc | PPO | 0.905296 | 705.491 | 1469.28 | 11.1942 | 1 | 1.39035 | 429.914 |
| small | no_preference_token | PPO | 0.905296 | 705.491 | 1469.28 | 11.1942 | 1 | 1.08864 | 420.656 |
| small | no_route_arc_tokens | PPO | 0.8966 | 705.395 | 1485.92 | 10.957 | 1 | 1.17102 | 396.694 |

## Variants

- `full_transformer`: full Transformer with preference token and route-arc tokens.
- `hybrid_transformer_fc`: Transformer token branch plus FC global branch.
- `no_preference_token`: removes preference token.
- `no_route_arc_tokens`: removes route-arc tokens.