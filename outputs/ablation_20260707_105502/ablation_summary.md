# Ablation Experiment Summary

## Variant Summary

| scale | variant | method | weighted_objective_normalized_mean | weighted_objective_raw_mean | cost_mean | risk_mean | feasible_rate | runtime_mean | train_time_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| large | full_transformer | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 0.915159 | 36.7422 |
| large | heuristic | Heuristic | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 0 | nan |
| large | hybrid_transformer_fc | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 0.936309 | 36.6322 |
| large | no_preference_token | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 1.01542 | 33.9758 |
| large | no_route_arc_tokens | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 0.757905 | 24.0495 |
| small | full_transformer | PPO | 0.905296 | 705.491 | 1469.28 | 11.0533 | 1 | 0.325189 | 10.3744 |
| small | heuristic | Heuristic | 1 | 815.464 | 1619.85 | 11.0752 | 1 | 0 | nan |
| small | hybrid_transformer_fc | PPO | 0.8966 | 705.395 | 1485.92 | 11.0978 | 1 | 0.329395 | 10.0299 |
| small | no_preference_token | PPO | 0.905296 | 705.491 | 1469.28 | 10.9711 | 1 | 0.323052 | 9.83765 |
| small | no_route_arc_tokens | PPO | 0.905296 | 705.491 | 1469.28 | 10.8302 | 1 | 0.323182 | 9.85584 |

## Variants

- `full_transformer`: full Transformer with preference token and route-arc tokens.
- `hybrid_transformer_fc`: Transformer token branch plus FC global branch.
- `no_preference_token`: removes preference token.
- `no_route_arc_tokens`: removes route-arc tokens.