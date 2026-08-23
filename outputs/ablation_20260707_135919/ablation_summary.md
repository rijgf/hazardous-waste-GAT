# Ablation Experiment Summary

## Variant Summary

| scale | variant | method | weighted_objective_normalized_mean | weighted_objective_raw_mean | cost_mean | risk_mean | feasible_rate | runtime_mean | train_time_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| large | full_transformer | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 2.66912 | 278.902 |
| large | heuristic | Heuristic | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 0 | nan |
| large | hybrid_transformer_fc | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 2.69931 | 280.292 |
| large | no_preference_token | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 2.71814 | 276.685 |
| large | no_route_arc_tokens | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 2.31756 | 204.044 |
| small | full_transformer | PPO | 0.905296 | 705.491 | 1469.28 | 11.0533 | 1 | 1.21915 | 101.707 |
| small | heuristic | Heuristic | 1 | 815.464 | 1619.85 | 11.0752 | 1 | 0 | nan |
| small | hybrid_transformer_fc | PPO | 0.905296 | 705.491 | 1469.28 | 10.8302 | 1 | 1.14553 | 94.2527 |
| small | no_preference_token | PPO | 0.8966 | 705.395 | 1485.92 | 10.8748 | 1 | 1.12131 | 95.8555 |
| small | no_route_arc_tokens | PPO | 0.905296 | 705.491 | 1469.28 | 11.1942 | 1 | 1.33919 | 99.8218 |

## Variants

- `full_transformer`: full Transformer with preference token and route-arc tokens.
- `hybrid_transformer_fc`: Transformer token branch plus FC global branch.
- `no_preference_token`: removes preference token.
- `no_route_arc_tokens`: removes route-arc tokens.