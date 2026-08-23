# Ablation Experiment Summary

## Variant Summary

| scale | variant | method | weighted_objective_normalized_mean | weighted_objective_raw_mean | cost_mean | risk_mean | feasible_rate | runtime_mean | train_time_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| large | full_transformer | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 0.823603 | 3.46413 |
| large | heuristic | Heuristic | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 0 | nan |
| large | hybrid_transformer_fc | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 1.01151 | 5.14178 |
| large | no_preference_token | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 0.892357 | 4.70342 |
| large | no_route_arc_tokens | PPO | 1 | 16157.6 | 30261.4 | 2053.87 | 1 | 0.701654 | 2.52418 |
| small | full_transformer | PPO | 0.749958 | 818.944 | 1641.93 | 11.3748 | 1 | 0.2911 | 1.65808 |
| small | heuristic | Heuristic | 1 | 858.762 | 1699.54 | 17.9835 | 1 | 0 | nan |
| small | hybrid_transformer_fc | PPO | 0.750284 | 825.772 | 1651.13 | 11.1064 | 1 | 0.289791 | 1.12576 |
| small | no_preference_token | PPO | 0.750284 | 825.772 | 1651.13 | 11.1064 | 1 | 0.290792 | 1.09793 |
| small | no_route_arc_tokens | PPO | 0.749958 | 818.944 | 1641.93 | 11.3748 | 1 | 0.296928 | 1.04008 |

## Variants

- `full_transformer`: full Transformer with preference token and route-arc tokens.
- `hybrid_transformer_fc`: Transformer token branch plus FC global branch.
- `no_preference_token`: removes preference token.
- `no_route_arc_tokens`: removes route-arc tokens.