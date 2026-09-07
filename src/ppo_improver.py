from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from hazardous_waste_model import ModelParams, pickup_owner, pickup_type
from src.heuristics import build_greedy_initial_plan
from src.operators import OPERATORS, OperatorAction, apply_operator
from src.reproducibility import plan_sha256
from src.solution_utils import RoutePlan, evaluate_solution, route_plan_to_solution


TOKEN_FEATURES = 18
CHECKPOINT_FORMAT_VERSION = 2

_NETWORK_CONFIG_DEFAULTS: Dict[str, Any] = {
    "network_type": "full_transformer",
    "dropout": 0.0,
    "use_preference_token": True,
    "use_preference_in_global": False,
    "use_route_arc_tokens": True,
    "use_facility_tokens": True,
    "max_tokens": 512,
    "operator_count": len(OPERATORS),
    "object_count": 128,
}
_NETWORK_CONFIG_CASTERS = {
    "network_type": str,
    "embedding_dim": int,
    "transformer_layers": int,
    "attention_heads": int,
    "ff_hidden_dim": int,
    "dropout": float,
    "use_preference_token": bool,
    "use_preference_in_global": bool,
    "use_route_arc_tokens": bool,
    "use_facility_tokens": bool,
    "max_tokens": int,
    "operator_count": int,
    "object_count": int,
}


def _network_semantics(network_config: Dict[str, Any]) -> Dict[str, Any]:
    semantics: Dict[str, Any] = {}
    for field, caster in _NETWORK_CONFIG_CASTERS.items():
        if field in network_config:
            value = network_config[field]
        elif field in _NETWORK_CONFIG_DEFAULTS:
            value = _NETWORK_CONFIG_DEFAULTS[field]
        else:
            raise ValueError(f"network_config is missing required field {field!r}")
        semantics[field] = caster(value)
    return semantics


def _required_object_capacity(params: ModelParams) -> int:
    periods = len(params.periods)
    return max(
        len(params.pickup_nodes) * periods,
        len(params.vehicles) * periods,
        len(params.pickup_nodes),
        len(params.vehicles),
        len(params.facilities),
        len(params.waste_types),
        periods,
    )


def _validate_checkpoint_operator_vocabulary(checkpoint: Dict[str, Any]) -> None:
    checkpoint_operators = checkpoint.get("operators")
    format_version = int(checkpoint.get("checkpoint_format_version", 1))
    if checkpoint_operators is None:
        if format_version >= CHECKPOINT_FORMAT_VERSION:
            raise ValueError("checkpoint is missing operator vocabulary metadata")
        return
    if list(checkpoint_operators) != list(OPERATORS):
        raise ValueError(
            "checkpoint operator vocabulary does not match the current implementation"
        )


@dataclass
class PPOEvalResult:
    solution: Dict[str, Any]
    plan: RoutePlan
    history: List[Dict[str, float]]
    trace: List[Dict[str, Any]]


def _safe(value: float, scale: float) -> float:
    return float(value) / max(float(scale), 1e-9)


class StateEncoder:
    def __init__(self, params: ModelParams, network_config: Dict[str, Any]):
        self.params = params
        self.max_tokens = int(network_config.get("max_tokens", 512))
        self.use_preference_token = bool(network_config.get("use_preference_token", True))
        self.use_preference_in_global = bool(network_config.get("use_preference_in_global", False))
        self.use_route_arc_tokens = bool(network_config.get("use_route_arc_tokens", True))
        self.use_facility_tokens = bool(network_config.get("use_facility_tokens", True))
        self.max_distance = max(params.distance.values()) if params.distance else 1.0
        self.max_generation = max(params.generation.values()) if params.generation else 1.0
        self.max_capacity = max([params.vehicle_capacity, *params.producer_capacity.values(), *params.facility_capacity.values()])
        self.max_period = max(params.periods) if params.periods else 1

    @staticmethod
    def required_token_capacity(
        params: ModelParams, network_config: Dict[str, Any]
    ) -> int:
        """Return a safe token bound for any valid plan on ``params``."""
        periods = len(params.periods)
        pickups = len(params.pickup_nodes)
        vehicle_periods = len(params.vehicles) * periods
        tokens = 1
        if bool(network_config.get("use_preference_token", True)):
            tokens += 1
        tokens += periods
        tokens += vehicle_periods
        tokens += pickups * periods
        if bool(network_config.get("use_facility_tokens", True)):
            tokens += len(params.facilities) * len(params.waste_types) * periods
        if bool(network_config.get("use_route_arc_tokens", True)):
            visits = pickups * periods
            nonempty_routes = min(vehicle_periods, visits)
            tokens += visits + nonempty_routes
        return tokens

    def encode(self, plan: RoutePlan, preference: Tuple[float, float], step_ratio: float, no_improve_ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        solution = route_plan_to_solution(self.params, plan)
        metrics = evaluate_solution(self.params, solution, preference, check_constraints=False)
        raw = solution["raw"]
        tokens: List[List[float]] = []

        global_values = [step_ratio, no_improve_ratio, _safe(metrics["cost"], 1000.0), _safe(metrics["risk"], 100.0), _safe(metrics["weighted_objective"], 1000.0)]
        if self.use_preference_in_global:
            global_values = [preference[0], preference[1], *global_values]
        global_features = self._feature(
            0,
            *global_values,
        )
        tokens.append(global_features)

        if self.use_preference_token:
            tokens.append(self._feature(1, preference[0], preference[1]))

        for period in self.params.periods:
            generated = sum(self.params.generation[i, s, period] for i in self.params.producers for s in self.params.waste_types)
            producer_inventory = sum(raw.get(("IG", node, period), 0.0) for node in self.params.pickup_nodes)
            facility_inventory = sum(raw.get(("ID", j, s, period), 0.0) for j in self.params.facilities for s in self.params.waste_types)
            processed = sum(raw.get(("p", j, s, period), 0.0) for j in self.params.facilities for s in self.params.waste_types)
            collected = sum(raw.get(("q", node, vehicle, period), 0.0) for node in self.params.pickup_nodes for vehicle in self.params.vehicles)
            used_vehicles = sum(1 for vehicle in self.params.vehicles if (vehicle, period) in plan and len(plan[(vehicle, period)]) > 2)
            tokens.append(
                self._feature(
                    2,
                    _safe(period, self.max_period),
                    _safe(generated, self.max_generation * max(1, len(self.params.pickup_nodes))),
                    _safe(collected, self.max_generation * max(1, len(self.params.pickup_nodes))),
                    _safe(producer_inventory, self.max_capacity * max(1, len(self.params.pickup_nodes))),
                    _safe(facility_inventory, self.max_capacity * max(1, len(self.params.facilities) * len(self.params.waste_types))),
                    _safe(processed, self.max_capacity * max(1, len(self.params.facilities) * len(self.params.waste_types))),
                    _safe(used_vehicles, max(1, len(self.params.vehicles))),
                )
            )

        for vehicle in self.params.vehicles:
            for period in self.params.periods:
                route = plan.get((vehicle, period), [])
                used = 1.0 if len(route) > 2 else 0.0
                length = sum(self.params.distance.get((a, b), 0.0) for a, b in zip(route, route[1:]))
                load = sum(raw.get(("q", node, vehicle, period), 0.0) for node in self.params.pickup_nodes)
                task_count = max(0, len(route) - 2)
                arc_risk = 0.0
                for a, b in zip(route, route[1:]):
                    if (a, b) in self.params.accident_probability:
                        arc_risk += self.params.accident_probability[a, b]
                tokens.append(
                    self._feature(
                        3,
                        _safe(self.params.vehicles.index(vehicle) + 1, len(self.params.vehicles)),
                        _safe(period, self.max_period),
                        used,
                        _safe(length, self.max_distance * max(1, len(self.params.pickup_nodes))),
                        _safe(load, self.params.vehicle_capacity),
                        _safe(task_count, max(1, len(self.params.pickup_nodes))),
                        _safe(arc_risk, max(self.params.accident_probability.values()) * max(1, len(route) - 1)),
                    )
                )

        for node in self.params.pickup_nodes:
            i = pickup_owner(node)
            s = pickup_type(node)
            for period in self.params.periods:
                assigned = 1.0 if any(key[1] == period and node in route for key, route in plan.items()) else 0.0
                available = raw.get(("BG", node, period), 0.0)
                remaining = raw.get(("IG", node, period), 0.0)
                collected = sum(raw.get(("q", node, vehicle, period), 0.0) for vehicle in self.params.vehicles)
                tokens.append(
                    self._feature(
                        4,
                        _safe(self.params.producers.index(i) + 1, len(self.params.producers)),
                        _safe(self.params.waste_types.index(s) + 1, len(self.params.waste_types)),
                        _safe(period, self.max_period),
                        assigned,
                        _safe(self.params.generation[i, s, period], self.max_generation),
                        _safe(available, self.max_capacity),
                        _safe(collected, self.max_capacity),
                        _safe(remaining, self.max_capacity),
                        self.params.waste_consequence[s] / max(self.params.waste_consequence.values()),
                        self.params.producer_inventory_risk[i, s] / max(self.params.producer_inventory_risk.values()),
                    )
                )

        if self.use_facility_tokens:
            for j in self.params.facilities:
                for s in self.params.waste_types:
                    for period in self.params.periods:
                        cap = self.params.processing_capacity[j, s, period]
                        used_processing = raw.get(("p", j, s, period), 0.0)
                        received = raw.get(("R", j, s, period), 0.0)
                        before_inventory = raw.get(("BD", j, s, period), 0.0)
                        after_inventory = raw.get(("ID", j, s, period), 0.0)
                        tokens.append(
                            self._feature(
                                5,
                                _safe(self.params.facilities.index(j) + 1, len(self.params.facilities)),
                                _safe(self.params.waste_types.index(s) + 1, len(self.params.waste_types)),
                                _safe(period, self.max_period),
                                float(self.params.technology[j, s]),
                                _safe(cap, self.max_capacity),
                                _safe(received, self.max_capacity),
                                _safe(before_inventory, self.max_capacity),
                                _safe(used_processing, max(cap, 1.0)),
                                _safe(after_inventory, self.max_capacity),
                                self.params.processing_cost[j, s] / max(self.params.processing_cost.values()),
                                self.params.facility_inventory_risk[j, s] / max(self.params.facility_inventory_risk.values()),
                            )
                        )

        if self.use_route_arc_tokens:
            for (vehicle, period), route in sorted(plan.items(), key=lambda x: (x[0][1], x[0][0])):
                load = 0.0
                type_count = 0
                seen_types = set()
                for a, b in zip(route, route[1:]):
                    if a in self.params.pickup_nodes:
                        load += raw.get(("q", a, vehicle, period), 0.0)
                        seen_types.add(pickup_type(a))
                        type_count = len(seen_types)
                    if (a, b) not in self.params.distance:
                        continue
                    tokens.append(
                        self._feature(
                            6,
                            _safe(self.params.vehicles.index(vehicle) + 1, len(self.params.vehicles)),
                            _safe(period, self.max_period),
                            _safe(self.params.distance[a, b], self.max_distance),
                            self.params.accident_probability[a, b] / max(self.params.accident_probability.values()),
                            _safe(load, self.params.vehicle_capacity),
                            _safe(type_count, max(1, len(self.params.waste_types))),
                        )
                    )

        if len(tokens) > self.max_tokens:
            raise ValueError(
                f"state requires {len(tokens)} tokens but max_tokens={self.max_tokens}; "
                "increase max_tokens instead of silently truncating the state"
            )
        mask = [1.0] * len(tokens)
        while len(tokens) < self.max_tokens:
            tokens.append([0.0] * TOKEN_FEATURES)
            mask.append(0.0)
        global_vec = torch.tensor(global_features, dtype=torch.float32)
        return torch.tensor(tokens, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32), global_vec

    @staticmethod
    def _feature(token_type: int, *values: float) -> List[float]:
        feature = [0.0] * TOKEN_FEATURES
        feature[0] = float(token_type) / 10.0
        for idx, value in enumerate(values[: TOKEN_FEATURES - 1], start=1):
            feature[idx] = float(value)
        return feature


class PPOPolicy(nn.Module):
    def __init__(self, network_config: Dict[str, Any]):
        super().__init__()
        d_model = int(network_config["embedding_dim"])
        heads = int(network_config["attention_heads"])
        layers = int(network_config["transformer_layers"])
        hidden = int(network_config["ff_hidden_dim"])
        dropout = float(network_config.get("dropout", 0.0))
        self.network_type = str(network_config.get("network_type", "full_transformer"))
        self.operator_count = int(network_config.get("operator_count", len(OPERATORS)))
        self.object_count = int(network_config.get("object_count", 128))
        self.token_type_count = 7
        self.input_layers = nn.ModuleList(nn.Linear(TOKEN_FEATURES, d_model) for _ in range(self.token_type_count))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=hidden, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.global_mlp = nn.Sequential(nn.Linear(TOKEN_FEATURES, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        fused_dim = d_model if self.network_type == "full_transformer" else d_model * 2
        self.operator_head = nn.Linear(fused_dim, self.operator_count)
        self.object1_head = nn.Linear(fused_dim, self.object_count)
        self.object2_head = nn.Linear(fused_dim, self.object_count)
        self.object3_head = nn.Linear(fused_dim, self.object_count)
        # Keep cost and risk values separate so sparse risk improvements are not
        # hidden inside a single scalar critic target.
        self.value_head = nn.Linear(fused_dim, 2)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, global_vec: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        type_ids = torch.clamp((tokens[..., 0] * 10.0).round().long(), 0, self.token_type_count - 1)
        x = torch.zeros(*tokens.shape[:-1], self.input_layers[0].out_features, device=tokens.device, dtype=tokens.dtype)
        for token_type, input_layer in enumerate(self.input_layers):
            type_mask = type_ids == token_type
            if type_mask.any():
                x[type_mask] = input_layer(tokens[type_mask])
        key_padding_mask = mask <= 0.0
        encoded = self.transformer(x, src_key_padding_mask=key_padding_mask)
        weights = mask.unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        if self.network_type == "hybrid_transformer_fc":
            global_emb = self.global_mlp(global_vec)
            pooled = torch.cat([pooled, global_emb], dim=-1)
        logits = [
            self.operator_head(pooled),
            self.object1_head(pooled),
            self.object2_head(pooled),
            self.object3_head(pooled),
        ]
        value = self.value_head(pooled)
        return logits, value


class PPOImprover:
    def __init__(
        self,
        params: ModelParams,
        algorithm_config: Dict[str, Any],
        network_config: Dict[str, Any],
        seed: int = 0,
        *,
        trainable: bool = True,
    ):
        self.params = params
        self.algorithm_config = algorithm_config
        self.network_config = network_config
        self.seed = seed
        self.rng = random.Random(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        # --- GPU device setup ---
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")

        self._frozen_state_sha256: str | None = None

        configured_operators = int(network_config.get("operator_count", len(OPERATORS)))
        if configured_operators != len(OPERATORS):
            raise ValueError(
                f"network operator_count={configured_operators} does not match {len(OPERATORS)} operators"
            )
        required_objects = _required_object_capacity(params)
        if int(network_config.get("object_count", 128)) < required_objects:
            raise ValueError(
                f"object_count must be at least {required_objects} for this instance"
            )

        self.encoder = StateEncoder(params, network_config)
        self.policy = PPOPolicy(network_config).to(self.device)
        self._is_frozen = not trainable
        self.optimizer = (
            optim.Adam(
                self.policy.parameters(),
                lr=float(algorithm_config["ppo"]["learning_rate"]),
            )
            if trainable
            else None
        )

        if self.device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            param_count = sum(p.numel() for p in self.policy.parameters())
            print(f"[PPOImprover] GPU: {gpu_name} ({gpu_mem:.1f} GB) | model params: {param_count:,}")

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    def policy_state_sha256(self) -> str:
        """Return a stable hash of the policy state, independent of its device."""
        digest = hashlib.sha256()
        for name, tensor in sorted(self.policy.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def from_frozen_checkpoint(
        cls,
        params: ModelParams,
        checkpoint_path: str | Path,
        *,
        algorithm_config: Dict[str, Any] | None = None,
        network_config: Dict[str, Any] | None = None,
        seed: int = 0,
    ) -> "PPOImprover":
        """Create an inference-only improver for ``params`` from one checkpoint.

        Checkpoints without config metadata remain supported when both configs are
        supplied explicitly. A legacy checkpoint without an ``operators`` entry is
        interpreted using the current operator order, matching ``load_checkpoint``.
        """
        checkpoint = torch.load(
            Path(checkpoint_path), map_location="cpu", weights_only=False
        )
        _validate_checkpoint_operator_vocabulary(checkpoint)
        checkpoint_network = checkpoint.get("network_config")
        resolved_algorithm = (
            algorithm_config
            if algorithm_config is not None
            else checkpoint.get("algorithm_config")
        )
        resolved_network = (
            network_config if network_config is not None else checkpoint_network
        )
        if resolved_algorithm is None:
            raise ValueError(
                "legacy checkpoint has no algorithm_config; provide algorithm_config explicitly"
            )
        if resolved_network is None:
            raise ValueError(
                "legacy checkpoint has no network_config; provide network_config explicitly"
            )
        if network_config is not None and checkpoint_network is not None:
            requested_semantics = _network_semantics(network_config)
            checkpoint_semantics = _network_semantics(checkpoint_network)
            mismatches = [
                field
                for field in _NETWORK_CONFIG_CASTERS
                if requested_semantics[field] != checkpoint_semantics[field]
            ]
            if mismatches:
                details = ", ".join(
                    f"{field}: checkpoint={checkpoint_semantics[field]!r}, "
                    f"requested={requested_semantics[field]!r}"
                    for field in mismatches
                )
                raise ValueError(f"checkpoint network_config mismatch ({details})")

        max_tokens = int(resolved_network.get("max_tokens", 512))
        required_tokens = StateEncoder.required_token_capacity(params, resolved_network)
        if max_tokens < required_tokens:
            raise ValueError(
                f"checkpoint max_tokens={max_tokens} cannot encode the target instance; "
                f"requires at least {required_tokens}"
            )

        improver = cls(
            params,
            dict(resolved_algorithm),
            dict(resolved_network),
            seed=seed,
            trainable=False,
        )
        improver.load_checkpoint(checkpoint_path)
        improver.policy.requires_grad_(False)
        improver.policy.eval()
        return improver

    def _sample_preference(self) -> Tuple[float, float]:
        cfg = self.algorithm_config["ppo"].get("preference_sampling", {})
        endpoint_probability = float(cfg.get("endpoint_probability_each", 0.0))
        draw = self.rng.random()
        if draw < endpoint_probability:
            return 1.0, 0.0
        if draw < 2.0 * endpoint_probability:
            return 0.0, 1.0
        concentration = float(cfg.get("beta_concentration", 1.0))
        risk_weight = self.rng.betavariate(concentration, concentration)
        return 1.0 - risk_weight, risk_weight

    def load_checkpoint(self, path: str | Path, load_optimizer: bool = False) -> Dict[str, Any]:
        if load_optimizer and self.optimizer is None:
            raise RuntimeError("frozen PPOImprover cannot load optimizer state")
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        _validate_checkpoint_operator_vocabulary(checkpoint)
        self.policy.load_state_dict(checkpoint["state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.policy.eval()
        if self.is_frozen:
            self.policy.requires_grad_(False)
            self._frozen_state_sha256 = self.policy_state_sha256()
        return checkpoint

    def _action_from_logits(self, logits: List[torch.Tensor]) -> Tuple[OperatorAction, torch.Tensor, torch.Tensor]:
        dists = [torch.distributions.Categorical(logits=item) for item in logits]
        samples = [dist.sample() for dist in dists]
        logprob = sum(dist.log_prob(sample) for dist, sample in zip(dists, samples))
        entropy = sum(dist.entropy() for dist in dists)
        return OperatorAction(*(int(sample.item()) for sample in samples)), logprob, entropy

    def _logprob_entropy(self, logits: List[torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logprob = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.float32)
        entropy = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.float32)
        for idx, item in enumerate(logits):
            dist = torch.distributions.Categorical(logits=item)
            logprob = logprob + dist.log_prob(actions[:, idx])
            entropy = entropy + dist.entropy()
        return logprob, entropy

    # ------------------------------------------------------------------
    # Micro-batch size calculator (based on VRAM headroom)
    # ------------------------------------------------------------------
    def _compute_micro_batch_size(self, safety_margin: float = 0.85) -> int:
        """Find the largest micro-batch that fits in available GPU memory."""
        if self.device.type != "cuda":
            return 64  # CPU fallback

        d_model = int(self.network_config["embedding_dim"])
        layers = int(self.network_config["transformer_layers"])
        heads = int(self.network_config["attention_heads"])
        ff = int(self.network_config["ff_hidden_dim"])
        max_tokens = int(self.network_config.get("max_tokens", 256))

        per_layer_bytes = (
            3 * max_tokens * d_model * 4
            + heads * max_tokens * max_tokens * 4
            + max_tokens * d_model * 4 * 2
            + max_tokens * ff * 4
            + max_tokens * d_model * 4 * 2
        )
        per_sample_bytes = layers * per_layer_bytes * 3

        free_bytes = torch.cuda.mem_get_info()[0]
        max_samples = int(free_bytes * safety_margin / max(per_sample_bytes, 1))
        return max(16, min(max_samples, 2048))

    # ------------------------------------------------------------------
    # Parallel-episode state container
    # ------------------------------------------------------------------
    @dataclass
    class _EpState:
        plan: RoutePlan
        pref: Tuple[float, float]
        old_eval: Dict[str, Any]
        cost_scale: float
        risk_scale: float
        no_improve: int
        best_obj: float
        ep_seed: int

    # ------------------------------------------------------------------
    # Training  — 30 parallel episodes, batched GPU forward per step
    # ------------------------------------------------------------------
    def train(self, save_path: str | Path | None = None) -> Dict[str, List[float]]:
        if self.is_frozen:
            raise RuntimeError("frozen PPOImprover cannot be trained")
        if self.optimizer is None:
            raise RuntimeError("trainable PPOImprover has no optimizer")
        self.policy.train()
        cfg = self.algorithm_config["ppo"]
        history: Dict[str, List[float]] = {"episode_reward": [], "best_objective": []}
        gamma = float(cfg["gamma"])
        lam = float(cfg["gae_lambda"])
        train_iterations = int(cfg["train_iterations"])
        episode_steps = int(cfg["episode_steps"])
        num_parallel = int(cfg.get("num_parallel_episodes", 30))
        update_epochs = int(cfg["update_epochs"])
        clip_ratio = float(cfg["clip_ratio"])
        value_coef = float(cfg["value_coef"])
        entropy_coef = float(cfg["entropy_coef"])

        effective_batch = num_parallel * episode_steps
        micro_batch_size = self._compute_micro_batch_size()
        num_micro_batches = max(1, (effective_batch + micro_batch_size - 1) // micro_batch_size)

        if self.device.type == "cuda":
            # CUDA flash and memory-efficient SDP backward kernels can be
            # nondeterministic. Training uses the math kernel so a recorded
            # seed identifies one repeatable update path on the same stack.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            print(
                f"[PPOImprover] train: {train_iterations} iters × {num_parallel} parallel eps × {episode_steps} steps"
            )
            print(
                f"  effective_batch={effective_batch}  micro_batch={micro_batch_size}"
                f"  num_micro={num_micro_batches}  update_epochs={update_epochs}"
                f"  free_vram={free_gb:.1f}GB"
            )

        for iteration in range(train_iterations):
            # ==== Phase 1: initialise N parallel episodes ====
            ep_states: List[PPOImprover._EpState] = []
            for i in range(num_parallel):
                ep_seed = self.seed + iteration * num_parallel + i
                plan = build_greedy_initial_plan(self.params, seed=ep_seed)
                pref = self._sample_preference()
                solution = route_plan_to_solution(self.params, plan)
                init_eval = evaluate_solution(self.params, solution, pref, check_constraints=False)
                ep_states.append(PPOImprover._EpState(
                    plan=plan, pref=pref, old_eval=init_eval,
                    cost_scale=max(init_eval["cost"], 1.0),
                    risk_scale=max(init_eval["risk"], 1.0),
                    no_improve=0, best_obj=1.0,
                    ep_seed=ep_seed,
                ))

            # Per-episode trajectory buffers (CPU)
            traj_tokens: List[List[torch.Tensor]] = [[] for _ in range(num_parallel)]
            traj_mask: List[List[torch.Tensor]]   = [[] for _ in range(num_parallel)]
            traj_global: List[List[torch.Tensor]] = [[] for _ in range(num_parallel)]
            traj_actions: List[List[torch.Tensor]]  = [[] for _ in range(num_parallel)]
            traj_logprob: List[List[torch.Tensor]]  = [[] for _ in range(num_parallel)]
            traj_rewards: List[List[torch.Tensor]]  = [[] for _ in range(num_parallel)]
            traj_values: List[List[torch.Tensor]]   = [[] for _ in range(num_parallel)]

            # ==== Phase 2: step all episodes together ====
            for step in range(episode_steps):
                # --- 2a: encode all N states on CPU, then stack for GPU ---
                cpu_tokens: List[torch.Tensor] = []
                cpu_masks: List[torch.Tensor] = []
                cpu_globals: List[torch.Tensor] = []
                for i, es in enumerate(ep_states):
                    t, m, g = self.encoder.encode(
                        es.plan, es.pref,
                        step / max(1, episode_steps),
                        es.no_improve / max(1, episode_steps),
                    )
                    cpu_tokens.append(t)
                    cpu_masks.append(m)
                    cpu_globals.append(g)

                tokens_dev = torch.stack(cpu_tokens).to(self.device)    # [N, max_tok, 18]
                masks_dev = torch.stack(cpu_masks).to(self.device)      # [N, max_tok]
                globals_dev = torch.stack(cpu_globals).to(self.device)  # [N, 18]

                # --- 2b: single batched GPU forward for all N episodes ---
                logits_b, values_b = self.policy(tokens_dev, masks_dev, globals_dev)
                # logits_b: list of 4 tensors, each [N, out_dim]
                # values_b: [N]

                # --- 2c: per-episode action sampling & environment step (CPU) ---
                for i, es in enumerate(ep_states):
                    logits_i = [logits_b[0][i], logits_b[1][i], logits_b[2][i], logits_b[3][i]]
                    value_i = values_b[i]
                    action, logprob, _ = self._action_from_logits(logits_i)

                    rng = random.Random(es.ep_seed + step)
                    candidate, ok, _ = apply_operator(self.params, es.plan, action, seed=rng.randint(0, 1_000_000))
                    new_solution = route_plan_to_solution(self.params, candidate if ok else es.plan)
                    new_eval = evaluate_solution(self.params, new_solution, es.pref, check_constraints=False)

                    old_components = torch.tensor(
                        [es.old_eval["cost"] / es.cost_scale, es.old_eval["risk"] / es.risk_scale],
                        dtype=torch.float32,
                    )
                    new_components = torch.tensor(
                        [new_eval["cost"] / es.cost_scale, new_eval["risk"] / es.risk_scale],
                        dtype=torch.float32,
                    )
                    preference_tensor = torch.tensor(es.pref, dtype=torch.float32)
                    old_norm = float((old_components * preference_tensor).sum())
                    new_norm = float((new_components * preference_tensor).sum())

                    if ok:
                        reward = old_components - new_components
                        if torch.all(torch.abs(reward) < 1e-9):
                            reward -= float(cfg["no_change_penalty"])
                        es.plan = candidate
                        es.old_eval = new_eval
                        es.no_improve = 0 if new_norm < old_norm - 1e-9 else es.no_improve + 1
                    else:
                        reward = torch.full((2,), -float(cfg["repair_failure_penalty"]), dtype=torch.float32)
                        es.no_improve += 1

                    es.best_obj = min(es.best_obj, new_norm if ok else old_norm)

                    # Store trajectory (CPU tensors)
                    traj_tokens[i].append(cpu_tokens[i])
                    traj_mask[i].append(cpu_masks[i])
                    traj_global[i].append(cpu_globals[i])
                    traj_actions[i].append(torch.tensor(
                        [action.operator_id, action.object_1, action.object_2, action.object_3],
                        dtype=torch.long))
                    traj_logprob[i].append(logprob.detach().cpu())
                    traj_rewards[i].append(reward)
                    traj_values[i].append(value_i.detach().cpu())

            # ==== Phase 3: stack all episode trajectories → GPU batch ====
            episodes_data = []
            iter_best_obj = float("inf")
            iter_total_reward = 0.0
            for i in range(num_parallel):
                ep_dict = {
                    "tokens": torch.stack(traj_tokens[i]),
                    "mask": torch.stack(traj_mask[i]),
                    "global": torch.stack(traj_global[i]),
                    "actions": torch.stack(traj_actions[i]),
                    "logprob": torch.stack(traj_logprob[i]),
                    "rewards": torch.stack(traj_rewards[i]),
                    "values": torch.stack(traj_values[i]),
                    "preferences": torch.tensor([ep_states[i].pref] * episode_steps, dtype=torch.float32),
                }
                episodes_data.append(ep_dict)
                iter_best_obj = min(iter_best_obj, ep_states[i].best_obj)
                pref_tensor = torch.tensor(ep_states[i].pref, dtype=torch.float32)
                iter_total_reward += float(sum((reward * pref_tensor).sum().item() for reward in traj_rewards[i]))

            tokens_b = torch.cat([e["tokens"] for e in episodes_data]).to(self.device)
            mask_b = torch.cat([e["mask"] for e in episodes_data]).to(self.device)
            global_b = torch.cat([e["global"] for e in episodes_data]).to(self.device)
            actions_b = torch.cat([e["actions"] for e in episodes_data]).to(self.device)
            old_logprob_b = torch.cat([e["logprob"] for e in episodes_data]).to(self.device)
            preferences_b = torch.cat([e["preferences"] for e in episodes_data]).to(self.device)

            # ==== Phase 4: per-episode GAE ====
            advantages_list: List[torch.Tensor] = []
            returns_list: List[torch.Tensor] = []
            for e in episodes_data:
                r = e["rewards"].to(self.device)
                v = e["values"].to(self.device)
                ret, adv = self._gae(r, v, gamma, lam)
                advantages_list.append(adv)
                returns_list.append(ret)
            advantages_vector = torch.cat(advantages_list)
            returns = torch.cat(returns_list)
            advantages_vector = (
                advantages_vector - advantages_vector.mean(dim=0, keepdim=True)
            ) / advantages_vector.std(dim=0, keepdim=True).clamp_min(1e-6)
            advantages = (advantages_vector * preferences_b).sum(dim=-1)
            advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)

            # ==== Phase 5: PPO update with gradient accumulation ====
            total_t = tokens_b.shape[0]
            actual_micro = min(micro_batch_size, total_t)
            actual_num_micro = max(1, (total_t + actual_micro - 1) // actual_micro)

            for _ in range(update_epochs):
                perm = torch.randperm(total_t, device=self.device)
                for micro_step in range(actual_num_micro):
                    start = micro_step * actual_micro
                    end = min(start + actual_micro, total_t)
                    idx = perm[start:end]

                    logits, values = self.policy(tokens_b[idx], mask_b[idx], global_b[idx])
                    logprob, entropy = self._logprob_entropy(logits, actions_b[idx])
                    ratio = torch.exp(logprob - old_logprob_b[idx])
                    adv_micro = advantages[idx]
                    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_micro
                    policy_loss = -torch.min(ratio * adv_micro, clipped).mean()
                    value_loss = (returns[idx] - values).pow(2).mean()
                    entropy_loss = -entropy.mean()
                    loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
                    (loss / actual_num_micro).backward()

                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            history["episode_reward"].append(iter_total_reward / num_parallel)
            history["best_objective"].append(float(iter_best_obj))

            if self.device.type == "cuda" and iteration % 10 == 9:
                torch.cuda.empty_cache()

        if save_path is not None:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
                    "state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "network_config": self.network_config,
                    "algorithm_config": self.algorithm_config,
                    "seed": self.seed,
                    "operators": list(OPERATORS),
                },
                path,
            )

        return history

    @staticmethod
    def _gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        lastgaelam = torch.zeros(rewards.shape[-1], device=rewards.device)
        next_value = torch.zeros(rewards.shape[-1], device=rewards.device)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value - values[t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages[t] = lastgaelam
            next_value = values[t]
        returns = advantages + values
        return returns.detach(), advantages.detach()

    # ------------------------------------------------------------------
    # Inference-time improvement  (GPU, batch=1 + multi-sample)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def improve(
        self,
        preference: Tuple[float, float],
        steps: int | None = None,
        seed: int | None = None,
        initial_plan: RoutePlan | None = None,
    ) -> PPOEvalResult:
        self.policy.eval()
        before_state_hash: str | None = None
        if self.is_frozen:
            before_state_hash = self.policy_state_sha256()
            if (
                self._frozen_state_sha256 is not None
                and before_state_hash != self._frozen_state_sha256
            ):
                raise RuntimeError(
                    "frozen PPO policy state changed after checkpoint loading"
                )
        cfg = self.algorithm_config["ppo"]
        limit = int(steps or cfg["eval_steps"])
        candidate_samples = int(cfg.get("eval_candidate_samples", 8))
        eval_seed = self.seed if seed is None else seed
        torch.manual_seed(eval_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed)
        plan = (
            {key: list(route) for key, route in initial_plan.items()}
            if initial_plan is not None
            else build_greedy_initial_plan(self.params, seed=eval_seed)
        )
        best_plan = {key: list(route) for key, route in plan.items()}
        best_solution = route_plan_to_solution(self.params, best_plan)
        best_eval = evaluate_solution(self.params, best_solution, preference, check_constraints=False)
        cost_scale = max(best_eval["cost"], 1.0)
        risk_scale = max(best_eval["risk"], 1.0)
        history = [dict(best_eval)]
        trace: List[Dict[str, Any]] = []
        no_improve = 0
        rng = random.Random(eval_seed)

        def scalar(metrics: Dict[str, Any]) -> float:
            return preference[0] * metrics["cost"] / cost_scale + preference[1] * metrics["risk"] / risk_scale

        for step in range(limit):
            before_plan_hash = plan_sha256(plan)
            tokens, mask, global_vec = self.encoder.encode(
                plan, preference,
                step / max(1, limit),
                no_improve / max(1, limit),
            )
            tokens_dev = tokens.to(self.device)
            mask_dev = mask.to(self.device)
            global_dev = global_vec.to(self.device)

            logits, _ = self.policy(
                tokens_dev.unsqueeze(0),
                mask_dev.unsqueeze(0),
                global_dev.unsqueeze(0),
            )
            squeezed = [item.squeeze(0) for item in logits]

            best_candidate = None
            best_candidate_eval = None
            best_candidate_score = float("inf")
            best_action: OperatorAction | None = None
            best_operator_seed: int | None = None
            for sample_idx in range(candidate_samples):
                if sample_idx == 0:
                    action = OperatorAction(*(int(torch.argmax(item).item()) for item in squeezed))
                else:
                    action, _, _ = self._action_from_logits(squeezed)
                operator_seed = rng.randint(0, 2**31 - 1)
                candidate, ok, _ = apply_operator(self.params, plan, action, seed=operator_seed)
                if not ok:
                    continue
                eval_now = evaluate_solution(self.params, route_plan_to_solution(self.params, candidate), preference, check_constraints=False)
                score = scalar(eval_now)
                if score < best_candidate_score:
                    best_candidate = candidate
                    best_candidate_eval = eval_now
                    best_candidate_score = score
                    best_action = action
                    best_operator_seed = operator_seed

            if best_candidate is None or best_candidate_eval is None:
                no_improve += 1
                history.append(dict(best_eval))
                trace.append(
                    {
                        "step": step,
                        "accepted": False,
                        "reason": "no_valid_candidate",
                        "evaluation_seed": eval_seed,
                        "before_plan_sha256": before_plan_hash,
                        "after_plan_sha256": before_plan_hash,
                    }
                )
                continue

            incumbent_score = scalar(best_eval)
            plan = best_candidate
            accepted = best_candidate_score < incumbent_score - 1e-12
            if accepted:
                best_eval = best_candidate_eval
                best_plan = {key: list(route) for key, route in plan.items()}
                no_improve = 0
            else:
                no_improve += 1
            history.append(dict(best_candidate_eval))
            trace.append(
                {
                    "step": step,
                    "operator": OPERATORS[best_action.operator_id % len(OPERATORS)] if best_action else None,
                    "objects": [
                        best_action.object_1,
                        best_action.object_2,
                        best_action.object_3,
                    ] if best_action else None,
                    "operator_seed": best_operator_seed,
                    "evaluation_seed": eval_seed,
                    "accepted": accepted,
                    "candidate_cost": best_candidate_eval["cost"],
                    "candidate_risk": best_candidate_eval["risk"],
                    "candidate_objective": best_candidate_score,
                    "incumbent_objective_before": incumbent_score,
                    "before_plan_sha256": before_plan_hash,
                    "after_plan_sha256": plan_sha256(plan),
                    "incumbent_plan_sha256": plan_sha256(best_plan),
                }
            )

        best_solution = route_plan_to_solution(self.params, best_plan)
        result = PPOEvalResult(
            solution=best_solution,
            plan=best_plan,
            history=history,
            trace=trace,
        )
        if before_state_hash is not None:
            after_state_hash = self.policy_state_sha256()
            if after_state_hash != before_state_hash:
                raise RuntimeError("frozen PPO policy state changed during inference")
        return result
