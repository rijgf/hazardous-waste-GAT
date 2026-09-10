"""Conditional full-object policy: one Transformer, three separate 512 heads.

Later heads see sampled earlier arguments and raw directed edge attributes.
No objective-ranked candidates, repair, or non-neural argument selection.
"""
import numpy as np
import torch
from torch import nn
from src.ppo_objects_v7 import ObjectPolicy as PreviousPolicy, OBJECTS, DUMMY, OPS, Action


class ObjectPolicy(PreviousPolicy):
    def __init__(self, dimension=96, layers=1):
        if layers != 1:
            raise ValueError('Exactly one Transformer layer is permitted')
        super().__init__(dimension, layers)
        self.op_embedding = nn.Embedding(len(OPS), dimension)
        self.queries = nn.ModuleList([nn.Sequential(nn.Linear(4*dimension, dimension), nn.GELU()) for _ in range(3)])
        self.keys = nn.ModuleList([nn.Linear(dimension, dimension, bias=False) for _ in range(3)])
        self.pairs = nn.ModuleList([nn.Sequential(nn.Linear(9, 32), nn.GELU(), nn.Linear(32, 1)) for _ in range(2)])
        for pair in self.pairs:
            nn.init.zeros_(pair[-1].bias)
            nn.init.orthogonal_(pair[-1].weight, gain=.01)

    def encode(self, features, links, edges, interaction):
        batch, count, _ = features.shape
        nodes = self.edge_encoder(edges.flatten(-2))
        nodes = torch.cat([nodes, nodes.new_zeros(batch, 1, self.dimension)], dim=1)
        linked = nodes[torch.arange(batch, device=nodes.device)[:, None, None], links]
        h = self.features(features) + self.link_projection(linked.flatten(-2))
        h = h + self.slot_embedding(torch.arange(count, device=h.device))[None] + self.interaction(interaction)[:, None]
        h = self.norm(self.transformer(h))
        global_h = self.context(torch.cat([h.mean(1), features[:, 0, 54:59]], dim=-1))
        # Pad the physical-node matrix on BOTH axes, including the missing-node
        # link. Padding is zero; unavailable edges retain their explicit flag.
        physical_count = edges.shape[1]
        matrix = nn.functional.pad(edges[:, :, :physical_count], (0, 0, 0, 1, 0, 1))
        return h, global_h, links, matrix

    def head_logits(self, encoded, chosen):
        h, global_h, links, edges = encoded
        if chosen.shape[1] == 0:
            return self.operator(global_h)
        index = chosen.shape[1] - 1
        batch, count, _ = h.shape
        zero = h.new_zeros(batch, self.dimension)
        padded = nn.functional.pad(h, (0, 0, 0, OBJECTS-count))
        rows = torch.arange(batch, device=h.device)
        first = padded[rows, chosen[:, 1]] if index >= 1 else zero
        second = padded[rows, chosen[:, 2]] if index >= 2 else zero
        query = self.queries[index](torch.cat([global_h, self.op_embedding(chosen[:, 0]), first, second], dim=-1))
        local = self.local_heads[index](h).squeeze(-1)
        local = local + (self.keys[index](h) * query[:, None]).sum(-1) / self.dimension**.5
        if index >= 1:
            missing = edges.shape[1]-1
            padded_links = nn.functional.pad(links, (0, 0, 0, OBJECTS-count), value=missing)
            selected_node = padded_links[rows, chosen[:, 1], 0]
            own, successor = links[:, :, 0], links[:, :, 2]
            # For a route-slot insertion, its anchor is the depot and its next
            # node is links[1]; pickup-slot insertion uses the pickup successor.
            # Route slots have all four links and own==home, unlike pickups.
            route_like = (links[:, :, 0] == links[:, :, 3]) & (links[:, :, 0] != missing)
            successor = torch.where(route_like, links[:, :, 1], successor)
            r = rows[:, None]
            selected = selected_node[:, None]
            pair = torch.cat([edges[r, selected, own], edges[r, own, selected],
                              edges[r, selected, successor]], dim=-1)
            local = local + self.pairs[index-1](pair).squeeze(-1)
        return self.heads[index](query) + nn.functional.pad(local, (0, OBJECTS-count))

    def forward(self, features, links, edges, interaction, actions):
        encoded = self.encode(features, links, edges, interaction)
        return [self.head_logits(encoded, actions[:, :i]) for i in range(4)], self.value(encoded[1]).squeeze(-1)


def sample_batch(policy, encoded, states, plans, rng, temperature=1., object_ablation='none'):
    """Teacher forcing and rollout use exactly the same conditional logits."""
    device = encoded[0].device
    chosen = np.zeros((len(states), 0), dtype=np.int64)
    all_masks = [[] for _ in states]; logprob = np.zeros(len(states))
    for index in range(4):
        scores = policy.head_logits(encoded, torch.tensor(chosen, device=device)).detach().cpu().numpy() / temperature
        if object_ablation == 'uniform' and index:
            scores.fill(0.)
        values = []
        for row, (state, plan) in enumerate(zip(states, plans)):
            mask = state.masks(plan, chosen[row].tolist())
            if not mask.any():
                mask = state.mask([DUMMY])
            allowed = np.flatnonzero(mask); weights = np.asarray(scores[row, allowed], dtype=np.float64)
            weights = np.exp(weights-weights.max()); weights /= weights.sum()
            choice = int(rng.choice(len(allowed), p=weights)); values.append(int(allowed[choice]))
            logprob[row] += float(np.log(max(weights[choice], 1e-300))); all_masks[row].append(mask)
        chosen = np.column_stack([chosen, values])
    return [(Action(*map(int, choices)), masks, float(lp)) for choices, masks, lp in zip(chosen, all_masks, logprob)]
