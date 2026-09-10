"""User-requested critic capacity ablation; actor and Transformer unchanged.

Current shared context already includes GELU. This adds critic-private nonlinear
capacity, not an activation that constrains the final scalar value's range.
"""
from torch import nn
from src.ppo_objects_v8c import ObjectPolicy as PreviousPolicy


class ObjectPolicy(PreviousPolicy):
    def __init__(self,dimension=96,layers=1):
        super().__init__(dimension,layers)
        self.value=nn.Sequential(nn.Linear(dimension,dimension),nn.GELU(),nn.Linear(dimension,1))
