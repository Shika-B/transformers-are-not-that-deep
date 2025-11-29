import torch
from torch import nn

import math


def _init_weights(mod, d_model=None):
    """
    Following recommandations
    from [Attention Is All You Need](https://arxiv.org/pdf/1706.03762).
    """
    if isinstance(mod, nn.LayerNorm):
        nn.init.ones_(mod.weight)
        nn.init.zeros_(mod.bias)

    elif isinstance(mod, nn.Linear):
        nn.init.xavier_uniform_(mod.weight)
        if mod.bias is not None:
            nn.init.zeros_(mod.bias)

    elif isinstance(mod, nn.Embedding):
        if d_model is None:
            raise ValueError("Need to know d_model to initialization embedding layers")
        nn.init.normal_(mod.weight, mean=0.0, std=1.0 / math.sqrt(d_model))


class _Proj1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, *args, **kwargs):
        return x1
