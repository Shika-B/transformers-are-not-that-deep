import torch
from torch import nn

import math

from attention import MultiHeadAttention
from my_utils import _Proj1, _init_weights


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()

        # [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)
        # [d_model/2]
        # mathematically equivalent to
        # div_term = 10000**(torch.arange(0, d_model, 2) / d_model)
        # but avoids numerical instability
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer so it is not trainable (i.e not a parameter)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
        - x with shape [batch_size, seq_len, d_model]

        Output:
        x + PE
        where
            PE[b, s, 2i] = sin(s/10000^(2i/d_model))
            PE[b, s, 2i+1] = cos(s/10000^(2i/d_model))
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len]


class TransformerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        expand_size: int,
        activation: nn.GELU,
        atn_drop: float = 0.1,
        out_drop: float = 0.1,
        mlp_drop: float = 0.1,
        cross: bool = False,
        bias: bool = False,
    ):

        super().__init__()
        self.norm_atn = nn.LayerNorm(d_model)
        self.atn = MultiHeadAttention(
            d_model,
            num_heads,
            atn_drop=atn_drop,
            out_drop=out_drop,
            bias=bias,
        )

        self.cross = cross

        if self.cross:
            self.norm_cross_atn = nn.LayerNorm(d_model)
            self.cross_atn = MultiHeadAttention(
                d_model,
                num_heads,
                atn_drop=atn_drop,
                out_drop=out_drop,
                bias=bias,
            )
        else:
            self.norm_cross_atn = nn.Identity()
            self.cross_atn = _Proj1()

        self.norm_mlp = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, expand_size, bias=bias),
            activation(),
            nn.Linear(expand_size, d_model, bias=bias),
            nn.Dropout(mlp_drop),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        tgt_mask: torch.BoolTensor | None = None,
        memory_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """
        Input:
        - x is a Tensor with shape [batch_size, tgt_len, d_model]
        - y is a Tensor with shape [batch_size, src_len, d_model]
        - tgt_mask is a BoolTensor with shape broadcastable to
        [batch_size, num_heads, tgt_len, tgt_len]
        - memory_mask is a BoolTensor with shape broadcastable to
        [batch_size, num_heads, tgt_len, src_len]

        Output:
        z = x + Attention(Norm1(x), y, tgt_mask)
        w = z + CrossAttention(Norm2(z), y, memory_mask)
        Output = w + MLP(Norm3(w))

        where
        - x = x if y is None
        - Norm_i are learnable layer normalization layers
        - Attention is the underlying attention of the transformer block
        - if self.cross is True then applies the underlying cross attention layer
        otherwise Cross Attention returns the first parameter
        and Norm2 act as the identity
        - MLP is the underlying two layers perceptron
        """

        assert self.cross ^ (y is None)

        x = x + self.atn(self.norm_atn(x), mask=tgt_mask)
        x = x + self.cross_atn(self.norm_cross_atn(x), y, mask=memory_mask)

        return x + self.mlp(self.norm_mlp(x))


if __name__ == "__main__":
    pass
