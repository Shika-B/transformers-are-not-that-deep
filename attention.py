import torch
from torch import nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        atn_drop: float = 0.1,
        out_drop: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = self.d_model // self.num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=bias)
        self.Wk = nn.Linear(d_model, d_model, bias=bias)
        self.Wv = nn.Linear(d_model, d_model, bias=bias)
        self.Wo = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(atn_drop)
        self.out_drop = nn.Dropout(out_drop)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """
        Input:
        x and y are tensor with shapes (batch_size, sequence_length, d_model)
        mask is a boolean tensor of shape broadcastable* to (batch_size, num_heads, query_len, keyvalue_len)
            *See https://docs.pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics

        Output:
        If y is None, sets y = x

        Returns proj(concat(atn_i(x, y, mask))) where:
        - atn_i is the i-th Attention block (with 1 <= i <= self.num_heads) that returns
                softmax(Qx (Ky)^T / sqrt(d_k) - mask_inf) @ (Vy)
                    where:
                    - K, V, Q are learned matrices
                    - mask_inf is a (batch_size x seq_len) matrix with
                            mask_inf[i, j] = +oo if mask[i, j] else 0
                    - d_k is a variance normalization parameter equal to the number of columns in K.
                    - concat concatenates along the feature dimension
        - proj is a learnable linear layer
        """

        if y is None:
            y = x

        batch_size, q_len, _input_dim = x.shape
        batch_size, kv_len, _input_dim = y.shape

        # [batch_size, num_heads, q_len, d_head]
        q = (
            self.Wq(x)
            .reshape(batch_size, q_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        # Both [batch_size, num_heads, kv_len, d_head]
        k = (
            self.Wk(y)
            .reshape(batch_size, kv_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )
        v = (
            self.Wv(y)
            .reshape(batch_size, kv_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

        # [batch_size, num_heads, q_len, kv_len]
        # Here scores[b, h, i, j] measures how much query i attends to key j in the h-th head of the b-th element of the batch.
        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(k.size(-1))

        if mask is not None:
            # mask get broadcasted here
            scores = scores.masked_fill(mask, float("-inf"))

        # Softmax so that we get probability distributions associated to each query.
        attn = scores.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v

        x = x.transpose(1, 2).reshape(batch_size, q_len, self.d_model)
        return self.out_drop(self.Wo(x))


if __name__ == "__main__":
    pass
