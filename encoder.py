import torch
from torch import nn

from transformers import TransformerBlock, SinusoidalPositionalEncoding
from my_utils import _init_weights


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        max_seq_len: int,
        vocab_size: int,
        embed_drop: float = 0.1,
        atn_drop: float = 0.1,
        out_drop: float = 0.1,
        mlp_drop: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()

        self.vocab_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEncoding(max_seq_len, d_model)

        self.embed_drop = nn.Dropout(embed_drop)

        self.transformers = nn.ModuleList(
            [
                TransformerBlock(
                    num_heads,
                    d_model,
                    4 * d_model,
                    atn_drop=atn_drop,
                    out_drop=out_drop,
                    mlp_drop=mlp_drop,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

        # Initializes weight for all submodules.
        self.apply(lambda mod: _init_weights(mod, d_model))

    def forward(self, x: torch.Tensor, pad_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Input:
        - x has shape (batch_size, src_len)
        - encoded has shape (batch_size, src_len, d_model)
        - pad_mask has shape (batch_size, 1, 1, tgt_len)

        Output:
        - Returns a tensor of shape [batch_size, src_len, d_model]
        """
        tokens = self.vocab_embed(x)

        tokens_with_pos = self.pos_embed(tokens)
        x = self.embed_drop(tokens_with_pos)

        for block in self.transformers:
            x = block(x, tgt_mask=pad_mask)
        return x
