import torch
from torch import nn

from transformers import TransformerBlock, SinusoidalPositionalEncoding
from my_utils import _init_weights


class Decoder(nn.Module):
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
        tie_weights: bool = True,
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
                    nn.GELU,
                    atn_drop,
                    out_drop,
                    mlp_drop,
                    cross=True,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )

        self.out = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.out.weight = self.vocab_embed.weight

        # [1, 1, tgt_len, tgt_len]
        causal_mask = torch.triu(
            torch.ones((max_seq_len, max_seq_len), dtype=torch.bool), diagonal=1
        )[None, None, :, :]

        self.register_buffer("causal_mask", causal_mask, persistent=True)

        # Initializes weight for all submodules.
        self.apply(lambda mod: _init_weights(mod, d_model))

    def forward(
        self,
        x: torch.Tensor,
        encoded: torch.Tensor,
        tgt_pad_mask: torch.BoolTensor,
        src_pad_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Input:
        - x has shape (batch_size, tgt_len)
        - encoded has shape (batch_size, src_len, d_model)
        - tgt_pad_mask has shape (batch_size, 1, 1, tgt_len)
        - src_pad_mask has shape (batch_size, 1, 1, src_len)

        Output:
        Returns logits of shape [batch_size, tgt_len, vocab_size]
        """

        _batch_size, tgt_len = x.shape
        tokens = self.vocab_embed(x)
        tokens_with_pos = self.pos_embed(tokens)

        x = self.embed_drop(tokens_with_pos)

        causal = self.causal_mask[:, :, :tgt_len, :tgt_len]
        tgt_mask = causal | tgt_pad_mask
        for block in self.transformers:
            x = block(x, encoded, tgt_mask, memory_mask=src_pad_mask)

        return self.out(x)
