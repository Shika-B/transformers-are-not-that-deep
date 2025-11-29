import torch
from torch import nn

from transformers import TransformerBlock, SinusoidalPositionalEncoding
from encoder import Encoder
from decoder import Decoder
from my_utils import _init_weights

class TranslateModel(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        d_model: int,
        max_seq_len: int,
        vocab_size: int,
        pad_id: int,
        embed_drop: float = 0.1,
        atn_drop: float = 0.1,
        out_drop: float = 0.1,
        mlp_drop: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()

        self.encoder = Encoder(
            num_layers,
            num_heads,
            d_model,
            max_seq_len,
            vocab_size,
            embed_drop,
            atn_drop,
            out_drop,
            mlp_drop,
            bias,
        )

        self.decoder = Decoder(
            num_layers,
            num_heads,
            d_model,
            max_seq_len,
            vocab_size,
            embed_drop,
            atn_drop,
            out_drop,
            mlp_drop,
            bias,
        )

        self.pad_id = pad_id
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Input:
        - src has shape (batch_size, src_len)
        - tgt has shape (batch_size, tgt_len)

        Output:
        Returns logits with shape (batch_size, tgt_len, vocab_size)
        """
        # [batch_size, 1, 1, src_len]
        src_pad_mask = (src == self.pad_id)[:, None, None, :]  
        # [batch_size, 1, 1, tgt_len]
        tgt_pad_mask = (tgt == self.pad_id)[:, None, None, :]  

        encoded = self.encoder(src, src_pad_mask)
        decoded = self.decoder(tgt, encoded, tgt_pad_mask, src_pad_mask)
        
        return decoded

