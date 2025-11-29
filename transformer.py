import torch
from torch import nn

from attention import MultiHeadAttention

class TransformerBlock(nn.Module):

    def __init__(self,
                num_heads: int,
                context_size: int,
                in_out_size: int,
                hidden_size: int,
                atn_drop: float = 0.1,
                out_drop: float = 0.1,
                mlp_drop: float = 0.1,
                bias: bool = True):
        
        super().__init__()
        self.norm_atn = nn.LayerNorm(in_out_size)
        self.atn = MultiHeadAttention(
            in_out_size,
            num_heads,
            atn_drop=atn_drop,
            out_drop=out_drop,
            bias=bias,
            causal=True,
            context_size=context_size
        )

        self.norm_mlp = nn.LayerNorm(in_out_size)
        self.mlp = nn.Sequential(
            nn.Linear(in_out_size, hidden_size, bias=bias),
            nn.GELU(),
            nn.Linear(hidden_size, in_out_size, bias=bias),
            nn.Dropout(mlp_drop)
        )
    def forward(self, x: torch.Tensor):
        """
        Computes
        Y = X + Attention(Norm(X))
        Output = Y + MLP(Norm(Y))
        where 
        - Norm is layer normalization (Pre-Norm here)
        - Attention is the underlying attention
        - MLP is the underlying two layers MLP
        """
        y = x + self.atn(self.norm_atn(x))
        return y + self.mlp(self.norm_mlp(y))
    
if __name__ == "__main__":
    num_heads = 8
    batch_size = 32
    seq_len = 16
    context_size = seq_len
    feature_dim = 1024

    transformer = TransformerBlock(num_heads, context_size, feature_dim, 2*feature_dim)
    x =  torch.rand(batch_size, seq_len, feature_dim)
    transformer(x)
