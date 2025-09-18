import torch
import torch.nn as nn

from misc import *

class ACGT_LM(nn.Module):
    def __init__(self, d_model=64, n_layers=4, n_heads=4, block_size=128):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.ModuleList([
            self._build_block(d_model, n_heads)
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

    def _build_block(self, d_model, n_heads):
        return nn.ModuleDict({
            "self_attention": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
            "mlp": nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model)
            ),
            "norm1": nn.LayerNorm(d_model),
            "norm2": nn.LayerNorm(d_model)
        })

    def forward(self, idx):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block size {self.block_size}.")

        token_emb = self.token_embedding(idx)
        position_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = token_emb + position_emb

        for block in self.blocks:
            x_norm = block["norm1"](x)
            causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=idx.device), diagonal=1)
            
            attn_output, _ = block["self_attention"](
                x_norm, x_norm, x_norm,
                attn_mask=causal_mask
            )
            
            x = x + attn_output
            x_norm = block["norm2"](x)
            x = x + block["mlp"](x_norm)

        x = self.final_norm(x)
        logits = self.output_head(x)

        return logits