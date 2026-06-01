from enum import Enum

import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


class PositionalEmbeddingType(Enum):
    LEARNED = "learned"
    SINUSOIDAL = "sinusoidal"
    ROPE = "rope"


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding module."""

    pe: torch.Tensor

    def __init__(self,
        seq_len: int,
        d_model: int,
    ):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

    def forward(self,
        x: torch.Tensor,
    ) -> torch.Tensor:
         return self.pe[:, :x.size(1), :]


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding module."""

    pe: torch.Tensor

    def __init__(self,
        d_model: int,
        max_len: int = 4096,
    ):
        if d_model % 2 != 0:
            raise ValueError(f"Cannot use sinusoidal positional encoding with odd d_model (got {d_model})")

        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class RoPEMultiheadAttention(nn.Module):
    """Multihead attention module with RoPE positional encoding."""
    
    def __init__(self,
        d_model: int,
        n_heads: int,
        dropout: float,
    ):
        super().__init__()


class EncoderLayer(nn.Module):
    """Transformer encoder layer that can return attention weights."""

    def __init__(self,
        emb_type: PositionalEmbeddingType,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()

        if emb_type == PositionalEmbeddingType.ROPE:
            self.self_attn = RoPEMultiheadAttention(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
            )
        else:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_output, attn_weights = self.self_attn(x, x, x,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        x = self.norm1(x + self.dropout_attn(attn_output))

        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout_ff(ff))

        if return_attention:
            return x, attn_weights

        return x, None


class CATransformer(nn.Module):
    """A transformer model for learning cellular automaton rules for a fixed sequence length."""
    
    def __init__(self,
        emb_type: PositionalEmbeddingType = PositionalEmbeddingType.SINUSOIDAL,
        seq_len: int | None = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(2, d_model)

        if emb_type == PositionalEmbeddingType.LEARNED:
            if seq_len is None:
                raise ValueError("seq_len must be specified for learned positional embedding")
            self.pos_emb = LearnablePositionalEncoding(seq_len=seq_len, d_model=d_model)
        elif emb_type == PositionalEmbeddingType.SINUSOIDAL:
            self.pos_emb = SinusoidalPositionalEncoding(d_model=d_model)
        elif emb_type == PositionalEmbeddingType.ROPE:
            self.pos_emb = None
        else:
            raise ValueError(f"Unsupported positional embedding type: {emb_type}")

        self.layers = nn.ModuleList([
            EncoderLayer(
                emb_type=emb_type,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
    
        self.out = nn.Linear(d_model, 2)

    def forward(self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        h = self.token_emb(x)
        if self.pos_emb is not None:
            h = h + self.pos_emb(x)

        for layer in self.layers:
            h, _ = layer(h, return_attention=False)

        logits = self.out(h)
        return logits

    @torch.no_grad()
    def get_attention(self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        self.eval()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        h = self.token_emb(x)
        if self.pos_emb is not None:
            h = h + self.pos_emb(x)

        attentions = []
        for layer in self.layers:
            h, a = layer(h, return_attention=True)
            attentions.append(a)
        
        return [attn.mean(dim=0) for attn in attentions]

    @torch.no_grad()
    def get_average_attention(self,
        data_loader: DataLoader,
        device: str | None = None,
    ) -> list[torch.Tensor]:
        self.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        
        all_attentions = []
        num_batches = 0
        
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            batch_attentions = self.get_attention(batch_x)

            if len(all_attentions) == 0:
                all_attentions = [torch.zeros_like(attn) for attn in batch_attentions]
            for i, attn in enumerate(batch_attentions):
                all_attentions[i] += attn
        
            num_batches += 1
        
        for i in range(len(all_attentions)):
            all_attentions[i] = all_attentions[i] / num_batches
        
        return all_attentions