from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
import torch.nn as nn


class CATransformerInterface(nn.Module, ABC):
    """Interface for CATransformer models, defining common methods for both standard and generalized versions."""

    @abstractmethod
    def forward(self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | torch.Tensor:
        raise NotImplementedError("Subclasses must implement the forward method.")
    
    @abstractmethod
    @torch.no_grad()
    def get_attention(self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the get_attention method.")
    
    @abstractmethod
    @torch.no_grad()
    def get_average_attention(self,
        data_loader: DataLoader,
        device: str | None = None,
    ) -> list[torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the get_average_attention method.")


class EncoderLayer(nn.Module):
    """Transformer encoder layer that can return attention weights."""

    def __init__(self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()

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


class CATransformer(CATransformerInterface):
    """A transformer model for learning cellular automaton rules for a fixed sequence length."""
    
    def __init__(self,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.layers = nn.ModuleList([
            EncoderLayer(
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
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | torch.Tensor:
        h = self.token_emb(x) + self.pos_emb[:, :x.size(1), :]

        attentions = []

        for layer in self.layers:
            h, attn = layer(h, return_attention=return_attention)

            if return_attention:
                attentions.append(attn)

        logits = self.out(h)

        if return_attention:
            return logits, attentions

        return logits

    @torch.no_grad()
    def get_attention(self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        self.eval()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        _, attentions = self.forward(x, return_attention=True)
        
        if attentions and isinstance(attentions, list):
            return [attn.mean(dim=0) for attn in attentions]
        
        return []

    @torch.no_grad()
    def get_average_attention(self,
        data_loader: DataLoader,
        device: str | None = None,
    ) -> list[torch.Tensor]:
        self.eval()
        
        all_attentions = None
        total_samples = 0
        
        for batch_x, _ in data_loader:
            if device is not None:
                batch_x = batch_x.to(device)
            
            _, batch_attentions = self.forward(batch_x, return_attention=True)
            
            if batch_attentions and isinstance(batch_attentions, list):
                if all_attentions is None:
                    all_attentions = [attn.sum(dim=0) for attn in batch_attentions]
                else:
                    for i, attn in enumerate(batch_attentions):
                        all_attentions[i] += attn.sum(dim=0)
            
            total_samples += batch_x.size(0)
        
        if all_attentions is not None:
            for i in range(len(all_attentions)):
                all_attentions[i] = all_attentions[i] / total_samples
        
        return all_attentions if all_attentions is not None else []
    

class AlibiEncoderLayer(nn.Module):
    """Transformer encoder layer with bidirectional ALiBi attention bias."""

    def __init__(self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()

        self.n_heads = n_heads

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

    def _get_alibi_slopes(self,
        device: torch.device,
    ) -> torch.Tensor:
        slopes = torch.tensor(
            [2.0 ** (-(i + 1)) for i in range(self.n_heads)],
            dtype=torch.float32,
            device=device,
        )
        return slopes

    def _make_alibi_mask(self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        slopes = self._get_alibi_slopes(device)

        positions = torch.arange(seq_len, device=device)
        distances = torch.abs(
            positions[:, None] - positions[None, :]
        ).float()

        bias = -slopes[:, None, None] * distances[None, :, :]
        bias = bias.repeat(batch_size, 1, 1)

        return bias

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len, _ = x.shape

        attn_mask = self._make_alibi_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
        )

        attn_output, attn_weights = self.self_attn(x, x, x,
            attn_mask=attn_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        x = self.norm1(x + self.dropout_attn(attn_output))

        ff = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(x)
                )
            )
        )

        x = self.norm2(x + self.dropout_ff(ff))

        if return_attention:
            return x, attn_weights

        return x, None
    

class CATransformerGeneralized(CATransformerInterface):
    """
    A transformer model for learning cellular automaton rules that uses ALiBi positional
    embeddings for generalization to arbitrary sequence lengths.
    """

    def __init__(self,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(2, d_model)

        self.layers = nn.ModuleList([
            AlibiEncoderLayer(
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
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | torch.Tensor:
        h = self.token_emb(x)

        attentions = []

        for layer in self.layers:
            h, attn = layer(
                h,
                return_attention=return_attention,
            )

            if return_attention:
                attentions.append(attn)

        logits = self.out(h)

        if return_attention:
            return logits, attentions

        return logits

    @torch.no_grad()
    def get_attention(self,
        x: torch.Tensor,
    ) -> list[torch.Tensor]:
        self.eval()

        if x.dim() == 1:
            x = x.unsqueeze(0)

        _, attentions = self.forward(
            x,
            return_attention=True,
        )

        if attentions and isinstance(attentions, list):
            return [
                attn.mean(dim=0)
                for attn in attentions
            ]

        return []

    @torch.no_grad()
    def get_average_attention(self,
        data_loader: DataLoader,
        device: str | None = None,
    ) -> list[torch.Tensor]:
        self.eval()

        all_attentions = None
        total_samples = 0

        for batch_x, _ in data_loader:
            if device is not None:
                batch_x = batch_x.to(device)

            _, batch_attentions = self.forward(
                batch_x,
                return_attention=True,
            )

            if batch_attentions and isinstance(batch_attentions, list):
                if all_attentions is None:
                    all_attentions = [
                        attn.sum(dim=0)
                        for attn in batch_attentions
                    ]
                else:
                    for i, attn in enumerate(batch_attentions):
                        all_attentions[i] += attn.sum(dim=0)

            total_samples += batch_x.size(0)

        if all_attentions is not None:
            all_attentions = [
                attn / total_samples
                for attn in all_attentions
            ]

        return all_attentions if all_attentions is not None else []