from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
import torch.nn as nn


class CATransformerEncoderLayer(nn.Module):
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


class CATransformer(nn.Module):
    """A transformer model for learning cellular automaton rules."""
    
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
            CATransformerEncoderLayer(
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
        _, attentions = self.forward(x, return_attention=True)
        return attentions if isinstance(attentions, list) else []

    @torch.no_grad()
    def predict(self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        self.eval()
        logits, _ = self.forward(x)
        return logits.argmax(dim=-1)


@dataclass
class EvalMetrics:
    cell_accuracy: float
    sequence_accuracy: float


class TrainHistory:
    def __init__(self):
        self.losses: list[float] = []
        self.eval_metrics: list[EvalMetrics] = []
    
    def __len__(self) -> int:
        return len(self.losses)
    
    def add_epoch(self,
        loss: float, 
        eval_metrics: EvalMetrics,
    ) -> None:
        self.losses.append(loss)
        self.eval_metrics.append(eval_metrics)


@torch.no_grad()
def evaluate(
    model: CATransformer,
    data_loader: DataLoader,
    device: str | None = None,
) -> EvalMetrics:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    total_cells = 0
    correct_cells = 0
    total_sequences = 0
    correct_sequences = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            predictions = model(x).argmax(dim=-1)
            correct = predictions == y
            correct_cells += correct.sum().item()
            total_cells += correct.numel()
            correct_sequences += (correct.all(dim=1)).sum().item()
            total_sequences += x.size(0)

    return EvalMetrics(
        cell_accuracy=correct_cells / total_cells,
        sequence_accuracy=correct_sequences / total_sequences,
    )

    
def train(
    model: CATransformer,
    data_loader: DataLoader,
    device: str | None = None,
    n_epochs: int = 20,
    lr: float = 1e-3,
) -> TrainHistory:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = TrainHistory()

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits.view(-1, 2), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        eval_metrics = evaluate(model, data_loader, device)
        history.add_epoch(avg_loss, eval_metrics)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return history


def save_model(
    model: CATransformer,
    save_path: str,
) -> None:
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model(
    model: CATransformer,
    load_path: str,
    device: str | None = None,
) -> CATransformer:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {load_path}")
    return model