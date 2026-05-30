from __future__ import annotations
from dataclasses import dataclass


import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from transformer import CATransformer


@dataclass
class EvalMetrics:
    cell_accuracy: float
    sequence_accuracy: float


class TrainHistory:
    def __init__(self):
        self.train_losses: list[float] = []
        self.test_losses: list[float] = []
        self.eval_metrics: list[EvalMetrics] = []
    
    def __len__(self) -> int:
        return len(self.train_losses)
    
    def add_epoch(self,
        train_loss: float,
        test_loss: float,
        eval_metrics: EvalMetrics,
    ) -> None:
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.eval_metrics.append(eval_metrics)
    
    def save(self,
        save_path: str
    ) -> None:
        torch.save({
            "train_losses": self.train_losses,
            "test_losses": self.test_losses,
            "eval_metrics": self.eval_metrics,
        }, save_path)

    @classmethod
    def load(cls,
        load_path: str
    ) -> TrainHistory:
        with torch.serialization.safe_globals([EvalMetrics]):
            data = torch.load(load_path)
        history = cls()
        history.train_losses = data["train_losses"]
        history.test_losses = data["test_losses"]
        history.eval_metrics = data["eval_metrics"]
        return history


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
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
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
        avg_test_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits.view(-1, 2), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        if test_loader is not None:
            total_test_loss = 0.0
            model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, 2), y.view(-1))
                    total_test_loss += loss.item()
            avg_test_loss = total_test_loss / len(test_loader)

        avg_loss = total_loss / len(train_loader)
        eval_metrics = evaluate(model, train_loader, device)
        history.add_epoch(avg_loss, avg_test_loss, eval_metrics)
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