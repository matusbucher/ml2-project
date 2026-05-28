import torch
import matplotlib.pyplot as plt
import numpy as np

from data_generation import CellularAutomaton
from transformer import EvalMetrics, TrainHistory


def ca_state_history(
    ca: CellularAutomaton,
    init_state: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    history = [init_state.clone()]
    current_state = init_state.clone()

    for _ in range(steps):
        next_state = ca.step(current_state)
        history.append(next_state)
        current_state = next_state
    
    return torch.stack(history)


def transformer_state_history(
    model: torch.nn.Module,
    init_state: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    history = [init_state.clone()]
    current_state = init_state.clone()

    for _ in range(steps):
        with torch.no_grad():
            next_state = model(current_state.unsqueeze(0)).squeeze(0).argmax(dim=-1)
        history.append(next_state)
        current_state = next_state
    
    return torch.stack(history)


def visualize_state_histories(
    histories: torch.Tensor,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    n_samples = histories.shape[0]
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx in range(n_samples):
        history_np = histories[idx].cpu().numpy().astype(np.uint8)
        
        ax = axes[idx]
        im = ax.imshow(history_np, cmap="gray_r", aspect="auto", interpolation="nearest")
        ax.set_title(f"Sample {idx + 1}")
        ax.set_xticks([])
        
        if idx == 0:
            ax.set_ylabel("Time step")
    
    for idx in range(n_samples, len(axes)):
        axes[idx].axis("off")
    
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def visualize_loss_history(
    history: TrainHistory,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    epochs = range(1, len(history) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    if title is not None:
        plt.suptitle(title, fontsize=14, fontweight="bold")
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def visualize_metrics_history(
    history: TrainHistory,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    epochs = range(1, len(history) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [m.cell_accuracy for m in history.eval_metrics], marker="o")
    plt.title("Cell accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [m.sequence_accuracy for m in history.eval_metrics], marker="o", color="orange")
    plt.title("Sequence accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    if title is not None:
        plt.suptitle(title, fontsize=14, fontweight="bold")
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    