import torch
import matplotlib.pyplot as plt
import numpy as np

from data_generation import CellularAutomaton
from transformer import CATransformer, TrainHistory


def transformer_trajectory(
    model: torch.nn.Module,
    init_state: np.ndarray,
    steps: int,
) -> np.ndarray:
    current_state = torch.from_numpy(init_state).long()
    trajectory = [current_state]

    for _ in range(steps):
        with torch.no_grad():
            next_state = model(current_state.unsqueeze(0)).squeeze(0).argmax(dim=-1)
        trajectory.append(next_state)
        current_state = next_state
    
    np_trajectory = torch.stack(trajectory).cpu().numpy()
    return np_trajectory


def visualize_trajectories(
    trajectories: list[np.ndarray],
    labels: list[str] | None = None,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    n_samples = len(trajectories)
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx in range(n_samples):
        history_np = trajectories[idx].astype(np.uint8)
        
        ax = axes[idx]
        im = ax.imshow(history_np, cmap="gray_r", aspect="auto", interpolation="nearest")
        ax.set_xticks([])

        if labels is not None and idx < len(labels):
            ax.set_title(labels[idx], fontsize=20)
        else:
            ax.set_title(f"Sample {idx + 1}", fontsize=18)
        
        if idx == 0:
            ax.set_ylabel("Time step")
    
    for idx in range(n_samples, len(axes)):
        axes[idx].axis("off")
    
    if title is not None:
        fig.suptitle(title, fontsize=28, fontweight="bold")
    
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
    
    max_epoch = len(history)
    tick_positions = list(range(5, max_epoch + 1, 5))
    plt.xticks(tick_positions)

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
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    
    max_epoch = len(history)
    tick_positions = list(range(5, max_epoch + 1, 5))
    plt.xticks(tick_positions)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [m.sequence_accuracy for m in history.eval_metrics], marker="o", color="orange")
    plt.title("Sequence accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    
    plt.xticks(tick_positions)

    if title is not None:
        plt.suptitle(title, fontsize=14, fontweight="bold")
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def visualize_multiple_loss_histories(
    histories: dict[str, TrainHistory],
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=(8, 5))
    
    for label, history in histories.items():
        epochs = range(1, len(history) + 1)
        plt.plot(epochs, history.losses, marker="o", label=label)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    max_epoch = max(len(h) for h in histories.values())
    tick_positions = list(range(5, max_epoch + 1, 5))
    plt.xticks(tick_positions)

    if title is not None:
        plt.suptitle(title, fontsize=14, fontweight="bold")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def visualize_multiple_metrics_histories(
    histories: dict[str, TrainHistory],
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for label, history in histories.items():
        epochs = range(1, len(history) + 1)
        plt.plot(epochs, [m.cell_accuracy for m in history.eval_metrics], marker="o", label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Cell Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    
    max_epoch = max(len(h) for h in histories.values())
    tick_positions = list(range(5, max_epoch + 1, 5))
    plt.xticks(tick_positions)
    
    plt.subplot(1, 2, 2)
    for label, history in histories.items():
        epochs = range(1, len(history) + 1)
        plt.plot(epochs, [m.sequence_accuracy for m in history.eval_metrics], marker="x", label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Sequence Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.xticks(tick_positions)

    if title is not None:
        plt.suptitle(title, fontsize=14, fontweight="bold")

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def visualize_ca_trajectories(
    rule_number: int,
    init_states: np.ndarray,
    steps: int,
    save_path: str | None = None,
) -> None:
    ca = CellularAutomaton(rule_number)

    trajectories = [
        ca.trajectory(init_state, steps=steps, memoize=False)
        for init_state in init_states
    ]
    
    visualize_trajectories(
        trajectories=trajectories,
        title=f"Celular automaton: rule {rule_number}",
        save_path=save_path
    )


def visualize_predictions(
    rule_number: int,
    model: CATransformer,
    init_states: np.ndarray,
    steps: int,
    save_path: str | None = None,
) -> None:
    trajectories = [
        transformer_trajectory(model, init_state, steps=steps)
        for init_state in init_states
    ]

    visualize_trajectories(
        trajectories=trajectories,
        title=f"Transformer predictions: rule {rule_number}",
        save_path=save_path
    )


def visualize_ca_rules(
    rule_numbers: list[int],
    init_state: np.ndarray,
    steps: int,
    save_path: str | None = None,
) -> None:
    trajectories = []
    for rule_number in rule_numbers:
        ca = CellularAutomaton(rule_number)
        trajectory = ca.trajectory(init_state, steps=steps, memoize=False)
        trajectories.append(trajectory)

    labels = [f"Rule {rule_number}" for rule_number in rule_numbers]
    visualize_trajectories(
        trajectories=trajectories,
        labels=labels,
        title=f"CA rules comparison",
        save_path=save_path
    )


def visualize_attention(
    attention_weights: list[torch.Tensor],
    layer_idx: int = 0,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    if not attention_weights or layer_idx >= len(attention_weights):
        print(f"Invalid layer index: {layer_idx}")
        return
    
    attn_tensor = attention_weights[layer_idx]
    
    if attn_tensor.dim() != 3:
        print(f"Unexpected attention tensor shape: {attn_tensor.shape}. Expected 3D (heads, seq_len, seq_len)")
        return
    
    num_heads = attn_tensor.shape[0]
    attn_np = attn_tensor.detach().cpu().numpy()
    
    n_cols = int(np.ceil(np.sqrt(num_heads)))
    n_rows = int(np.ceil(num_heads / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()
    
    for head_idx in range(num_heads):
        ax = axes[head_idx]
        attn_head = attn_np[head_idx]
        
        im = ax.imshow(attn_head, cmap="viridis", aspect="auto")
        ax.set_title(f"Head {head_idx}", fontsize=18)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    for idx in range(num_heads, len(axes)):
        axes[idx].axis("off")
    
    if title is not None:
        fig.suptitle(title, fontsize=28, fontweight="bold")
    else:
        fig.suptitle(f"Attention Weights - Layer {layer_idx}", fontsize=28, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
