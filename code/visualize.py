import torch
import matplotlib.pyplot as plt
import numpy as np

from data_generation import CellularAutomaton


def ca_history(
    ca: CellularAutomaton,
    init_state: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Generates the history of a cellular automaton."""
    
    history = [init_state.clone()]
    current_state = init_state.clone()

    for _ in range(steps):
        next_state = ca.step(current_state)
        history.append(next_state)
        current_state = next_state
    
    return torch.stack(history)


def transformer_history(
    model: torch.nn.Module,
    init_state: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Generates the history of a transformer model."""
    
    history = [init_state.clone()]
    current_state = init_state.clone()

    for _ in range(steps):
        with torch.no_grad():
            next_state = model(current_state.unsqueeze(0)).squeeze(0).argmax(dim=-1)
        history.append(next_state)
        current_state = next_state
    
    return torch.stack(history)


def visualize_states(
    histories: torch.Tensor,
    title: str | None = None,
    save_path: str | None = None,
) -> None:
    """Visualizes a grid of pre-generated histories as pixel images."""

    n_samples = histories.shape[0]
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx in range(n_samples):
        history_np = histories[idx].cpu().numpy().astype(np.uint8)
        
        ax = axes[idx]
        im = ax.imshow(history_np, cmap='gray_r', aspect='auto', interpolation='nearest')
        ax.set_title(f'Sample {idx + 1}')
        ax.set_xticks([])
        
        if idx == 0:
            ax.set_ylabel('Time step')
    
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    if title is not None:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    