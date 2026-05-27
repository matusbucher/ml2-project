import torch
import matplotlib.pyplot as plt
import numpy as np

from data_generation import CellularAutomaton


def ca_history(
    ca: CellularAutomaton,
    init_state: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Generates the history of     a cellular automaton given an initial state and number of steps."""
    
    history = [init_state.clone()]
    current_state = init_state.clone()

    for _ in range(steps):
        next_state = ca.step(current_state)
        history.append(next_state)
        current_state = next_state
    
    return torch.stack(history)


def visualize_ca(
    ca: CellularAutomaton,
    init_states: torch.Tensor,
    steps: int,
    save_path: str | None = None,
) -> None:
    """Visualizes the history (or more) of a cellular automaton as a grid of pixel images."""

    n_samples = init_states.shape[0]
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx in range(n_samples):
        history = ca_history(ca, init_states[idx], steps)
        history_np = history.cpu().numpy().astype(np.uint8)
        
        ax = axes[idx]
        im = ax.imshow(history_np, cmap='gray_r', aspect='auto', interpolation='nearest')
        ax.set_title(f'Initial state {idx}')
        ax.set_xlabel('Position')
        ax.set_ylabel('Time step')
    
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    