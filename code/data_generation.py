import torch
from torch.utils.data import Dataset


def __rule_table(
    rule_number: int
) -> torch.Tensor:
    """Returns a rule table for a given rule number."""
    bits = [(rule_number >> i) & 1 for i in range(8)]
    return torch.tensor(bits, dtype=torch.long)


def __ca_step(
    state: torch.Tensor,
    rule_table: torch.Tensor
) -> torch.Tensor:
    """Performs one step of the cellular automaton given the current state and rule table."""
    N = state.shape[0]
    next_state = torch.zeros_like(state)
    
    for i in range(N):
        left = state[i - 1] if i > 0 else 0
        center = state[i]
        right = state[i + 1] if i < N - 1 else 0
        
        index = (left << 2) | (center << 1) | right
        next_state[i] = rule_table[index]
    
    return next_state


class CADataset(Dataset):
    """A PyTorch Dataset for generating cellular automaton sequences based on a given rule number."""
    
    def __init__(self, n_samples: int, seq_len: int, rule_number: int = 110, steps: int = 1):
        self._n_samples = n_samples
        self._seq_len = seq_len
        self._rule_table = __rule_table(rule_number)
        self._steps = steps
    
    def __len__(self):
        return self._n_samples
    
    def __getitem__(self, idx):
        x = torch.randint(0, 2, (self._seq_len,), dtype=torch.long)
        y = x.clone()
        
        for _ in range(self._steps):
            y = __ca_step(y, self._rule_table)
        
        return x, y