import torch
from torch.utils.data import Dataset


class CellularAutomaton:
    """A cellular automaton rule and update logic."""
    
    def __init__(self,
        rule_number: int,
    ):
        """Initialize a cellular automaton with a given rule number."""
        self.rule_table = self._generate_rule_table(rule_number)
    
    def _generate_rule_table(self, rule_number: int) -> torch.Tensor:
        """Generates a rule table for a given rule number."""
        bits = [(rule_number >> i) & 1 for i in range(8)]
        return torch.tensor(bits, dtype=torch.long)
    
    def step(self, state: torch.Tensor) -> torch.Tensor:
        """Performs one step of the cellular automaton given the current state."""
        # Pad with zeros at boundaries: [0, state..., 0]
        padded = torch.nn.functional.pad(state, (1, 1), mode='constant', value=0)
        
        # Get left, center, right neighbors
        left = padded[:-2]      # [0, state[0], state[1], ...]
        center = padded[1:-1]   # [state[0], state[1], state[2], ...]
        right = padded[2:]      # [state[1], state[2], state[3], ..., 0]
        
        # Compute indices for rule lookup
        indices = (left << 2) | (center << 1) | right
        
        # Apply rule table
        next_state = self.rule_table[indices]
        
        return next_state


class CADataset(Dataset):
    """A PyTorch Dataset for generating cellular automaton sequences based on a given rule number."""

    def __init__(self,
        n_samples: int,
        seq_len: int,
        rule_number: int = 110,
        steps: int = 1,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.ca = CellularAutomaton(rule_number)
        self.steps = steps
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        x = torch.randint(0, 2, (self.seq_len,), dtype=torch.long)
        y = x.clone()
        
        for _ in range(self.steps):
            y = self.ca.step(y)
        
        return x, y