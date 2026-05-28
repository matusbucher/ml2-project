import torch
from torch.utils.data import Dataset


class CellularAutomaton:
    """A cellular automaton rule and update logic."""

    def __init__(self,
        rule_number: int,
    ):
        self.rule_table = self._generate_rule_table(rule_number)

    def _generate_rule_table(self, rule_number: int) -> torch.Tensor:
        bits = [(rule_number >> i) & 1 for i in range(8)]
        return torch.tensor(bits, dtype=torch.long)

    def step(self, state: torch.Tensor) -> torch.Tensor:
        padded = torch.nn.functional.pad(state, (1, 1), mode="constant", value=0)
        left = padded[:-2]
        center = padded[1:-1]
        right = padded[2:]

        indices = (left << 2) | (center << 1) | right
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