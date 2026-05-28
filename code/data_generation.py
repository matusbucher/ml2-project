from torch.utils.data import Dataset
import numpy as np
import torch
import cellpylib as cpl


class CellularAutomaton:
    """Wrapper around CellPyLib for 1D elementary cellular automata."""

    def __init__(self,
        rule_number: int,
        radius: int = 1,
    ):
        self.rule = cpl.NKSRule(rule_number)
        self.radius = radius
    
    def trajectory(self,
        initial_state: np.ndarray,
        steps: int,
        memoize: bool = True,
    ) -> np.ndarray:
        evolved = cpl.evolve(
            initial_state,
            timesteps=steps + 1,
            apply_rule=self.rule,
            r=self.radius,
            memoize=memoize,
        )

        return evolved

    def evolve(self,
        initial_state: np.ndarray,
        steps: int,
        memoize: bool = True,
    ) -> np.ndarray:
        return self.trajectory(initial_state, steps, memoize)[-1]


class CADataset(Dataset):
    """PyTorch Dataset for CellPyLib-generated CA input/target pairs."""

    def __init__(
        self,
        n_samples: int,
        seq_len: int,
        rule_number: int = 110,
        steps: int = 1,
        radius: int = 1,
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.steps = steps
        self.ca = CellularAutomaton(
            rule_number=rule_number,
            radius=radius,
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = cpl.init_random(self.seq_len)
        y = self.ca.evolve(x, steps=self.steps)

        return torch.from_numpy(x).long(), torch.from_numpy(y).long()