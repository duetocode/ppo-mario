from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from ppo_mario import get_device


class MarioDataset(Dataset):
    """Dataset for the Mario imitation learning."""

    def __init__(self, data_dir: str | Path, device: str = None):
        """Initialize a dataset object from the given path. The path should contain the npz files, which contains the observations and actions."""
        data_dir = Path(data_dir)
        # check
        if not (data_dir.exists() and data_dir.is_dir()):
            raise ValueError(f"{data_dir} is not a valid directory")

        device = device if device is not None else get_device()

        # enumerate the data
        data = []
        for f in sorted(data_dir.glob("**/*.npz")):
            with np.load(f) as npz_file:
                data.append({"obs": npz_file["obs"], "action": npz_file["action"]})

        print(f"Found {len(data)} data points.")
        # split into observations(inputs) and actions(labels) and convert to torch.Tensor
        self.observations = [
            torch.as_tensor(
                d["obs"][:, 16:, :].squeeze(), dtype=torch.uint8, device=device
            )
            for d in data
        ]
        self.actions = np.asarray([d["action"] for d in data])

        # calculate the class weights
        self.class_counts = np.bincount(self.actions)
        class_weights = len(self.actions) / (len(self.class_counts) * self.class_counts)
        class_weights = class_weights / np.sum(class_weights)
        self.class_weights = torch.as_tensor(
            class_weights, dtype=torch.float, device=device
        )

        # conver the actions to tensors
        self.actions = torch.as_tensor(self.actions, dtype=torch.long, device=device)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        action = self.actions[idx]
        return self.observations[idx], self.actions[idx], self.class_weights[action]

    def split_train_val(self, validation_split: float = 0.2) -> Tuple[Dataset, Dataset]:
        """Split the dataset into trianing and validation datasets respect to the given ratio.

        parameters
        ----------
        validation_split: float
            The ratio of the validation dataset. Default is 0.2.

        return
        -------
        Tuple[Dataset, Dataset]
            The training and validation datasets.
        """

        train_size = int(len(self) * (1 - validation_split))
        val_size = len(self) - train_size
        return torch.utils.data.random_split(self, [train_size, val_size])
