from typing import Tuple
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm
from stable_baselines3 import PPO

from ppo_mario import get_device, TrainConfiguration, create_model
from .dataset import MarioDataset
from .losses import focal_loss
from .misc import DummyEnv


class BehaviorCloning:

    def __init__(
        self,
        expert_data_dir: str | Path,
        learning_rate: float,
        batch_size: int,
        model_save_path: str | Path,
        cfg: TrainConfiguration,
    ):
        """Train a new PPO model with expert replay."""
        expert_data_dir = Path(expert_data_dir)
        if not (
            expert_data_dir.exists()
            and expert_data_dir.is_dir()
            and len(list(expert_data_dir.glob("*.npz"))) > 0
        ):
            raise ValueError(f"Invalid expert data directory: {expert_data_dir}")

        self.device = get_device()
        print("Device:", self.device)

        # the work directory
        self.cfg = cfg
        self.cfg.device = self.device
        self.model_save_path = Path(model_save_path)

        # prepare the dataset and the data loader
        self.dataset = MarioDataset(expert_data_dir, device=self.device)
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            # drop the last batch if the last batch has only one sample
            # because the batch normalization layer requires at least two samples
            drop_last=len(self.dataset) % batch_size == 1,
        )

        # prepare the model
        self.model = create_model(cfg, env=DummyEnv())

        # prepare the optimizer
        self.optimizer = torch.optim.Adam(
            [
                *self.features_extractor.parameters(),
                *self.mlp_extractor.parameters(),
                *self.action_net.parameters(),
            ],
            lr=learning_rate,
            weight_decay=1e-3,
        )

        self.best_acc = 0
        self.best_model_info = None

    @property
    def features_extractor(self) -> nn.Module:
        return self.model.policy.pi_features_extractor

    @property
    def mlp_extractor(self) -> nn.Module:
        return self.model.policy.mlp_extractor.policy_net

    @property
    def action_net(self) -> nn.Module:
        return self.model.policy.action_net

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.model.policy.set_training_mode(True)
        losses = []
        pbar = tqdm.tqdm(self.loader, desc="Training")
        n_true_positives = 0
        for obs, action, weight in pbar:
            self.optimizer.zero_grad()
            # forward
            obs_norm = obs / 255.0
            features = self.features_extractor(obs_norm)
            features = torch.nn.functional.dropout(features, p=0.5)
            latent_pi = self.mlp_extractor.forward(features)
            logits = self.action_net(latent_pi)
            # calculate the loss
            loss = focal_loss(logits, action, alpha=weight)
            # backward
            loss.backward()
            # optimize
            self.optimizer.step()

            # statistics
            losses.append(loss.detach().cpu().numpy())
            predictions = logits.detach().cpu().numpy().argmax(-1)
            action = action.detach().cpu().numpy()
            n_true_positives += np.sum(predictions == action)

            pbar.set_description(f"[Epoch {epoch + 1}] Loss: {loss:.4f}", refresh=True)

        loss_avg = np.mean(losses)
        acc = n_true_positives / np.sum(self.dataset.class_counts)
        pbar.clear()

        pbar.write(
            f"[Epoch {epoch + 1}] ACC: {acc:.4f} Average Loss: {loss:.4f} {'*' if acc > self.best_acc * .01 else ''}"
        )

        return loss_avg, acc

    def train(self, n_epochs: int) -> dict:
        # run the epoches
        for e in range(n_epochs):
            # the epoch training
            loss_avg, acc = self.train_epoch(e)
            if acc > self.best_acc * 0.01:
                self.best_acc = acc
                # save the best model
                # but first, copy the weights of the actor model to the critic model
                self.model.policy.vf_features_extractor.load_state_dict(
                    self.features_extractor.state_dict()
                )
                self.model.save(str(self.model_save_path))
                self.best_model_info = {
                    "epoch": e,
                    "loss": float(loss_avg),
                    "accuracy": float(acc),
                }
                # write the info to the score file
                (self.model_save_path.parent / "score.txt").write_text(
                    json.dumps(self.best_model_info)
                )
