from typing import Tuple
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
import torch.utils
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
        l2: float,
    ):
        """Train a new PPO model with expert replay."""
        expert_data_dir = Path(expert_data_dir)
        if not (
            expert_data_dir.exists()
            and expert_data_dir.is_dir()
            and len(list(expert_data_dir.glob("**/*.npz"))) > 0
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
        train_dataset, val_dataset = self.dataset.split_train_val()
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            # drop the last batch if the last batch has only one sample
            # because the batch normalization layer requires at least two samples
            drop_last=len(train_dataset) % batch_size == 1,
        )
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
            weight_decay=l2,
        )

        # prepare the logger
        logging_dir = self.model_save_path.parent / "logs" / "imitation"
        logging_dir.mkdir(parents=True, exist_ok=True)
        self.logger = SummaryWriter(log_dir=str(logging_dir), filename_suffix="imitate")

        self.best_score = 0
        self.find_best_model = False
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

    def _forward(
        self, obs: Tensor, labels: Tensor, weight: Tensor, training: bool = True
    ) -> Tuple[Tensor, Tensor]:
        """Runt the forward pass of the model.
        parameter
        ---------
        obs: Tensor
            the input data
        return
        ------
        Tuple[Tensor, Tensor]
            the logits and the loss
        """
        obs_norm = obs / 255.0
        features = self.features_extractor(obs_norm)
        if training:
            features = torch.nn.functional.dropout(features, p=0.5)
        latent_pi = self.mlp_extractor.forward(features)
        logits = self.action_net(latent_pi)
        # calculate the loss
        loss = focal_loss(logits, labels, alpha=weight)
        return logits, loss

    def _validate(self, epoch: int):
        """Run the validation phase of the training"""
        self.model.policy.set_training_mode(False)

        losses, predictions, labels = [], [], []

        # run the forward
        pbar = tqdm.tqdm(self.val_loader, leave=False)
        with torch.no_grad():
            for obs, action, weight in pbar:
                # forward
                logits, loss = self._forward(obs, action, weight, training=False)
                losses.append(loss.cpu().numpy())
                predictions.append(logits.argmax(1).cpu().numpy())
                labels.append(action.cpu().numpy())
                pbar.set_description(
                    f"[Validation] Loss: {loss.sum():.4f}", refresh=True
                )

        # statistics
        loss_avg = np.mean(losses)

        # first, concatenate the predictions and labels
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        # calculate the macro precision, recall, and f1
        categories = np.unique(labels)
        tp = {c: np.sum(predictions[labels == c] == c) for c in categories}
        precision = [(n, np.sum(predictions == c)) for c, n in tp.items()]
        recall = [(n, np.sum(labels == c)) for c, n in tp.items()]
        precision = np.mean([(_tp / _p) if _p > 0 else 0 for _tp, _p in precision])
        recall = np.mean([(_tp / _r) if _r > 0 else 0 for _tp, _r in recall])
        f1 = 2 * precision * recall / (precision + recall)

        pbar.close()
        print(
            f"[Epoch {epoch}] Validation Loss: {loss_avg:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}"
        )
        self.logger.add_scalar("imitation/val_loss", loss_avg, epoch)
        self.logger.add_scalar("imitation/val_precision", precision, epoch)
        self.logger.add_scalar("imitation/val_recall", recall, epoch)
        self.logger.add_scalar("imitation/val_f1", f1, epoch)

        return {"f1": f1, "precision": precision, "recall": recall}

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        self.is_best_model = False
        self.model.policy.set_training_mode(True)
        losses = []
        pbar = tqdm.tqdm(self.train_loader, desc="Training", leave=False)
        n_true_positives = 0
        for obs, action, weight in pbar:
            self.optimizer.zero_grad()
            # forward
            logits, loss = self._forward(obs, action, weight)
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
        self.logger.add_scalar("imitation/loss", loss_avg, epoch)
        self.logger.add_scalar("imitation/accuracy", acc, epoch)

        scores = self._validate(epoch)
        self.logger.flush()

        self.is_best_model = scores["f1"] >= self.best_score
        self.best_score = max(self.best_score, scores["f1"])

        pbar.write(
            f"[Epoch {epoch + 1}] ACC: {acc:.4f} Average Loss: {loss:.4f} {'*' if self.is_best_model else ''}"
        )

        return loss_avg, scores

    def train(self, n_epochs: int) -> dict:
        # run the epoches
        for e in range(n_epochs):
            # the epoch training
            loss_avg, score = self.train_epoch(e)
            if self.is_best_model:
                # save the best model
                # but first, copy the weights of the actor model to the critic model
                self.model.policy.vf_features_extractor.load_state_dict(
                    self.features_extractor.state_dict()
                )
                self.model.save(str(self.model_save_path))
                self.best_model_info = {"epoch": e, **score}
                # write the info to the score file
                (self.model_save_path.parent / "score.txt").write_text(
                    json.dumps(self.best_model_info)
                )
        self.logger.close()
