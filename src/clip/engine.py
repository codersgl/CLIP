from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from clip.loss import CLIPLoss


class EarlyStopping:
    def __init__(
        self, patience=7, verbose=False, delta: float = 0.0, path="checkpoint.pt"
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        loss_func: CLIPLoss,
        optimizer: Optimizer,
        device: torch.device,
        epochs: int,
        use_early_stop: bool,
        patience: int,
        save_path: str,
        grad_clip_max_norm: float = 1.0,
        show_progress: bool = True,
        log_dir: Optional[str] = None,
    ) -> None:
        self.model = model
        self.loss_func = loss_func
        self.train_data_loader = train_dataloader
        self.val_data_loader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.grad_clip_max_norm = grad_clip_max_norm
        self.metrics: dict = {"train_loss": [], "valid_loss": []}
        self.save_path = Path(save_path)
        self.use_early_stop = use_early_stop
        self.show_progress = show_progress

        self.writer = None
        if log_dir is not None:
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard logs will be saved to {log_dir}")

        self.early_stop: Optional[EarlyStopping] = None
        if use_early_stop:
            self.early_stop = EarlyStopping(
                patience=patience,
                path=save_path,
                verbose=True,
            )
        else:
            self.early_stop = None

    def train_one_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        iterator = self.train_data_loader
        if self.show_progress:
            iterator = tqdm(iterator, desc=f"Epoch {epoch} Training", leave=False)

        for image, text_dict in iterator:
            image = image.to(self.device)
            input_ids = text_dict["input_ids"].to(self.device)
            attention_mask = text_dict["attention_mask"].to(self.device)

            image_features_normed, text_features_normed = self.model(
                image, input_ids, attention_mask
            )
            loss = self.loss_func(image_features_normed, text_features_normed)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_max_norm
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch: int):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        iterator = self.val_data_loader
        if self.show_progress:
            iterator = tqdm(iterator, desc=f"Epoch {epoch} Validation", leave=False)

        for image, text_dict in iterator:
            image = image.to(self.device)
            input_ids = text_dict["input_ids"].to(self.device)
            attention_mask = text_dict["attention_mask"].to(self.device)

            image_features_normed, text_features_normed = self.model(
                image, input_ids, attention_mask
            )
            loss = self.loss_func(image_features_normed, text_features_normed)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            valid_loss = self.evaluate_one_epoch(epoch)

            self.metrics["train_loss"].append(train_loss)
            self.metrics["valid_loss"].append(valid_loss)

            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/valid", valid_loss, epoch)
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("LR", current_lr, epoch)

            if self.early_stop is not None:
                self.early_stop(valid_loss, self.model)
                if self.early_stop.early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            print(
                f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {valid_loss:.6f}"
            )

        if self.writer is not None:
            self.writer.close()
            print("TensorBoard writer closed.")

        if not self.use_early_stop:
            self.save_checkpoint()

    def save_checkpoint(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.save_path)
        print(f"Model saved to {self.save_path}")
