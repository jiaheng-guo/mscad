"""Train MSCAD on a normal prefix and return raw per-timestamp anomaly scores."""

import math
import random
from numbers import Integral, Real
from typing import Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .model import MSCADModel, _dropout, _patch_sizes, _positive_int
from .scoring import windows_to_timeseries


DEFAULT_SEEDS = (1, 2, 3, 4, 5)


def _series(X) -> np.ndarray:
    raw = np.asarray(X)
    if np.iscomplexobj(raw):
        raise ValueError("X must contain real numeric values.")
    try:
        data = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must contain real numeric values.") from exc
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or min(data.shape) == 0:
        raise ValueError("X must be a nonempty array of shape (time,) or (time, channels).")
    if not np.isfinite(data).all():
        raise ValueError("X contains NaN or infinite values; preserve row alignment when cleaning data.")
    return data


class _WindowDataset(Dataset):
    """Slice FP32 windows on demand instead of materializing a window matrix."""

    def __init__(self, data: np.ndarray, win_size: int, stride: int):
        self.data = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        self.win_size = win_size
        self.stride = stride
        self.n_windows = max(0, (len(data) - win_size) // stride + 1)

    def __len__(self):
        return self.n_windows * self.data.shape[1]

    def __getitem__(self, index):
        channel, window = divmod(int(index), self.n_windows)
        start = window * self.stride
        return self.data[start:start + self.win_size, channel:channel + 1]


class MSCAD:
    """Channel-independent multi-scale reconstruction anomaly detector.

    ``fit(X)`` treats all of ``X`` as the normal training prefix; it never
    consumes labels. Validation windows are randomly held out after computing
    per-channel statistics on that prefix. ``decision_function(X)`` reuses
    those statistics and returns unthresholded nonnegative scores.

    ``device=None`` selects CUDA when available, otherwise CPU. Each fit resets
    Python, NumPy and PyTorch random generators to ``seed``; CPU runs are
    repeatable in the same environment. Hardware/library changes and CUDA
    kernels can still affect numerical reproducibility.
    """

    def __init__(self, win_size: int = 128, patch_sizes: Optional[Sequence[int]] = None,
                 d_model: int = 256, n_heads: int = 4, n_layers_per_scale: int = 2,
                 bridge_depth: int = 2, lr: float = 1e-3, epochs: int = 30,
                 patience: int = 5, batch_size: int = 128, validation_size: float = 0.1,
                 dropout: float = 0.1, seed: int = 1,
                 device: Optional[Union[str, torch.device]] = None):
        self.win_size = _positive_int("win_size", win_size)
        self.patch_sizes = _patch_sizes((4, 16, 64) if patch_sizes is None else patch_sizes)
        self.d_model = _positive_int("d_model", d_model)
        self.n_heads = _positive_int("n_heads", n_heads)
        self.n_layers_per_scale = _positive_int("n_layers_per_scale", n_layers_per_scale)
        self.bridge_depth = _positive_int("bridge_depth", bridge_depth)
        self.epochs = _positive_int("epochs", epochs)
        self.patience = _positive_int("patience", patience)
        self.batch_size = _positive_int("batch_size", batch_size)
        self.dropout = _dropout(dropout)
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        if max(self.patch_sizes) > self.win_size:
            raise ValueError("Every patch size must be <= win_size; patch sizes are never silently filtered.")
        if any((self.win_size - patch) // (patch // 2) + 1 > 512 for patch in self.patch_sizes):
            raise ValueError("A scale has more than 512 tokens; reduce win_size or increase patch sizes.")
        if (isinstance(lr, bool) or not isinstance(lr, Real)
                or not math.isfinite(lr) or lr <= 0):
            raise ValueError("lr must be positive and finite.")
        self.lr = float(lr)
        if (isinstance(validation_size, bool) or not isinstance(validation_size, Real)
                or not math.isfinite(validation_size) or not 0 < validation_size < 1):
            raise ValueError("validation_size must be in (0, 1).")
        self.validation_size = float(validation_size)
        if (isinstance(seed, bool) or not isinstance(seed, Integral)
                or not 0 <= seed < 2**32):
            raise ValueError("seed must be an integer in [0, 2**32).")
        self.seed = int(seed)
        self.device = torch.device(device if device is not None else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.stride_ = min(self.patch_sizes) // 2

    def _normalize(self, data: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            normalized = ((data - self.mean_) / self.scale_).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise ValueError("Normalized data exceeds finite FP32 range; rescale the input values.")
        return normalized

    def fit(self, X):
        """Fit one shared network using only the supplied normal training data.

        At least two complete channel windows are required, so training and
        validation are nonempty and never populated by duplicating one window.
        """
        data = _series(X)
        n_windows = max(0, (len(data) - self.win_size) // self.stride_ + 1)
        if n_windows * data.shape[1] < 2:
            raise ValueError("Training requires at least two complete windows for separate training and validation.")
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.n_features_in_ = data.shape[1]
        with np.errstate(over="ignore", invalid="ignore"):
            self.mean_ = data.mean(axis=0)
            self.scale_ = data.std(axis=0)
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.scale_).all():
            raise ValueError("Training statistics overflowed; rescale the input values.")
        self.scale_[self.scale_ < 1e-8] = 1.0
        dataset = _WindowDataset(self._normalize(data), self.win_size, self.stride_)
        order = np.random.permutation(len(dataset))
        n_val = min(len(dataset) - 1, max(1, int(len(dataset) * self.validation_size)))
        train_loader = DataLoader(Subset(dataset, order[n_val:]), batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(dataset, order[:n_val]), batch_size=self.batch_size)

        self.model_ = MSCADModel(
            self.patch_sizes, self.d_model, self.n_heads, self.n_layers_per_scale,
            self.bridge_depth, self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        self.history_ = []
        self.best_validation_loss_ = float("inf")
        best_state = None
        stale_epochs = 0
        for epoch in range(self.epochs):
            self.model_.train()
            train_total = 0.0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                _, scores = self.model_(batch)
                loss = scores.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError("Training loss is not finite; check the data scale and learning rate.")
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                train_total += loss.item() * len(batch)
            scheduler.step()

            self.model_.eval()
            val_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    _, scores = self.model_(batch.to(self.device))
                    val_total += scores.sum().item()
            val_loss = val_total / n_val
            if not math.isfinite(val_loss):
                raise RuntimeError("Validation loss is not finite; check the data scale and learning rate.")
            self.history_.append({"train_loss": train_total / (len(dataset) - n_val),
                                  "validation_loss": val_loss})
            # Strict improvement matches the manuscript pseudocode; restore
            # this minimum-loss checkpoint even when training exhausts epochs.
            if val_loss < self.best_validation_loss_:
                self.best_validation_loss_ = val_loss
                best_state = {key: value.detach().cpu().clone()
                              for key, value in self.model_.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= self.patience:
                break

        self.n_epochs_ = len(self.history_)
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.decision_scores_ = self.decision_function(data)
        return self

    def decision_function(self, X) -> np.ndarray:
        """Return a raw score per timestamp, averaging window and channel scores.

        Nonempty test series shorter than ``win_size`` are padded with zeros
        *after* applying training normalization. An uncovered final tail is
        assigned the last covered timestamp's score.
        """
        if not hasattr(self, "model_") or not hasattr(self, "n_epochs_"):
            raise RuntimeError("Call fit before decision_function.")
        data = _series(X)
        if data.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} channels, got {data.shape[1]}.")
        normalized = self._normalize(data)
        total_len = len(normalized)
        if total_len < self.win_size:
            normalized = np.pad(normalized, ((0, self.win_size - total_len), (0, 0)))
        dataset = _WindowDataset(normalized, self.win_size, self.stride_)
        all_scores = np.zeros(total_len, dtype=np.float64)
        self.model_.eval()
        with torch.no_grad():
            for channel in range(self.n_features_in_):
                start = channel * dataset.n_windows
                loader = DataLoader(Subset(dataset, range(start, start + dataset.n_windows)),
                                    batch_size=self.batch_size)
                window_scores = np.empty(dataset.n_windows, dtype=np.float64)
                offset = 0
                for batch in loader:
                    _, scores = self.model_(batch.to(self.device))
                    window_scores[offset:offset + len(batch)] = scores.cpu().numpy()
                    offset += len(batch)
                all_scores += windows_to_timeseries(window_scores, self.win_size,
                                                   self.stride_, total_len)
        return all_scores / self.n_features_in_
