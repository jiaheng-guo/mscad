import random

import numpy as np
import pytest
import torch

from mscad import DEFAULT_SEEDS, MSCAD
from mscad.scoring import windows_to_timeseries


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def detector(**kwargs):
    options = dict(win_size=16, patch_sizes=(4, 8), d_model=8, n_heads=2,
                   n_layers_per_scale=1, bridge_depth=2, dropout=0.0,
                   epochs=2, patience=2, batch_size=4, device='cpu')
    options.update(kwargs)
    return MSCAD(**options)


def training_data():
    t = np.linspace(0, 8, 48)
    return np.column_stack([np.sin(t), 10 + 3 * np.cos(t)])


def test_fit_scores_are_raw_finite_and_training_statistics_are_frozen():
    train = training_data()
    model = detector()
    assert model.fit(train) is model
    np.testing.assert_allclose(model.mean_, train.mean(axis=0))
    np.testing.assert_allclose(model.scale_, train.std(axis=0))
    mean, scale = model.mean_.copy(), model.scale_.copy()
    test = train + 40
    scores = model.decision_function(test)
    assert scores.shape == (48,)
    assert np.isfinite(scores).all() and (scores >= 0).all()
    assert scores.max() > 1  # No per-test min/max scaling.
    np.testing.assert_array_equal(model.mean_, mean)
    np.testing.assert_array_equal(model.scale_, scale)
    assert model.decision_scores_.shape == (48,)
    assert model.best_validation_loss_ == min(x['validation_loss'] for x in model.history_)


def test_seed_one_is_default_and_fit_resets_global_generators():
    assert DEFAULT_SEEDS == (1, 2, 3, 4, 5)
    train = training_data()[:, 0]
    first = detector(dropout=0.1).fit(train)
    expected = first.decision_function(train)
    random.seed(908)
    np.random.seed(12)
    torch.manual_seed(845)
    second = detector(dropout=0.1).fit(train)
    np.testing.assert_array_equal(second.decision_function(train), expected)
    for key, value in first.model_.state_dict().items():
        torch.testing.assert_close(second.model_.state_dict()[key], value, rtol=0, atol=0)
    third = detector(seed=2).fit(train)
    assert not np.array_equal(third.decision_function(train), expected)


def test_window_mapping_averages_overlaps_and_fills_uncovered_tail():
    actual = windows_to_timeseries(np.array([2., 6.]), win_size=4, stride=2, total_len=7)
    np.testing.assert_array_equal(actual, [2, 2, 4, 4, 6, 6, 6])
    np.testing.assert_array_equal(
        windows_to_timeseries(np.array([3.]), win_size=4, stride=2, total_len=2), [3, 3])


def test_channel_scores_share_network_then_average_with_exact_window_mapping():
    train = training_data()
    fitted = detector().fit(train)
    test = train[:23]
    expected = np.zeros(len(test))
    normalized = ((test - fitted.mean_) / fitted.scale_).astype(np.float32)
    fitted.model_.eval()
    with torch.no_grad():
        for channel in range(2):
            windows = np.stack([normalized[i:i + 16, channel:channel + 1]
                                for i in range(0, len(test) - 16 + 1, 2)])
            _, window_scores = fitted.model_(torch.from_numpy(windows))
            expected += windows_to_timeseries(window_scores.numpy(), 16, 2, len(test)) / 2
    np.testing.assert_allclose(fitted.decision_function(test), expected, rtol=1e-6)


def test_constant_training_and_short_test_use_zero_padding_after_normalization():
    fitted = detector().fit(np.full(20, 7.0))
    np.testing.assert_array_equal(fitted.scale_, [1.0])
    short = np.array([8.0, 6.0, 7.0])
    padded = torch.zeros(1, 16, 1)
    padded[0, :3, 0] = torch.tensor([1.0, -1.0, 0.0])
    with torch.no_grad():
        _, expected = fitted.model_(padded)
    np.testing.assert_allclose(fitted.decision_function(short), expected.item(), rtol=1e-6)


def test_validation_is_batched_and_weighted_by_window_count(monkeypatch):
    import mscad.detector as module

    observed = []
    original = module.MSCADModel

    class RecordingModel(original):
        def forward(self, batch):
            observed.append((self.training, len(batch)))
            return super().forward(batch)

    monkeypatch.setattr(module, 'MSCADModel', RecordingModel)
    train = training_data()
    fitted = detector(epochs=1, validation_size=0.35).fit(train)
    assert observed and all(size <= 4 for _, size in observed)
    assert any(not training and size == 3 for training, size in observed)
    normalized = ((train - fitted.mean_) / fitted.scale_).astype(np.float32)
    windows = np.stack([normalized[start:start + 16, channel:channel + 1]
                        for channel in range(2) for start in range(0, 33, 2)])
    n_val = int(len(windows) * 0.35)
    val_indices = np.random.RandomState(1).permutation(len(windows))[:n_val]
    with torch.no_grad():
        _, scores = fitted.model_(torch.from_numpy(windows[val_indices]))
    np.testing.assert_allclose(fitted.best_validation_loss_, scores.mean().item(), rtol=1e-6)


def test_strict_improvement_patience_restores_best_epoch(monkeypatch):
    import mscad.detector as module

    class ScheduledValidationModel(torch.nn.Module):
        """A trainable scalar isolates checkpoint bookkeeping from optimization."""

        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.epoch = 0
            self.states = {}

        def train(self, mode=True):
            if mode:
                self.epoch += 1
            elif self.training:
                self.states[self.epoch] = self.weight.detach().clone()
            return super().train(mode)

        def forward(self, batch):
            if self.training:
                value = (self.weight - 1).square()
            else:
                value = batch.new_tensor([3., 1., 1., 2.][self.epoch - 1])
            scores = value.expand(len(batch))
            return scores[:, None], scores

    monkeypatch.setattr(module, 'MSCADModel', ScheduledValidationModel)
    fitted = detector(epochs=10, patience=2).fit(training_data()[:, 0])
    assert fitted.n_epochs_ == 4  # Equality is stale, not an improvement.
    assert fitted.best_validation_loss_ == 1.0
    torch.testing.assert_close(fitted.model_.weight, fitted.model_.states[2], rtol=0, atol=0)
    assert fitted.model_.states[2] != fitted.model_.states[4]


@pytest.mark.parametrize('length', [1, 15, 16, 17])
def test_too_short_training_is_rejected_without_duplicate_holdout(length):
    with pytest.raises(ValueError, match='two|2'):
        detector().fit(np.zeros(length))


@pytest.mark.parametrize('data', [[], np.empty((3, 0)), np.zeros((2, 3, 4)),
                                 [1, np.nan], [1, np.inf], [1, -np.inf]])
def test_invalid_training_data_is_rejected(data):
    with pytest.raises(ValueError):
        detector().fit(data)


def test_scoring_before_fit_and_channel_mismatch_are_clear_errors():
    with pytest.raises(RuntimeError, match='fit'):
        detector().decision_function(np.zeros(20))
    fitted = detector().fit(training_data())
    with pytest.raises(ValueError, match='channel|feature'):
        fitted.decision_function(np.zeros(20))
    for invalid in [[], [np.nan], [np.inf]]:
        with pytest.raises(ValueError):
            fitted.decision_function(invalid)


@pytest.mark.parametrize('options', [
    {'win_size': 0}, {'win_size': 15.5}, {'patch_sizes': ()},
    {'patch_sizes': (4, 32)}, {'patch_sizes': (1,)}, {'patch_sizes': (3,)},
    {'patch_sizes': (4, 4)}, {'d_model': 7}, {'n_heads': 0},
    {'n_layers_per_scale': 0}, {'bridge_depth': 0}, {'lr': 0}, {'lr': float('nan')},
    {'epochs': 0}, {'patience': 0}, {'batch_size': 0}, {'validation_size': 0},
    {'validation_size': 1}, {'dropout': -1}, {'dropout': 1}, {'seed': -1},
    {'seed': 1.5}, {'win_size': 1028, 'patch_sizes': (2,)},
])
def test_invalid_parameters_are_rejected_early(options):
    with pytest.raises(ValueError):
        detector(**options)
