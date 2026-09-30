"""Recover timestamp scores from overlapping scalar window scores."""

import numpy as np

from .model import _positive_int


def windows_to_timeseries(window_scores: np.ndarray, win_size: int,
                          stride: int, total_len: int) -> np.ndarray:
    """Average the scores of covering windows and forward-fill an uncovered tail.

    Windows start at ``0, stride, 2 * stride, ...``. A short nonempty series
    uses one zero-padded window and receives that window's score throughout.
    """
    win_size = _positive_int("win_size", win_size)
    stride = _positive_int("stride", stride)
    total_len = _positive_int("total_len", total_len)
    if stride > win_size:
        raise ValueError("stride must not exceed win_size.")
    values = np.asarray(window_scores, dtype=np.float64)
    expected = max(1, (total_len - win_size) // stride + 1)
    if values.ndim != 1 or len(values) != expected:
        raise ValueError(f"Expected {expected} window scores, got shape {values.shape}.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Window scores must be finite and nonnegative.")

    scores = np.zeros(total_len, dtype=np.float64)
    counts = np.zeros(total_len, dtype=np.int64)
    for index, value in enumerate(values):
        start = index * stride
        stop = min(start + win_size, total_len)
        scores[start:stop] += value
        counts[start:stop] += 1
    last_covered = min((len(values) - 1) * stride + win_size, total_len)
    scores[:last_covered] /= counts[:last_covered]
    scores[last_covered:] = scores[last_covered - 1]
    return scores
