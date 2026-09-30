"""MSCAD: multi-scale cross-attention anomaly detection for time series."""

from .detector import DEFAULT_SEEDS, MSCAD
from .model import MSCADModel

__all__ = ["MSCAD", "MSCADModel", "DEFAULT_SEEDS"]
