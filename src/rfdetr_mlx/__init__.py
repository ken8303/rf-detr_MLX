"""Experimental lightweight detector; no pretrained weights are bundled."""
from .config import ModelConfig
from .model import LiteDETR

__all__ = ["ModelConfig", "LiteDETR"]
