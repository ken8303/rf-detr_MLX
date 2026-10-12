import json
from pathlib import Path

import mlx.nn as nn

from .config import ModelConfig
from .model import LiteDETR


def quantize(model, bits=4, group_size=32):
    if bits not in (4, 8) or group_size not in (32, 64, 128):
        raise ValueError("Supported bits: 4/8; group_size: 32/64/128")
    # Keep reference/query embeddings, convolutions and short input projections in float.
    nn.quantize(model, bits=bits, group_size=group_size,
                class_predicate=lambda path, module: isinstance(module, nn.Linear) and module.weight.shape[-1] % group_size == 0)


def save_checkpoint(model, directory, *, quantization=None, categories=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(directory / "model.safetensors"))
    document = {"format_version": 1, "architecture": "lite-detr", "config": model.config.to_dict(), "quantization": quantization, "categories": categories}
    (directory / "config.json").write_text(json.dumps(document, indent=2) + "\n")


def load_checkpoint(directory):
    directory = Path(directory)
    document = json.loads((directory / "config.json").read_text())
    if document.get("format_version") != 1 or document.get("architecture") != "lite-detr":
        raise ValueError("Unsupported checkpoint; upstream RF-DETR .pth weights are not compatible")
    model = LiteDETR(ModelConfig(**document["config"]))
    if document.get("quantization"):
        quantize(model, **document["quantization"])
    model.load_weights(str(directory / "model.safetensors"), strict=True)
    model.eval()
    return model, document
