from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int = 80
    resolution: int = 320
    hidden_dim: int = 128
    num_heads: int = 4
    num_queries: int = 100
    decoder_layers: int = 2
    num_points: int = 4
    backbone_channels: tuple[int, ...] = (24, 48, 96, 128)

    def __post_init__(self):
        for name in ("num_classes", "resolution", "hidden_dim", "num_heads", "num_queries", "decoder_layers", "num_points"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_dim % self.num_heads or self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by num_heads and 4")
        if self.resolution % 16:
            raise ValueError("resolution must be divisible by 16")
        if len(self.backbone_channels) != 4 or any(type(c) is not int or c <= 0 for c in self.backbone_channels):
            raise ValueError("backbone_channels must contain four positive integers")

    @classmethod
    def preset(cls, name="tiny", **overrides):
        presets = {
            "tiny": {},
            "nano": dict(resolution=256, hidden_dim=96, num_queries=50, backbone_channels=(16, 32, 64, 96)),
        }
        if name not in presets:
            raise ValueError(f"Unknown preset: {name}")
        return cls(**(presets[name] | overrides))

    def to_dict(self):
        return asdict(self)
