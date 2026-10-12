"""NHWC backbone, learned queries, deformable decoding and iterative boxes.

This is a smaller architecture inspired by RF-DETR, not a weight-compatible port.
"""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .config import ModelConfig


def inverse_sigmoid(x):
    x = mx.clip(x, 1e-5, 1 - 1e-5)
    return mx.log(x / (1 - x))


def position_encoding(height, width, channels):
    """Fixed sine/cosine XY features without a learned resolution-dependent table."""
    frequencies = mx.exp(-math.log(10000) * mx.arange(channels // 4) / max(channels // 4 - 1, 1))
    x = (mx.arange(width) + .5) / width * (2 * math.pi)
    y = (mx.arange(height) + .5) / height * (2 * math.pi)
    px, py = x[:, None] * frequencies, y[:, None] * frequencies
    px = mx.concatenate([mx.sin(px), mx.cos(px)], axis=-1)
    py = mx.concatenate([mx.sin(py), mx.cos(py)], axis=-1)
    return mx.concatenate([mx.broadcast_to(py[:, None], (height, width, channels//2)), mx.broadcast_to(px[None], (height, width, channels//2))], axis=-1)


def sample_bilinear(feature, locations):
    """Sample NHWC features at normalized xy coordinates with zero padding.

    locations: [B,Q,heads,points,2]; result: [B,Q,heads,points,head_dim].
    Pixel centers follow align_corners=False (x * width - 0.5).
    """
    b, h, w, channels = feature.shape
    heads = locations.shape[2]
    values = feature.reshape(b, h * w, heads, channels // heads)
    x, y = locations[..., 0] * w - 0.5, locations[..., 1] * h - 0.5
    x0 = mx.stop_gradient(mx.floor(x).astype(mx.int32))
    y0 = mx.stop_gradient(mx.floor(y).astype(mx.int32))
    dx, dy = x - x0, y - y0
    bi = mx.arange(b)[:, None, None, None]
    hi = mx.arange(heads)[None, None, :, None]

    def gather(ix, iy):
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        index = mx.clip(iy, 0, h - 1) * w + mx.clip(ix, 0, w - 1)
        return values[bi, index, hi] * valid[..., None]

    return (gather(x0, y0) * ((1-dx)*(1-dy))[..., None]
            + gather(x0+1, y0) * (dx*(1-dy))[..., None]
            + gather(x0, y0+1) * ((1-dx)*dy)[..., None]
            + gather(x0+1, y0+1) * (dx*dy)[..., None])


class SeparableBlock(nn.Module):
    def __init__(self, incoming, outgoing):
        super().__init__()
        self.depthwise = nn.Conv2d(incoming, incoming, 3, stride=2, padding=1, groups=incoming, bias=False)
        self.pointwise = nn.Conv2d(incoming, outgoing, 1, bias=False)
        self.norm = nn.LayerNorm(outgoing)

    def __call__(self, x):
        return nn.gelu(self.norm(self.pointwise(self.depthwise(x))))


class DeformableAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, heads, points = config.hidden_dim, config.num_heads, config.num_points
        self.heads, self.points = heads, points
        self.value = nn.Linear(d, d)
        self.offsets = nn.Linear(d, heads * points * 2)
        self.weights = nn.Linear(d, heads * points)
        self.output = nn.Linear(d, d)
        # Spread samples radially around each reference instead of identical points.
        angles = np.arange(heads) * 2 * math.pi / heads
        radial = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        radial /= np.max(np.abs(radial), axis=-1, keepdims=True)
        grid = radial[:, None, :] * np.arange(1, points+1)[None, :, None]
        self.offsets.weight = mx.zeros_like(self.offsets.weight)
        self.offsets.bias = mx.array(grid.reshape(-1), dtype=mx.float32)
        self.weights.weight = mx.zeros_like(self.weights.weight)
        self.weights.bias = mx.zeros_like(self.weights.bias)

    def __call__(self, query, feature, reference):
        b, q, d = query.shape
        h, w = feature.shape[1:3]
        offsets = self.offsets(query).reshape(b, q, self.heads, self.points, 2)
        locations = reference[:, :, None, None, :2] + offsets / mx.array([w, h])
        weights = mx.softmax(self.weights(query).reshape(b, q, self.heads, self.points), axis=-1)
        samples = sample_bilinear(self.value(feature), locations)
        return self.output(mx.sum(samples * weights[..., None], axis=3).reshape(b, q, d))


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.self_attention = nn.MultiHeadAttention(d, config.num_heads)
        self.cross_attention = DeformableAttention(config)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn1, self.ffn2 = nn.Linear(d, d * 2), nn.Linear(d * 2, d)

    def __call__(self, x, position, feature, reference):
        q = x + position
        x = self.norm1(x + self.self_attention(q, q, x))
        x = self.norm2(x + self.cross_attention(x + position, feature, reference))
        return self.norm3(x + self.ffn2(nn.gelu(self.ffn1(x))))


class BoxHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.layers = [nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, 4)]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        return self.layers[-1](x)


class LiteDETR(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        channels = (3,) + tuple(config.backbone_channels)
        self.backbone = [SeparableBlock(channels[i], channels[i+1]) for i in range(4)]
        self.projector = nn.Linear(channels[-1], d)
        self.query = nn.Embedding(config.num_queries, d)
        self.reference = nn.Embedding(config.num_queries, 4)
        self.reference.weight = mx.zeros_like(self.reference.weight)
        self.query_position = nn.Linear(4, d)
        self.decoder = [DecoderLayer(config) for _ in range(config.decoder_layers)]
        self.box_heads = [BoxHead(d) for _ in self.decoder]
        self.class_heads = [nn.Linear(d, config.num_classes) for _ in self.decoder]
        for head in self.class_heads:
            head.bias = mx.full(head.bias.shape, -math.log(99))

    def __call__(self, images):
        if images.ndim != 4 or images.shape[-1] != 3 or images.shape[0] < 1:
            raise ValueError("Expected nonempty NHWC RGB batch")
        if images.shape[1:3] != (self.config.resolution, self.config.resolution):
            raise ValueError("Image size must match config.resolution")
        feature = images
        for block in self.backbone:
            feature = block(feature)
        feature = self.projector(feature)
        feature = feature + position_encoding(feature.shape[1], feature.shape[2], self.config.hidden_dim)[None]
        b = images.shape[0]
        x = mx.broadcast_to(self.query.weight[None], (b, *self.query.weight.shape))
        reference = mx.broadcast_to(mx.sigmoid(self.reference.weight)[None], (b, self.config.num_queries, 4))
        outputs = []
        for layer, box_head, class_head in zip(self.decoder, self.box_heads, self.class_heads):
            x = layer(x, self.query_position(reference), feature, reference)
            boxes = mx.sigmoid(box_head(x) + inverse_sigmoid(reference))
            outputs.append({"pred_logits": class_head(x), "pred_boxes": boxes})
            reference = mx.stop_gradient(boxes)
        return outputs[-1] | {"aux_outputs": outputs[:-1]}
