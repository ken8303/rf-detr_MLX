"""Host-side Hungarian matching and differentiable MLX set prediction losses."""
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment


def corners(boxes):
    return mx.concatenate([boxes[..., :2] - boxes[..., 2:]/2, boxes[..., :2] + boxes[..., 2:]/2], axis=-1)


def generalized_iou(a, b):
    a, b = corners(a), corners(b)
    intersection = mx.prod(mx.maximum(mx.minimum(a[:, None, 2:], b[None, :, 2:]) - mx.maximum(a[:, None, :2], b[None, :, :2]), 0), axis=-1)
    area_a = mx.prod(mx.maximum(a[:, 2:] - a[:, :2], 0), axis=-1)
    area_b = mx.prod(mx.maximum(b[:, 2:] - b[:, :2], 0), axis=-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    enclosure = mx.prod(mx.maximum(mx.maximum(a[:, None, 2:], b[None, :, 2:]) - mx.minimum(a[:, None, :2], b[None, :, :2]), 0), axis=-1)
    return intersection / mx.maximum(union, 1e-7) - (enclosure-union) / mx.maximum(enclosure, 1e-7)


def match(outputs, targets):
    if len(targets) != outputs["pred_logits"].shape[0]:
        raise ValueError("One target required per image")
    matches = []
    for logits, boxes, target in zip(outputs["pred_logits"], outputs["pred_boxes"], targets):
        labels = np.asarray(target["labels"])
        truth = np.asarray(target["boxes"])
        if labels.ndim != 1 or truth.shape != (len(labels), 4) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Targets require integer labels [N] and boxes [N,4]")
        if len(labels) > len(boxes):
            raise ValueError("More objects than queries; increase num_queries")
        if len(labels) and (labels.min() < 0 or labels.max() >= logits.shape[-1]):
            raise ValueError("Target label outside model class range")
        if not np.isfinite(truth).all() or (truth < 0).any() or (truth > 1).any() or (truth[:, 2:] <= 0).any():
            raise ValueError("Boxes must be finite normalized cxcywh with positive dimensions")
        if not len(labels):
            matches.append((mx.array([], dtype=mx.int32), mx.array([], dtype=mx.int32)))
            continue
        p = mx.clip(mx.sigmoid(logits), 1e-6, 1-1e-6)
        positive = -.25 * (1-p)**2 * mx.log(p)
        negative = -.75 * p**2 * mx.log(1-p)
        cost_class = positive[:, target["labels"]] - negative[:, target["labels"]]
        cost_l1 = mx.sum(mx.abs(boxes[:, None] - target["boxes"][None]), axis=-1)
        cost = cost_class + 5 * cost_l1 - 2 * generalized_iou(boxes, target["boxes"])
        row, col = linear_sum_assignment(np.asarray(cost))
        matches.append((mx.array(row, dtype=mx.int32), mx.array(col, dtype=mx.int32)))
    return matches


def set_loss(outputs, targets, matches):
    normalization = max(sum(len(t["labels"]) for t in targets), 1)
    total = mx.array(0., dtype=mx.float32)
    for logits, boxes, target, (row, col) in zip(outputs["pred_logits"], outputs["pred_boxes"], targets, matches):
        desired = mx.zeros_like(logits)
        if len(row):
            desired[row, target["labels"][col]] = 1
        p = mx.sigmoid(logits)
        bce = mx.maximum(logits, 0) - logits * desired + mx.log1p(mx.exp(-mx.abs(logits)))
        pt = p * desired + (1-p) * (1-desired)
        alpha = .25 * desired + .75 * (1-desired)
        total = total + mx.sum(alpha * (1-pt)**2 * bce) / normalization
        if len(row):
            selected, truth = boxes[row], target["boxes"][col]
            total = total + 5 * mx.sum(mx.abs(selected-truth)) / normalization
            total = total + 2 * mx.sum(1 - mx.diag(generalized_iou(selected, truth))) / normalization
    return total


def train_step(model, optimizer, images, targets):
    model.train()
    predictions = model(images)
    all_outputs = [*predictions["aux_outputs"], predictions]
    assignments = [match(output, targets) for output in all_outputs]

    def loss_fn(module):
        predictions = module(images)
        return sum(set_loss(output, targets, assignment) for output, assignment in zip([*predictions["aux_outputs"], predictions], assignments))

    loss, gradients = nn.value_and_grad(model, loss_fn)(model)
    mx.eval(loss, gradients)
    if not bool(mx.isfinite(loss)):
        raise FloatingPointError("Nonfinite training loss")
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state)
    return float(loss)
