import json

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import pytest
from PIL import Image

from rfdetr_mlx import LiteDETR, ModelConfig
from rfdetr_mlx.checkpoint import load_checkpoint, quantize, save_checkpoint
from rfdetr_mlx.data import CocoDataset, postprocess
from rfdetr_mlx.model import sample_bilinear
from rfdetr_mlx.training import generalized_iou, match, train_step


def small_model():
    mx.random.seed(3)
    return LiteDETR(ModelConfig(num_classes=2, resolution=32, hidden_dim=32, num_queries=4, backbone_channels=(8, 16, 24, 32)))


def test_sampling_centers_padding_and_gradient():
    feature = mx.array([[[[1.], [2.]], [[3.], [4.]]]])
    locations = mx.array([[[[[.25, .25], [.5, .5], [-1., -1.]]]]])
    np.testing.assert_allclose(np.asarray(sample_bilinear(feature, locations)).reshape(-1), [1, 2.5, 0])
    gradient = mx.grad(lambda loc: mx.sum(sample_bilinear(feature, loc)))(mx.array([[[[[.5, .5]]]]]))
    np.testing.assert_allclose(np.asarray(gradient).reshape(-1), [2, 4])


def test_forward_and_input_validation():
    model = small_model()
    result = model(mx.zeros((2, 32, 32, 3)))
    assert result["pred_logits"].shape == (2, 4, 2)
    assert result["pred_boxes"].shape == (2, 4, 4)
    assert len(result["aux_outputs"]) == 1
    assert bool(mx.all(mx.isfinite(result["pred_logits"])))
    assert bool(mx.all((result["pred_boxes"] >= 0) & (result["pred_boxes"] <= 1)))
    with pytest.raises(ValueError):
        model(mx.zeros((1, 3, 32, 32)))


def test_matching_permutation_and_empty():
    boxes = mx.array([[.2, .2, .1, .1], [.8, .8, .2, .2]])
    result = {"pred_logits": mx.array([[[8., -8.], [-8., 8.]]]), "pred_boxes": boxes[None]}
    target = {"labels": mx.array([1, 0]), "boxes": boxes[::-1]}
    row, col = match(result, [target])[0]
    np.testing.assert_array_equal(np.asarray(row), [0, 1])
    np.testing.assert_array_equal(np.asarray(col), [1, 0])
    np.testing.assert_allclose(np.diag(np.asarray(generalized_iou(boxes, boxes))), 1, atol=1e-6)
    empty = {"labels": mx.array([], dtype=mx.int32), "boxes": mx.zeros((0, 4))}
    assert len(match(result, [empty])[0][0]) == 0


def test_training_updates_backbone_and_head():
    model = small_model()
    images = mx.random.normal((1, 32, 32, 3))
    targets = [{"labels": mx.array([1]), "boxes": mx.array([[.5, .5, .2, .3]])}]
    before_backbone = np.asarray(model.backbone[0].pointwise.weight).copy()
    before_head = np.asarray(model.class_heads[-1].weight).copy()
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0)
    losses = [train_step(model, optimizer, images, targets) for _ in range(5)]
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]
    assert not np.array_equal(before_backbone, np.asarray(model.backbone[0].pointwise.weight))
    assert not np.array_equal(before_head, np.asarray(model.class_heads[-1].weight))
    loss = train_step(model, optimizer, images, [{"labels": mx.array([], dtype=mx.int32), "boxes": mx.zeros((0, 4))}])
    assert np.isfinite(loss)


def test_checkpoint_and_quantization(tmp_path):
    model = small_model()
    images = mx.random.normal((1, 32, 32, 3))
    expected = model(images)
    save_checkpoint(model, tmp_path / "float")
    restored, _ = load_checkpoint(tmp_path / "float")
    np.testing.assert_allclose(np.asarray(expected["pred_logits"]), np.asarray(restored(images)["pred_logits"]), atol=1e-6)
    quantize(model)
    save_checkpoint(model, tmp_path / "int4", quantization={"bits": 4, "group_size": 32})
    restored, _ = load_checkpoint(tmp_path / "int4")
    actual = restored(images)
    np.testing.assert_allclose(np.asarray(model(images)["pred_logits"]), np.asarray(actual["pred_logits"]), atol=1e-6)
    assert bool(mx.all(mx.isfinite(actual["pred_boxes"])))
    assert (tmp_path / "int4/model.safetensors").stat().st_size < (tmp_path / "float/model.safetensors").stat().st_size


def test_coco_and_postprocess(tmp_path):
    Image.new("RGB", (100, 50)).save(tmp_path / "image.png")
    document = {"images": [{"id": 1, "file_name": "image.png"}], "categories": [{"id": 7, "name": "object"}], "annotations": [{"image_id": 1, "category_id": 7, "bbox": [10, 5, 40, 20]}]}
    (tmp_path / "annotations.json").write_text(json.dumps(document))
    dataset = CocoDataset(tmp_path / "annotations.json", tmp_path, 32)
    image, target = dataset[0]
    assert image.shape == (32, 32, 3)
    np.testing.assert_allclose(np.asarray(target["boxes"]), [[.3, .3, .4, .4]])
    assert int(target["labels"][0]) == 0
    output = {"pred_logits": mx.array([[[10.]]]), "pred_boxes": target["boxes"][None]}
    result = postprocess(output, [(100, 50)])[0]
    np.testing.assert_allclose(result["boxes"], [[10, 5, 50, 25]], atol=1e-5)
    assert result["labels"] == [0]


def test_invalid_config_and_targets():
    with pytest.raises(ValueError):
        ModelConfig(hidden_dim=95)
    model = small_model()
    outputs = model(mx.zeros((1, 32, 32, 3)))
    with pytest.raises(ValueError):
        match(outputs, [{"labels": mx.array([2]), "boxes": mx.array([[.5, .5, .2, .2]])}])
