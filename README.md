# RF-DETR MLX Lite

An experimental, trainable, lightweight detector built in **MLX**, using
[Roboflow RF-DETR](https://github.com/roboflow/rf-detr) as an architecture reference.
It runs on MLX CPU in this cloud environment; on Apple Silicon, MLX uses Metal.

**This is an RF-DETR-inspired student architecture, not a numerically equivalent
port. No pretrained weights are included.** Upstream `.pth` weights cannot load
into this architecture. Train it on your dataset before expecting useful detections.
Detection accuracy and Apple GPU performance have not been measured.

## What is implemented

- Four stride-2 depthwise-separable convolution blocks replacing the DINOv2 backbone.
- Single-scale stride-16 features with sine/cosine spatial position encoding.
- Learned object queries, multi-head self-attention and deformable cross-attention
  with differentiable bilinear sampling and zero padding.
- Two decoder layers, iterative normalized box refinement and auxiliary losses.
- Hungarian assignment, sigmoid focal classification, L1 box loss and generalized IoU.
- COCO bounding-box data loading, training, image prediction and strict safetensors checkpoints.
- Optional 4-bit or 8-bit weight quantization for compatible linear layers.
  Embeddings, convolutions and projections with incompatible dimensions remain floating point.

RF-DETR's windowed DINOv2 backbone, multiscale projection, two-stage encoder query
selection, grouped training, segmentation, upstream weight conversion, teacher
distillation and COCO mAP evaluation are not implemented. The small training loop
has no augmentation, scheduler, optimizer resume, or validation split; it is a
baseline for developing and evaluating a lightweight model, not a production trainer.

## Install

Python 3.11+ is required. The cloud setup was tested with Python 3.12 on Linux x86_64.

```bash
cd /workspace/rf-detr_MLX
python -m venv /workspace/.venvs/rfdetr-mlx
source /workspace/.venvs/rfdetr-mlx/bin/activate
python -m pip install -r requirements-linux-dev.txt
python -m pip install -e '.[cpu,dev]'
```

On an Apple Silicon Mac, use a local virtual environment and the native MLX package:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Do not install `mlx-cpu` or the Linux requirements file on macOS. Check available
hardware with `python -c 'import mlx.core as mx; print(mx.default_device())'`.
No services, API keys, PyTorch, or external weight downloads are needed.

## Verify and benchmark

```bash
python -m pytest -q
rfdetr-mlx smoke --steps 5 --output artifacts/smoke
rfdetr-mlx benchmark --preset nano --runs 20
rfdetr-mlx benchmark --preset tiny --runs 20
```

`smoke` uses a 32×32 test model and synthetic targets; its checkpoint is for
verification only. `benchmark` uses random initialization unless supplied with
`--checkpoint`. Timing evaluates lazy MLX outputs, excludes warmup and preprocessing,
and reports median and p95 wall-clock latency. It is not an accuracy benchmark.

| Preset | Resolution | Width | Queries | Layers | Parameters | FP32 parameter storage |
| --- | --- | --- | --- | --- | --- | --- |
| nano | 256 | 96 | 50 | 2 | 274,283 | 1.10 MB |
| tiny | 320 | 128 | 100 | 2 | 481,203 | 1.92 MB |

Counts assume 80 classes. Reference RF-DETR Nano uses a DINOv2 small backbone,
256-wide decoder and 300 queries at 384 resolution; these lightweight variants
trade representational capacity, image detail and maximum detections for size.
No upstream-versus-student accuracy or latency comparison has been run.

## Train on a COCO bounding-box dataset

```bash
rfdetr-mlx train \
  --annotations /path/to/train/_annotations.coco.json \
  --images /path/to/train \
  --preset nano --epochs 50 --batch-size 2 \
  --learning-rate 0.0001 --output artifacts/my-model
```

Category IDs are sorted and mapped to contiguous zero-based model labels; the
mapping is stored in the checkpoint. Crowd annotations are ignored. Images resize
to a square with ImageNet normalization; boxes remain normalized to the original
image dimensions. An image with more non-crowd objects than queries is rejected
rather than silently dropping objects. A larger query budget can be configured
through the Python API. The training command starts from scratch and saves model
weights after each epoch. Training duration above is an example, not a validated
recipe for achieving a particular accuracy.

```bash
rfdetr-mlx predict --checkpoint artifacts/my-model \
  --image /path/to/image.jpg --threshold 0.3
```

Prediction returns JSON with pixel `xyxy` boxes, sigmoid scores, contiguous labels,
and original category IDs when present. It retains one class per query, thresholds,
and caps results at 100. No NMS is applied. Very low scores from an untrained
checkpoint should not be interpreted as working detection.

## Reduce checkpoint size

```bash
rfdetr-mlx quantize --checkpoint artifacts/my-model \
  --output artifacts/my-model-int4 --bits 4
rfdetr-mlx benchmark --checkpoint artifacts/my-model-int4 --runs 20
rfdetr-mlx predict --checkpoint artifacts/my-model-int4 --image /path/to/image.jpg
```

Quantization is weight-only, uses affine groups of 32, and writes a separate
checkpoint. Re-evaluate accuracy on a held-out dataset before deploying it.
Smaller weights do not imply lower latency: quantized kernels were slower on the
tested CPU. Quantized checkpoints are intended for inference, not continued training.

## Validation in this cloud instance

Linux x86_64, Python 3.12.14, MLX/MLX CPU 0.32.3:

- Seven tests passed, covering sampling values and coordinate gradients, batched
  forward passes, matching, training updates, empty targets, checkpoint roundtrips,
  quantized forward passes, COCO loading and pixel-coordinate postprocessing.
- Five synthetic training steps reduced loss from 7.89 to 4.05; checkpoint max
  output error after reload was zero.
- A full-resolution nano model completed one training step through the COCO CLI,
  saved its category mapping, and completed the prediction command.
- Five timed CPU runs: nano median 46.7 ms, tiny median 185.3 ms. Nano int4 median
  56.9 ms in a separate run. These are short, hardware-specific smoke measurements.
- Nano checkpoint size: about 1.11 MB floating point and 0.28 MB int4, approximately
  75% smaller. This is mixed quantization, not all parameters stored at four bits.

Generated checkpoints and synthetic fixtures live under ignored `artifacts/`.
They are not pretrained models. Apple Metal validation, real-data training and
held-out mAP evaluation remain necessary to judge the accuracy/size tradeoff.

## Python API

```python
import mlx.core as mx
from rfdetr_mlx import LiteDETR, ModelConfig

model = LiteDETR(ModelConfig.preset("nano", num_classes=3))
outputs = model(mx.zeros((1, 256, 256, 3)))  # NHWC normalized image batch
mx.eval(outputs)
print(outputs["pred_logits"].shape, outputs["pred_boxes"].shape)
```

Checkpoint and training helpers are in `rfdetr_mlx.checkpoint` and
`rfdetr_mlx.training`. Boxes are normalized `(center_x, center_y, width, height)`;
classification uses independent sigmoid probabilities with no background channel.

## Attribution

Apache 2.0; see `LICENSE` and `NOTICE`. Upstream reference commit:
`eca736acab9f6fe93f2cbc82a1d9f6edfb53f0ea`. This is not an official Roboflow release.
