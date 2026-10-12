import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from PIL import Image

from .checkpoint import load_checkpoint, quantize, save_checkpoint
from .config import ModelConfig
from .data import CocoDataset, postprocess, preprocess
from .model import LiteDETR
from .training import train_step


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description="Experimental MLX lightweight detector (requires trained weights for useful detections)")
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark", help="Synthetic forward timing; random initialization unless --checkpoint given")
    bench.add_argument("--preset", choices=["tiny", "nano"], default="nano")
    bench.add_argument("--checkpoint")
    bench.add_argument("--runs", type=positive_int, default=10)
    bench.add_argument("--seed", type=int, default=0)
    smoke = sub.add_parser("smoke", help="Forward + backward + checkpoint roundtrip using synthetic targets")
    smoke.add_argument("--output", default="artifacts/smoke")
    smoke.add_argument("--steps", type=positive_int, default=3)
    train = sub.add_parser("train", help="Train from scratch on COCO-format bounding boxes")
    train.add_argument("--annotations", required=True)
    train.add_argument("--images", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--preset", choices=["tiny", "nano"], default="nano")
    train.add_argument("--epochs", type=positive_int, default=1)
    train.add_argument("--batch-size", type=positive_int, default=1)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=0)
    predict = sub.add_parser("predict")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--image", required=True)
    predict.add_argument("--threshold", type=float, default=.3)
    compress = sub.add_parser("quantize", help="Weight-only quantization of compatible linear layers")
    compress.add_argument("--checkpoint", required=True)
    compress.add_argument("--output", required=True)
    compress.add_argument("--bits", type=int, choices=[4, 8], default=4)
    args = parser.parse_args()
    if args.command == "benchmark":
        mx.random.seed(args.seed)
        model = load_checkpoint(args.checkpoint)[0] if args.checkpoint else LiteDETR(ModelConfig.preset(args.preset))
        model.eval()
        mx.eval(model.parameters())
        images = mx.random.normal((1, model.config.resolution, model.config.resolution, 3))
        mx.eval(images)
        mx.eval(model(images))
        samples = []
        for _ in range(args.runs):
            start = time.perf_counter()
            mx.eval(model(images))
            samples.append((time.perf_counter()-start) * 1000)
        arrays = [v for _, v in tree_flatten(model.parameters())]
        print(json.dumps({"device": str(mx.default_device()), "parameters_or_packed_elements": sum(a.size for a in arrays), "parameter_bytes": sum(a.nbytes for a in arrays), "median_ms": float(np.median(samples)), "p95_ms": float(np.percentile(samples, 95)), "runs": args.runs, "resolution": model.config.resolution, "checkpoint": args.checkpoint, "trained_accuracy": "not measured"}, indent=2))
    elif args.command == "smoke":
        mx.random.seed(0)
        model = LiteDETR(ModelConfig(num_classes=2, resolution=32, hidden_dim=32, num_queries=4, backbone_channels=(8, 16, 24, 32)))
        images = mx.random.normal((1, 32, 32, 3))
        targets = [{"labels": mx.array([1]), "boxes": mx.array([[.5, .5, .3, .3]])}]
        optimizer = optim.AdamW(learning_rate=1e-3)
        losses = [train_step(model, optimizer, images, targets) for _ in range(args.steps)]
        model.eval()
        before = model(images)
        save_checkpoint(model, args.output)
        restored, _ = load_checkpoint(args.output)
        after = restored(images)
        error = float(mx.max(mx.abs(before["pred_logits"] - after["pred_logits"])))
        if error > 1e-6:
            raise RuntimeError("Checkpoint roundtrip mismatch")
        print(json.dumps({"losses": losses, "checkpoint_max_error": error, "output": args.output}, indent=2))
    elif args.command == "train":
        if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("learning-rate must be finite and positive")
        mx.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        dataset = CocoDataset(args.annotations, args.images, ModelConfig.preset(args.preset).resolution)
        if not len(dataset):
            parser.error("Dataset has no images")
        model = LiteDETR(ModelConfig.preset(args.preset, num_classes=len(dataset.categories)))
        optimizer = optim.AdamW(learning_rate=args.learning_rate)
        for epoch in range(args.epochs):
            order, losses = rng.permutation(len(dataset)), []
            for start in range(0, len(order), args.batch_size):
                samples = [dataset[int(i)] for i in order[start:start+args.batch_size]]
                losses.append(train_step(model, optimizer, mx.stack([s[0] for s in samples]), [s[1] for s in samples]))
            save_checkpoint(model, args.output, categories=dataset.categories)
            print(json.dumps({"epoch": epoch+1, "mean_loss": float(np.mean(losses)), "steps": len(losses)}), flush=True)
    elif args.command == "predict":
        model, metadata = load_checkpoint(args.checkpoint)
        with Image.open(args.image) as image:
            tensor, size = preprocess(image, model.config.resolution)
        result = postprocess(model(tensor[None]), [size], threshold=args.threshold)[0]
        if metadata.get("categories"):
            result["category_ids"] = [metadata["categories"][label]["id"] for label in result["labels"]]
        print(json.dumps(result, indent=2))
    elif args.command == "quantize":
        if Path(args.output).resolve() == Path(args.checkpoint).resolve():
            parser.error("Use a separate output directory to preserve floating-point weights")
        model, metadata = load_checkpoint(args.checkpoint)
        if metadata.get("quantization"):
            parser.error("Checkpoint is already quantized")
        specification = {"bits": args.bits, "group_size": 32}
        quantize(model, **specification)
        save_checkpoint(model, args.output, quantization=specification, categories=metadata.get("categories"))
        print(json.dumps({"output": args.output, "weight_bytes": (Path(args.output)/"model.safetensors").stat().st_size}))


if __name__ == "__main__":
    main()
