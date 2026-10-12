import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageOps


def preprocess(image, resolution):
    """Resize RGB input to a square; normalized boxes map back to original size."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    size = image.size
    array = np.asarray(image.resize((resolution, resolution), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    array = (array - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    return mx.array(array), size


class CocoDataset:
    """COCO bounding-box dataset; category IDs become contiguous zero-based labels."""
    def __init__(self, annotations, image_root, resolution):
        document = json.loads(Path(annotations).read_text())
        self.images = document["images"]
        self.categories = sorted(document["categories"], key=lambda c: c["id"])
        self.label_map = {category["id"]: i for i, category in enumerate(self.categories)}
        if not self.label_map or len(self.label_map) != len(self.categories):
            raise ValueError("Categories must be nonempty with unique IDs")
        self.annotations = {image["id"]: [] for image in self.images}
        for annotation in document["annotations"]:
            if not annotation.get("iscrowd", 0):
                self.annotations[annotation["image_id"]].append(annotation)
        self.root, self.resolution = Path(image_root).resolve(), resolution

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        record = self.images[index]
        path = (self.root / record["file_name"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Image path escapes image_root")
        with Image.open(path) as image:
            # COCO boxes are defined in the stored pixel orientation.
            image = image.convert("RGB")
            tensor, (width, height) = preprocess(image, self.resolution)
        labels, boxes = [], []
        for annotation in self.annotations[record["id"]]:
            x, y, w, h = annotation["bbox"]
            coords = np.asarray([x, y, x+w, y+h], dtype=np.float32)
            if not np.isfinite(coords).all():
                raise ValueError("Nonfinite annotation box")
            x0, y0, x1, y1 = np.clip(coords, [0, 0, 0, 0], [width, height, width, height])
            if x1 <= x0 or y1 <= y0:
                continue
            labels.append(self.label_map[annotation["category_id"]])
            boxes.append([(x0+x1)/(2*width), (y0+y1)/(2*height), (x1-x0)/width, (y1-y0)/height])
        return tensor, {"labels": mx.array(labels, dtype=mx.int32), "boxes": mx.array(np.asarray(boxes, dtype=np.float32).reshape(-1, 4))}


def postprocess(outputs, sizes, threshold=.3, top_k=100):
    if not 0 <= threshold <= 1 or top_k <= 0:
        raise ValueError("threshold must be in [0,1] and top_k positive")
    probabilities = np.asarray(mx.sigmoid(outputs["pred_logits"]))
    boxes = np.asarray(outputs["pred_boxes"])
    if len(sizes) != len(boxes):
        raise ValueError("One original image size required per batch item")
    results = []
    for scores, cxcywh, (width, height) in zip(probabilities, boxes, sizes):
        # One class per query prevents duplicate class hypotheses for the same box.
        labels = scores.argmax(axis=-1)
        scores = scores.max(axis=-1)
        order = np.argsort(-scores, kind="stable")[:top_k]
        order = order[scores[order] >= threshold]
        selected = cxcywh[order]
        xyxy = np.concatenate([selected[:, :2] - selected[:, 2:]/2, selected[:, :2] + selected[:, 2:]/2], axis=-1)
        xyxy = np.clip(xyxy, 0, 1) * np.array([width, height, width, height])
        results.append({"boxes": xyxy.tolist(), "scores": scores[order].tolist(), "labels": labels[order].tolist()})
    return results
