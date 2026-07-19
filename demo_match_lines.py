#!/usr/bin/env python3
"""Match UPAL point-seeded line segments by their endpoint descriptors."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from upal import load_model
from upal.demo_utils import draw_features, load_image, side_by_side
from upal.postprocess import detect_lines, match_lines_from_endpoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", type=Path, default=Path("assets/boat1.png"))
    parser.add_argument("--image1", type=Path, default=Path("assets/boat2.png"))
    parser.add_argument("--weights", type=Path, default=Path("weights/upal.tar"))
    parser.add_argument("--output", type=Path, default=Path("outputs/line_matches.png"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-size", type=int, default=640)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--max-matches", type=int, default=50)
    args = parser.parse_args()

    model = load_model(args.weights, device=args.device, max_num_keypoints=args.max_keypoints)
    images, tensors, predictions, lines = [], [], [], []
    for path in (args.image0, args.image1):
        image, tensor = load_image(path, args.max_size)
        model_input = tensor.to(args.device)
        with torch.inference_mode():
            prediction = model(model_input)
        images.append(image)
        tensors.append(tensor)
        predictions.append(prediction)
        lines.append(
            detect_lines(
                model_input,
                prediction["line_distance_field"][0],
                prediction["keypoints"][0],
            )
        )
    if not all(map(len, lines)):
        line_matches = np.empty((0, 2), dtype=np.int64)
        scores = np.empty((0,), dtype=np.float32)
    else:
        with torch.inference_mode():
            endpoint_descriptors = [
                model.describe_keypoints(
                    tensor.to(args.device),
                    torch.from_numpy(line.reshape(1, -1, 2)).to(args.device),
                )[0].reshape(-1, 2, prediction["descriptors"].shape[-1])
                for tensor, line, prediction in zip(tensors, lines, predictions)
            ]
        line_matches, scores = match_lines_from_endpoints(*endpoint_descriptors)
    line_matches, scores = line_matches[: args.max_matches], scores[: args.max_matches]

    canvas, offset = side_by_side(*[draw_features(image, lines=line) for image, line in zip(images, lines)])
    rng = np.random.default_rng(11)
    for (index0, index1), score in zip(line_matches, scores):
        midpoint0 = np.rint(lines[0][index0].mean(axis=0)).astype(int)
        midpoint1 = np.rint(lines[1][index1].mean(axis=0)).astype(int)
        color = tuple(int(v) for v in rng.integers(80, 256, 3))
        cv2.line(canvas, tuple(midpoint0), (int(midpoint1[0] + offset), int(midpoint1[1])), color, 1, cv2.LINE_AA)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), canvas)
    print(f"point-seeded lines: {len(lines[0])}, {len(lines[1])}; line matches shown: {len(line_matches)}")
    print(f"visualization: {args.output}")


if __name__ == "__main__":
    main()
