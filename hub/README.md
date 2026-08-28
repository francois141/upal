---
license: apache-2.0
library_name: upal
pipeline_tag: keypoint-detection
tags:
  - keypoint-detection
  - line-detection
  - image-matching
  - local-features
  - pytorch_model_hub_mixin
  - model_hub_mixin
  - arxiv:2608.19894
---

# UPAL: Unified and Efficient Point-Line Local Features (ECCV 2026)

François Costa<sup>*</sup> · Raphael Kreft<sup>*</sup> · Eckhard Goedeke · Felix Möller · Hardik Shah ·
Ramanathan Rajaraman · Shaohui Liu · Rémi Pautrat · Marc Pollefeys

<small><sup>*</sup> denotes equal contribution</small>

Joint **keypoint + line** local feature extractor. One forward pass predicts sub-pixel
keypoints with confidence scores, 128-D L2-normalised descriptors, a dense keypoint/junction
heatmap and a dense line distance field. In the released `0.1.0` inference package, line
segments are obtained by seeding a modified LSD detector with the learned keypoints and
filtering its proposals with the distance field.

- Paper: [arXiv 2608.19894](https://arxiv.org/abs/2608.19894)
- Code: [github.com/francois141/upal](https://github.com/francois141/upal)

![UPAL on the boat pair](assets/boat_demo.png)

*Red: learned keypoints · green: line segments supported by the learned distance field ·
coloured links: mutual-nearest descriptor matches.*

## Installation

```bash
pip install "upal[lines]"   # network + point-seeded line detector (prebuilt wheels)
pip install upal            # points only, no compiled dependency
```

## Usage

```python
import cv2
import torch
from upal import UPAL, mutual_nearest_neighbors, match_lines_from_endpoints

model = UPAL.from_pretrained("rkreft/upal").to("cuda" if torch.cuda.is_available() else "cpu")

def read(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

feats0, feats1 = model.extract(read("img0.png")), model.extract(read("img1.png"))
feats0["keypoints"]            # N x 2 pixel (x, y)
feats0["descriptors"]          # N x 128, L2-normalised
feats0["keypoint_scores"]      # N
feats0["keypoint_dispersity"]  # N, spread of the score peak (lower = sharper)
feats0["keypoint_heatmap"]     # H x W
feats0["line_distance_field"]  # H x W, distance to the nearest line in pixels
feats0["lines"]                # L x 2 x 2 endpoints (needs the `lines` extra)

# Point matching: mutual nearest neighbours on descriptors.
point_matches = mutual_nearest_neighbors(feats0["descriptors"], feats1["descriptors"])  # M x 2 indices

# Line matching: describe both endpoints of each segment, then solve a one-to-one assignment.
desc0 = model.describe_lines(read("img0.png"), feats0["lines"])  # L0 x 2 x 128
desc1 = model.describe_lines(read("img1.png"), feats1["lines"])  # L1 x 2 x 128
line_matches, scores = match_lines_from_endpoints(desc0, desc1)   # K x 2 indices, K scores (NumPy)
```

Without the `lines` extra, call `model.extract(image, lines=False)`; `lines` is then empty
(`0 x 2 x 2`) and everything else is unchanged.

`extract(image, lines=True, max_lines=200, min_line_length=25.0, max_line_distance=2.0)`
takes a `C x H x W` tensor in `[0, 1]` (RGB or grayscale); the number of keypoints is set by
`UPAL.from_pretrained("rkreft/upal", max_num_keypoints=2048)`. Images are padded to a
multiple of 32 internally and outputs are returned in input-image coordinates.

## Paper and release implementation

The shared network architecture and weights implement the encoder and prediction heads
described in the paper. The line and matching post-processing in the `0.1.0` inference package
differs in two documented ways:

- The paper's Fast-LSD uses a stride-2 seed grid and retains the 20% of seed pixels with the
  lowest predicted distance-field values. This package instead seeds `points-lsd` with the
  learned keypoints and filters the resulting segments by their mean predicted distance-field
  value.
- The paper's visual-localisation experiment solves endpoint-based line assignment with
  Sinkhorn. The packaged `match_lines_from_endpoints` helper uses an exact maximum-weight
  one-to-one assignment.

These differences are confined to post-processing; they do not alter the released network
weights or its point, descriptor, heatmap, and distance-field predictions.

## Model details

- **Encoder:** ALIKED-style four-level encoder (16/32/64/128 channels) with deformable
  convolutions in the two deepest stages, fused to a 128-channel full-resolution feature map.
- **Heads:** keypoint/junction score head; sparse deformable descriptor head (SDDH, 16
  samples, 128-D); line distance-field decoder (values in `[0, 5]` px).
- **Post-processing:** NMS (radius 3) with differentiable sub-pixel refinement;
  keypoint-seeded LSD (`points-lsd`) filtered by the mean distance-field value along the
  segment; mutual-nearest-neighbour point matching; endpoint-descriptor line matching with a
  maximum-weight assignment.
- **Weights:** this repository holds the inference subset of the trained checkpoint
  (`model.safetensors`, 0.79M parameters, 3.2 MB) and its configuration (`config.json`).
- **Training:** distilled from SuperPoint, ALIKED, DaD and DeepLSD teachers; see the paper
  for datasets, losses and evaluation.

## Files

| File | Content |
|---|---|
| `model.safetensors` | inference weights |
| `config.json` | `max_num_keypoints`, `nms_radius`, `line_neighborhood` |
| `assets/boat_demo.png` | demo visualisation |

## License

Apache-2.0 (weights and code). Line detection uses [`points-lsd`](https://pypi.org/project/points-lsd/),
whose LSD core is AGPL-3.0-or-later.

## Citation

```bibtex
@misc{costa2026unifiedefficientpointlinelocal,
  title={Unified and Efficient Point-Line Local Features},
  author={François Costa and Raphael Kreft and Eckhard Goedeke and Felix Möller and Hardik Shah and Ramanathan Rajaraman and Shaohui Liu and Rémi Pautrat and Marc Pollefeys},
  year={2026},
  eprint={2608.19894},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.19894},
}
```

## Acknowledgements

Parts of the code reuse [ALIKED](https://github.com/Shiaoming/ALIKED) and
[glue-factory](https://github.com/cvg/glue-factory). We thank the authors of SuperPoint,
ALIKED, DaD and DeepLSD for releasing the pre-trained models used as teachers.
