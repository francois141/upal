<div align="center">
  <h1>UPAL: Unified and Efficient Point-Line Local Features - ECCV 2026</h1>
  <p>
    François Costa * · Raphael Kreft * · Eckhard Goedeke · Felix Möller · Hardik Shah · Ramanathan Rajaraman · Shaohui Liu · Rémi Pautrat · Marc Pollefeys
  </p>
  <p><small>* denotes equal contribution</small></p>

  <h2><a href="https://arxiv.org/pdf/2608.19894">Paper</a></h2>
  <p align="center">
    <img src="https://raw.githubusercontent.com/francois141/upal/main/assets/boat_demo.png" alt="UPAL inference on the boat pair" width="80%" />
    <br />
    <em>Red dots are learned keypoints, green segments are line detections supported by the learned distance field, and colored links are mutual-nearest descriptor matches.</em>
  </p>
</div>

## Upal

Standalone PyTorch inference for **UPAL**, an ECCV 2026–accepted joint point-line detector. The repository contains the network architecture, trained weights, and runnable inference, point-matching, and line-matching examples on the classic `boat1` / `boat2` images. For every input image, UPAL predicts:

- sub-pixel keypoints and their confidence scores;
- 128-dimensional, L2-normalized local descriptors;
- a dense keypoint/junction heatmap;
- a dense line distance field.

The inference network uses an ALIKED-style multi-scale encoder, deformable convolutions in its deeper stages, a sparse deformable descriptor head, a point/junction scoring head, and a line-distance-field decoder. The demo uses the `points-lsd` detector (source bundled as a submodule), seeded from UPAL's learned keypoints, and filters its proposals with the learned distance field.

## Installation

Python 3.10 or newer.

```bash
pip install "upal[lines]"   # network + point-seeded line detector (prebuilt wheels)
pip install upal            # points only, no compiled dependency
```

[`points-lsd`](https://pypi.org/project/points-lsd/) is the point-seeded line segment detector used
for line extraction; wheels exist for Linux, macOS and Windows (CPython 3.10–3.14). Without it, use `model.extract(image, lines=False)` — points,
descriptors and the dense maps work without any compiled dependency.

To develop against this repository instead:

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -e ".[lines]"     # or `-e .` plus the detector built from source, below
```

Or build the detector from the bundled submodule (pinned to the released `points-lsd` version):

```bash
git submodule update --init --recursive
python3 -m pip install ./third_party/points_lsd
```

For CUDA, install the matching PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/) first.

## Quick start

```python
import torch
from upal import UPAL, load_model, mutual_nearest_neighbors, match_lines_from_endpoints
from upal.demo_utils import load_image  # reads a file into an RGB uint8 array and a 1 x 3 x H x W tensor

model = UPAL.from_pretrained("rkreft/upal")          # weights from the Hugging Face Hub
# model = load_model("weights/upal.tar")               # or the checkpoint in this repository

_, image0 = load_image("assets/boat1.png", max_size=640)  # paths relative to this repository
_, image1 = load_image("assets/boat2.png", max_size=640)
feats0, feats1 = model.extract(image0), model.extract(image1)

feats0["keypoints"]            # N x 2 pixel (x, y)
feats0["descriptors"]          # N x 128, L2-normalised
feats0["keypoint_scores"]      # N
feats0["keypoint_dispersity"]  # N, spread of the score peak (lower = sharper)
feats0["keypoint_heatmap"]     # H x W
feats0["line_distance_field"]  # H x W, pixels
feats0["lines"]                # L x 2 x 2 endpoints (needs the `lines` extra)

point_matches = mutual_nearest_neighbors(feats0["descriptors"], feats1["descriptors"])
desc0, desc1 = model.describe_lines(image0, feats0["lines"]), model.describe_lines(image1, feats1["lines"])
line_matches, scores = match_lines_from_endpoints(desc0, desc1)   # NumPy: K x 2 indices, K scores
```

`extract(image, lines=True, max_lines=200, min_line_length=25.0, max_line_distance=2.0)` accepts a
`C x H x W` (or `1 x C x H x W`) tensor in `[0, 1]`, RGB or grayscale. Padding to a multiple of 32 is
handled internally and removed from every output. The keypoint budget is a model argument:
`UPAL.from_pretrained("rkreft/upal", max_num_keypoints=2048)` or `load_model(..., max_num_keypoints=2048)`.
Calling the model directly (`model(batch)`) runs the network only on a `B x C x H x W` batch and returns
batched tensors; see `upal/model.py`.

## Run the demos

From the repository root, run inference on one image and save its feature overlay:

```bash
python demo_inference.py
```

Match learned point descriptors across an image pair:

```bash
python demo_match_points.py
```

Match field-supported line segments across the same pair:

```bash
python demo_match_lines.py
```

The scripts write `outputs/inference.png`, `outputs/point_matches.png`, and `outputs/line_matches.png`, respectively. CPU inference is supported; CUDA is selected automatically when available. The line matcher extracts descriptors at the two endpoints of each detected segment, scores both endpoint orientations, and solves a one-to-one line assignment.

Useful options:

```bash
python demo_match_points.py \
  --image0 path/to/first.jpg \
  --image1 path/to/second.jpg \
  --device cuda \
  --max-size 800 \
  --max-keypoints 1500 \
  --output outputs/my_pair.png
```

## Hugging Face Hub

The release workflow publishes the weights as [`rkreft/upal`](https://huggingface.co/rkreft/upal)
(`model.safetensors` + `config.json`, loaded through `PyTorchModelHubMixin`). To generate
the Hub folder from `weights/upal.tar` and verify it reproduces the checkpoint:

```bash
python scripts/push_to_hub.py            # writes outputs/hub/upal and checks the round trip
python scripts/push_to_hub.py --push --private  # upload while the Hub repository is private
```

Uploads verify the repository visibility before writing. The production release workflow
expects `rkreft/upal` to be public, so make the Hub repository public before pushing the
release tag.

## BibTeX

```
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

## Acknowledgments

Parts of this codebase reuse code from [ALIKED](https://github.com/Shiaoming/ALIKED) and [glue-factory](https://github.com/cvg/glue-factory); we thank their authors for making it available. We also thank the authors of [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork), [ALIKED](https://github.com/Shiaoming/ALIKED), [DaD](https://github.com/daniela-b/DaD), and [DeepLSD](https://github.com/cvg/DeepLSD) for releasing their pre-trained models, which we use as teachers.
