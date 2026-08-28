# UPAL on Hugging Face — integration plan

Status: in progress (2026-08-27) · Context: [issue #1](https://github.com/francois141/upal/issues/1)

## Status update — 2026-08-27

**Step 1 (`points-lsd` wheels): DONE — 2026-08-28.** PR [#2](https://github.com/francois141/points_lsd/pull/2)
merged (`38ecf88`), tag `v0.1.0`, published to [PyPI](https://pypi.org/project/points-lsd/0.1.0/)
(25 wheels + sdist, CPython 3.10–3.14, Linux x86_64/aarch64, macOS arm64/x86_64, Windows)
via trusted publishing; TestPyPI dry run done before. Verified: clean-venv install from PyPI,
smoke tests, and 12 LSD scenarios on UPAL outputs bit-identical to a source build. Tag
`pre-workflow-and-packaging` marks the state before this work. `upal` submodule pinned to `38ecf88`.

History of the step:

- Branch `pypi-wheels`: module renamed `pytlsd` → `points_lsd`, metadata in `pyproject.toml`
  (SPDX `MIT`, AGPL text of the LSD core shipped under `LICENSES/`), pybind11 `v3.1.0`,
  OpenCV dropped from the module, OpenMP opt-in, input validation in `lsd_from_points`
  (missing gradients / out-of-image seeds / bad shapes / `grad_nfa` now raise instead of
  crashing), all array inputs forced C-contiguous, numpy-only smoke test, full suite renamed
  to `tests/test_lsd.py`, new `wheels.yml` (cibuildwheel v4, sdist, tests on 3 OSes,
  tag-vs-version guard, TestPyPI via `workflow_dispatch`, PyPI on `v*` tags, trusted
  publishing). Reviewed by four independent review passes; all findings applied.
- Verified: output bit-identical to the pre-rename build (seeded and full-image LSD, and
  UPAL's demo line detections on the boat pair); sdist round-trip; Linux aarch64 wheel
  built and tested in Docker.
- First CI run: Linux x86_64/aarch64 and macOS arm64/x86_64 wheels green, full test suite
  green on Ubuntu/macOS. **Windows failed** with heap corruption (`0xC0000374`): the
  point-seeded LSD functions `free()` list buffers that are only allocated by `ll_angle`,
  which that path never calls — undefined behaviour that gcc/clang tolerated and MSVC did
  not. Fixed in `src/lsd.cpp` (pointers initialised to `nullptr`); awaiting the CI rerun.
- All of the above completed; remaining nicety: required reviewers on the `pypi` GitHub
  environment (needs repo admin).

**Steps 2–3 (`upal` package, Hub mixin): implemented locally; PyPI and Hub uploads pending.**

- `pyproject.toml`: dist renamed to `upal` (0.1.0), Apache-2.0 SPDX, `huggingface_hub` +
  `safetensors` dependencies, `[lines]` extra → `points-lsd>=0.1.0`, project URLs.
- `upal/model.py`: `UPAL` now mixes in `PyTorchModelHubMixin` (`repo_url`, `paper_url`,
  `library_name="upal"`, `license`, `pipeline_tag="keypoint-detection"`, tags) — nothing
  else changed in the network. New convenience API: `UPAL.extract(image, lines=True, ...)`
  (single image → un-batched outputs + `lines`) and `UPAL.describe_lines(image, lines)`
  (`L x 2 x D` endpoint descriptors for `match_lines_from_endpoints`). `load_model` kept.
- `upal/__init__.py` exports `UPAL`, `load_model`, `detect_lines`,
  `mutual_nearest_neighbors`, `match_lines_from_endpoints`, `__version__`.
- `scripts/push_to_hub.py`: converts `weights/upal.tar` → `outputs/hub/upal/`
  (`model.safetensors` 3.2 MB / 0.79M params, `config.json`, model card, demo image) and
  asserts `UPAL.from_pretrained(folder)` reproduces `load_model` exactly; `--push --repo-id`
  uploads via `upload_folder` (the mixin's `push_to_hub` would regenerate and overwrite
  the card). `hub/README.md` is the model card (YAML: `license`, `library_name: upal`,
  `pipeline_tag`, tags incl. `arxiv:2608.19894`). `from_pretrained` loads strictly
  (overridden; the mixin default is non-strict). Requires `huggingface_hub>=0.30`.
- README rewritten: `pip install "upal[lines]"`, quick start with `from_pretrained` /
  `extract` / matching, Hub section.
- Verified: sdist + wheel build; wheel installed in a clean venv → `from_pretrained` on the
  converted folder, `extract` and the demos run; after the `points-lsd` release,
  `pip install -e ".[lines]"` resolves from PyPI and the demos reproduce (1024/88, 100, 88/78/50).
- Release safety added: four focused tests cover the public exports/version, point-only
  `extract`, an exact local Hub round trip, and checkpoint-to-line end-to-end extraction plus
  endpoint descriptors. `.github/workflows/package.yml` runs them on Linux (Python 3.10 and
  3.14), macOS and Windows, builds/checks the wheel and sdist, supports an explicit TestPyPI
  dispatch, publishes to PyPI only for matching `v*` tags, and uploads the Hub model only after
  the tagged PyPI publication succeeds. No publish workflow has been run.
- Hub namespace resolved to the authenticated personal account, `rkreft/upal`; the GitHub
  discussion requested a dedicated model repository but did not agree on the `ETH-CVG`
  namespace. Configure the `testpypi`, `pypi`, and `huggingface` GitHub
  environments, add the PyPI trusted-publisher records and an `HF_TOKEN` with write access,
  then create the release tag when publication is intended. Ask HF to link the model to the
  paper page after upload.



---

## 1. Summary (for sharing)

**Goal.** Make UPAL — *both* the learned point detector/descriptor *and* the point-seeded
line detector — installable and runnable from the Hugging Face Hub with a plain
`pip install`, and link it to the paper page
(<https://huggingface.co/papers/2608.19894>).

**Decision.** Go with the `PyTorchModelHubMixin` route (as Niels suggested) instead of a
full `transformers` integration, and pair it with **prebuilt wheels for the modified
LSD** (`points_lsd`). Note that the mixin comes from `huggingface_hub`, not from
`transformers`: it adds `from_pretrained` / `push_to_hub` to *our own* `UPAL` class and
needs no model code inside `transformers`. Reasoning:

- The network (keypoints, descriptors, junction heatmap, line distance field) is pure
  PyTorch and maps 1:1 onto the mixin; no code restructuring is needed.
- Line detection depends on exactly one non-Python component, the C++
  `points_lsd.lsd_from_points` binding. Everything else in the line pipeline (gradients,
  seeding, distance-field filtering, endpoint matching) is already Torch/NumPy.
  The binding has **no external native dependencies** (no OpenCV at link time), so
  wheels for all platforms are straightforward to build with `cibuildwheel`.
- A `transformers` integration could never include the C++ detector, so lines would be
  a separate package in that route anyway. It does not solve the line problem; it only
  moves the split. It remains an optional later step for the point branch.

**Target user experience.**

```bash
pip install "upal[lines]"          # points-only: pip install upal
```
```python
from upal import UPAL
model = UPAL.from_pretrained("<namespace>/upal")
out = model.extract(image)         # keypoints, scores, descriptors, lines
```

**How the pieces fit.** The Hub stores *weights and config*, not code. `from_pretrained`
downloads `model.safetensors` + `config.json` and then calls `UPAL(**config)`, so the
`UPAL` class must already be importable. The pip package supplies that class and all
post-processing; the Hub repo supplies the trained weights. Neither works alone.

| Piece                                           | What it provides                                                     |
|-------------------------------------------------|----------------------------------------------------------------------|
| `upal` (pip)                                    | network code, `extract()`, point/line matching, line post-processing |
| `points-lsd` (pip, via `upal[lines]`)           | the C++ line detector binding                                        |
| Hub repo `<namespace>/upal`                     | `model.safetensors`, `config.json`, model card, paper link           |
| `PyTorchModelHubMixin` (from `huggingface_hub`) | the `from_pretrained` / `push_to_hub` glue between the two           |

A `transformers` integration is a different thing: the network would be *re-implemented
inside `transformers`* so users never install `upal` for the point branch. That is why it
cannot cover lines (the C++ detector cannot live in `transformers`) and why it is only an
optional later step here.

**Deliverables.**

| # | Deliverable                                                         | Owner                          | Effort   |
|---|---------------------------------------------------------------------|--------------------------------|----------|
| 1 | `points-lsd` wheels on PyPI (Linux/macOS/Windows, py3.10–3.14)      | UPAL authors                   | 1–2 days |
| 2 | `upal` on PyPI with `[lines]` extra and `extract()` convenience API | UPAL authors                   | 1 day    |
| 3 | `PyTorchModelHubMixin` on `UPAL`, weights pushed as safetensors     | UPAL authors                   | 0.5 day  |
| 4 | Hub model repo + model card, linked to the paper page               | UPAL authors + HF              | 0.5 day  |
| 5 | Demo Space (ZeroGPU) showing points + lines on an image pair        | UPAL authors                   | 1–2 days |
| 6 | *(optional, later)* `transformers` integration of the point branch  | UPAL authors + HF (@sbucaille) | weeks    |

**What we would appreciate from HF.**

- Confirmation of the tag set for a joint point-line model (`pipeline_tag:
  keypoint-detection` plus `line-detection` / `image-matching` tags), and whether moving
  `rkreft/upal` to an organization namespace later would improve discoverability.
- Linking the model repo and Space to the paper page once uploaded.
- A ZeroGPU grant for the demo Space if the two free personal ones are not enough.
- A pointer on whether a `transformers` port of the point branch is wanted (step 6).

---

## 2. Detailed steps

### Step 0 — Decisions to make first

| Decision | Recommendation |
|----------|----------------|
| Hub namespace | `rkreft/upal` (personal namespace; the issue did not agree on `ETH-CVG`) |
| One repo per checkpoint | Yes (HF recommendation). Currently a single checkpoint: `weights/upal.tar` → `<namespace>/upal` |
| PyPI name of the LSD binding | **`points-lsd`** with import name `points_lsd`. `pytlsd` already exists on PyPI (Iago Suárez's upstream), so the fork must not reuse that name; two distributions providing the same `pytlsd` module would clash. Renaming the module touches the import, call, docstring and error message in `upal/postprocess.py`. |
| PyPI name of the model package | `upal` (currently `upal-local-features` in `pyproject.toml`) |
| Licenses | UPAL: Apache-2.0 (repo `LICENSE`). `points_lsd`: the binding is MIT (inherited from pytlsd), **but `src/lsd.cpp` is AGPL-3.0-or-later** and is statically linked into the wheel — decide how to label the PyPI package (and whether that matters for downstream users) before publishing. |

### Step 1 — Build and publish `points_lsd` wheels

This is the step that unlocks "full point + line" everywhere, including HF Spaces.

1. **Confirm the binding is dependency-free.** Verified: `src/lsd.cpp` and `src/PYAPI.cpp`
   include no OpenCV headers; `CMakeLists.txt` only links OpenCV if `find_package(OpenCV QUIET)`
   succeeds, and only the C++ test target needs it. Done: the `points_lsd` target no longer
   looks for OpenCV; the C++ test is behind `POINTS_LSD_BUILD_TESTS` (OFF). OpenMP (used only
   by the batched full-image entry point) is behind `POINTS_LSD_USE_OPENMP` (OFF) so wheels
   carry no runtime OpenMP dependency.
2. **Modernise the packaging** in `third_party/points_lsd`:
   - All metadata moves to `pyproject.toml` (`name = "points-lsd"`, `version`, SPDX
     `license = "MIT"`, `requires-python = ">=3.10"`); `setup.py` keeps only the CMake glue
     and builds the `points_lsd` extension.
   - `pyproject.toml` `[tool.cibuildwheel]`: `build = "cp310-* cp311-* cp312-* cp313-* cp314-*"`,
     `skip = "*-musllinux_*"`,
     `archs = ["auto64"]` on Linux, `["AMD64"]` on Windows and `["arm64", "x86_64"]` on macOS,
     default `manylinux_2_28` images (`manylinux2014` is EOL and its containers cannot
     install a NumPy wheel for the test stage), and a numpy-only smoke test
     (`tests/test_smoke.py`) as `test-command`.
   - Remove the old `package/build-wheels-*.sh` scripts or mark them deprecated.
3. **CI**: `.github/workflows/wheels.yml` in the `points_lsd` repo: `pypa/cibuildwheel@v4`
   across `ubuntu-latest`, `ubuntu-24.04-arm`, `macos-14`, `windows-latest`; an sdist job;
   the full test suite on 3 OSes; a tag-vs-version guard; TestPyPI upload via
   `workflow_dispatch` and PyPI upload on `v*` tags, both through trusted publishing
   (environments `testpypi` / `pypi`).
4. **Sanity check** locally: `pip install points-lsd` in a fresh venv, run
   `demo_inference.py` and compare line counts against the source build.
5. **Update UPAL** (`upal/postprocess.py`): `import points_lsd` and the error message in
   `detect_lines`, pointing to `pip install points-lsd` (switch to `pip install "upal[lines]"`
   once Step 2 adds that extra).
6. **API hardening done alongside the rename:** `lsd_from_points` now raises `ValueError`
   when `gradnorm`/`gradangle` are omitted (the C++ dereferenced them unconditionally and
   segfaulted) or when a seed lies outside the image; malformed seed arrays raise
   `TypeError`; `grad_nfa=True` (unsupported on the seeded path, previously `exit(1)`) raises
   `ValueError`; all array inputs are forced C-contiguous so strided views are safe. UPAL
   always passes gradients and pre-filters seeds, so its behaviour is unchanged (verified
   bit-identical against the pre-rename build).
7. **Known quirk (not changed):** column 4 of the returned `N x 5` array is LSD's angle
   precision `p` (constant `0.125`), not `-log10(NFA)` — the upstream binding drops `width`
   and NFA. `upal/postprocess.py` sorts by this column, which is therefore a no-op; exposing
   the NFA is a separate, additive decision.

### Step 2 — Make `upal` pip-installable with a `[lines]` extra

1. `pyproject.toml`: `name = "upal"`, add `huggingface_hub>=0.25` to dependencies, add
   `[project.optional-dependencies] lines = ["points-lsd>=<version>"]`, add URLs
   (repo, paper, Hub model).
2. Add a high-level API so users need no knowledge of the internals, e.g. in
   `upal/model.py` (or a thin `upal/api.py`):
   ```python
   def extract(self, image, *, lines: bool = True, max_lines: int = 200, ...) -> dict
   ```
   which runs `forward`, then `detect_lines` when `lines=True` (raising a clear
   `ImportError` with the install hint if `points_lsd` is missing), and returns
   `keypoints`, `keypoint_scores`, `descriptors`, `lines`, plus the dense maps.
3. Optionally expose `match_points(desc0, desc1)` and `match_lines(model, img0, lines0,
   img1, lines1)` wrappers over `mutual_nearest_neighbors` and
   `match_lines_from_endpoints` (the latter currently requires the user to call
   `describe_keypoints` on endpoints themselves, as in `demo_match_lines.py`).
4. Publish to PyPI (tag-triggered workflow in this repo).

### Step 3 — Add `PyTorchModelHubMixin` and push the weights

The mixin lives in `huggingface_hub` (already added as a dependency in Step 2). It only
adds `from_pretrained` / `push_to_hub`; the architecture code stays in the `upal` package.
Strictly, this step works without Step 2 (`pip install git+https://github.com/francois141/upal`
is enough to make `UPAL` importable); PyPI publishing is for convenience and the `[lines]` extra.

1. In `upal/model.py`:
   ```python
   from huggingface_hub import PyTorchModelHubMixin

   class UPAL(
       nn.Module,
       PyTorchModelHubMixin,
       repo_url="https://github.com/francois141/upal",
       paper_url="https://arxiv.org/abs/2608.19894",
       pipeline_tag="keypoint-detection",
       license="apache-2.0",
       tags=["line-detection", "image-matching", "local-features"],
   ):
   ```
   No other change is needed: the constructor arguments (`max_num_keypoints`,
   `nms_radius`, `line_neighborhood`) are plain numbers and are serialised to
   `config.json` automatically. Keep `load_model()` for local `.tar` checkpoints.
2. Conversion script `scripts/push_to_hub.py`:
   ```python
   model = load_model("weights/upal.tar")          # strips "extractor." prefix, filters to inference subset
   model.push_to_hub("<namespace>/upal", commit_message="Initial UPAL release")
   ```
   This uploads a **clean `model.safetensors`** + `config.json`. Do not upload
   `upal.tar` as the primary artifact — it is a training checkpoint that only loads
   through the filtering logic in `load_model`.
3. Verify the round trip: `UPAL.from_pretrained("<namespace>/upal")` and
   `load_model("weights/upal.tar")` must give identical outputs on `assets/boat1.png`
   (`torch.testing.assert_close` on keypoints, descriptors, heatmap, distance field).
4. Note on `max_num_keypoints`: it is baked into `config.json`; document that users can
   override it with `UPAL.from_pretrained(..., max_num_keypoints=2048)`.

### Step 4 — Hub model repo and model card

Files in `<namespace>/upal`: `model.safetensors`, `config.json`, `README.md`
(model card), and `assets/boat_demo.png` for the card.

Model card YAML front matter:
```yaml
license: apache-2.0
pipeline_tag: keypoint-detection
tags: [line-detection, image-matching, local-features, pytorch_model_hub_mixin, model_hub_mixin, "arxiv:2608.19894"]
library_name: upal
```
Body: one-paragraph description (points + lines, ALIKED-style encoder, SDDH descriptors,
line distance field), the install/usage snippet from the summary, output shapes (from the
GitHub README "Python API" section), the optional-lines explanation, the boat demo image,
BibTeX, and acknowledgements (ALIKED, glue-factory, pytlsd, teacher models).
Then claim the paper on the paper page and ask HF to link the model repo.

### Step 5 — Update the GitHub README

Replace the build-from-source instructions with `pip install "upal[lines]"`, keep the
source build as a fallback, add the `from_pretrained` snippet, and link the Hub repo and
the Space.

### Step 6 — Demo Space

Two options; the first gets the most visibility, the second is faster to ship.

- **A. PR to [image-matching-webui](https://huggingface.co/spaces/Realcat/image-matching-webui)**
  adding UPAL as a point matcher (MNN on descriptors) and a line matcher (endpoint
  assignment). Requires `points-lsd` in the Space's requirements — possible only after
  Step 1.
- **B. Standalone Gradio Space** on ZeroGPU: upload an image pair (default: `boat1`/`boat2`),
  sliders for `max_keypoints`, `max_size`, `min_length`, `max_mean_distance`; outputs
  the three visualisations produced by the existing demo scripts. `requirements.txt`:
  `upal[lines]`, `gradio`, `spaces`. Reuse `upal/demo_utils.py` for drawing.

### Step 7 — Optional: `transformers` integration of the point branch

Only if there is appetite on both sides. Unlike Steps 2–3, this means re-implementing
the network *inside* `transformers` (users would not need the `upal` package for points).
Template: SuperPoint
(`SuperPointForKeypointDetection` + `SuperPointImageProcessor` +
`post_process_keypoint_detection`); UPAL fits the same output structure with two extra
dense tensors (`keypoint_heatmap`, `line_distance_field`).

Required: `configuration_upal.py`, `modular_upal.py` (encoder, score head, SDDH,
distance-field head, NMS + sub-pixel refinement), `image_processing_upal(_fast).py`,
conversion script, tests, docs, auto-class registration.

Known obstacles:
- `torchvision.ops.deform_conv2d` in `DeformableConv2d` (`upal/model.py:75`) —
  torchvision is a soft dependency in `transformers`; expect to add a
  `requires_backends` guard or a pure-PyTorch fallback validated against torchvision.
- Line detection stays outside `transformers` (C++). The `upal` package from Step 2
  would consume the `transformers` outputs (`line_distance_field`, keypoints) to produce
  lines, so the two routes stay compatible.

---

## 3. Release runbook (in order)

**A. `points-lsd`** — DONE (v0.1.0 on PyPI). Kept for the next release:

```bash
cd third_party/points_lsd
git add src/lsd.cpp
git commit -m "fix: initialise unused LSD list buffers on the point-seeded path (MSVC heap corruption)"
git push                                   # PR #2 CI reruns; Windows legs should go green
gh pr ready 2                              # undraft, then merge on GitHub
```

Then, once on `main`: create PyPI + TestPyPI accounts (2FA), add a *pending trusted
publisher* on each (project `points-lsd`, owner `francois141`, repo `points_lsd`, workflow
`wheels.yml`, environment `testpypi` / `pypi`); optionally have François add required
reviewers to the `pypi` environment. Dry run: *Actions → Wheels → Run workflow* with
`publish_testpypi = true`, then `pip install -i https://test.pypi.org/simple/ points-lsd`
in a clean venv. Release:

```bash
git checkout main && git pull
git tag v0.1.0 && git push origin v0.1.0    # publish_pypi job uploads to PyPI
```

**B. `upal`** (this repo)

```bash
git add README.md pyproject.toml upal/ scripts/ hub/ HF_INTEGRATION_PLAN.md third_party/points_lsd
git commit -m "feat: pip package with lines extra, Hugging Face Hub mixin, extract() API"
git push
```

(The `third_party/points_lsd` entry records the submodule commit; commit it only after the
submodule fix above is committed so the pointer is not dangling.)

PyPI (needs `points-lsd` live first, otherwise `pip install "upal[lines]"` cannot resolve):
either add a `wheels.yml`-style trusted-publishing workflow here or upload manually:

```bash
uvx --from build pyproject-build -o dist .
uvx twine upload dist/*                    # PyPI API token or trusted publisher
```

**C. Hugging Face Hub**

```bash
hf auth login                              # once
python scripts/push_to_hub.py --push --repo-id rkreft/upal    # converts, verifies, uploads
```

Then on the Hub: check the model page renders the card and the `from_pretrained` snippet,
link the model to the paper page (https://huggingface.co/papers/2608.19894 → "add model"),
claim the paper, and reply on issue #1 with the links.

## 4. Checklist

- [ ] Step 0: namespace, package names agreed
- [x] Step 1: `points-lsd` 0.1.0 on PyPI (2026-08-28)
- [~] Step 2: `upal` package implemented; PyPI upload to do
- [~] Step 3: mixin added, round-trip verified; Hub push to do
- [~] Step 4: model card written (`hub/README.md`); paper link to do after upload
- [x] Step 5: GitHub README updated
- [ ] Step 6: Space live
- [ ] Step 7: decide on `transformers` port
