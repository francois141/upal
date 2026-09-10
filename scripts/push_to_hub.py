#!/usr/bin/env python3
"""Convert the training checkpoint to a Hugging Face Hub model repo and (optionally) push it.

Default run: writes ``model.safetensors`` + ``config.json`` + model card + demo image to
``--output`` and verifies that ``UPAL.from_pretrained(output)`` reproduces ``load_model``
on the demo image. Add ``--push`` to upload the folder to ``--repo-id`` (needs
``hf auth login`` or ``HF_TOKEN``). Uploads require the existing repository to
match the requested visibility: public by default, or private with ``--private``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from huggingface_hub import HfApi

    from upal import UPAL

ROOT = Path(__file__).resolve().parents[1]


def verify_round_trip(reference: "UPAL", output: Path, image_path: Path) -> None:
    import torch

    from upal import UPAL
    from upal.demo_utils import load_image

    restored = UPAL.from_pretrained(output).eval()
    _, tensor = load_image(image_path, 640)
    with torch.inference_mode():
        expected = reference(tensor)
        actual = restored(tensor)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0, msg=f"{key} differs")
    print(f"round trip OK: {len(expected)} outputs identical on {image_path.name}")


def prepare_output(output: Path) -> Path:
    """Resolve and clear a generated-output directory without risking broad deletion."""
    output = output.expanduser()
    if output.is_symlink():
        raise ValueError(f"refusing to replace symlink passed as --output: {output}")
    output = output.resolve()
    protected = {Path("/").resolve(), Path.home().resolve(), ROOT.resolve()}
    if output in protected:
        raise ValueError(f"refusing to use protected path as --output: {output}")
    if output.exists():
        if not output.is_dir():
            raise ValueError(f"--output exists and is not a directory: {output}")
        shutil.rmtree(output)
    return output


def validate_prepared_output(output: Path) -> Path:
    """Validate a previously generated Hub folder before uploading it."""
    output = output.expanduser().resolve()
    required = {"README.md", "config.json", "model.safetensors", "assets/boat_demo.png"}
    missing = sorted(path for path in required if not (output / path).is_file())
    if missing:
        raise ValueError(f"prepared Hub folder is missing: {missing}")
    return output


def verify_repo_visibility(api: "HfApi", repo_id: str, *, private: bool) -> None:
    """Refuse to upload when an existing Hub repository has unexpected visibility."""
    info = api.repo_info(repo_id, repo_type="model")
    actual_private = bool(info.private)
    if actual_private != private:
        actual = "private" if actual_private else "public"
        expected = "private" if private else "public"
        flag = " with --private" if private else " without --private"
        raise RuntimeError(
            f"Hub repository {repo_id} is {actual}, but this upload requires it to be "
            f"{expected}{flag}; change the repository visibility before uploading"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/upal.tar")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/hub/upal")
    parser.add_argument("--repo-id", default="rkreft/upal")
    parser.add_argument("--push", action="store_true", help="upload --output to --repo-id")
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="upload an existing prepared --output folder without regenerating it",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="create or require a private Hub repo (uploads require public visibility by default)",
    )
    parser.add_argument("--commit-message", default="Add UPAL inference weights")
    args = parser.parse_args()

    if args.upload_only:
        if not args.push:
            parser.error("--upload-only requires --push")
        args.output = validate_prepared_output(args.output)
    else:
        from upal import load_model

        args.output = prepare_output(args.output)
        model = load_model(args.weights)
        model.save_pretrained(args.output)  # model.safetensors + config.json (+ generated README)
        shutil.copy(ROOT / "hub/README.md", args.output / "README.md")  # our model card wins
        (args.output / "assets").mkdir(exist_ok=True)
        shutil.copy(ROOT / "assets/boat_demo.png", args.output / "assets/boat_demo.png")
        print("wrote:", *sorted(str(p.relative_to(args.output)) for p in args.output.rglob("*") if p.is_file()))

        verify_round_trip(model, args.output, ROOT / "assets/boat1.png")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        verify_repo_visibility(api, args.repo_id, private=args.private)
        url = api.upload_folder(
            folder_path=args.output,
            repo_id=args.repo_id,
            repo_type="model",
            commit_message=args.commit_message,
            allow_patterns=["README.md", "config.json", "model.safetensors", "assets/*"],
        )
        print("pushed:", url)
    else:
        print(f"dry run; re-run with --push to upload to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
