"""GPU smoke test for the runtime image.

Checks, in order:

1. torch sees the GPU, ships kernels for its architecture (sm_120 on the RTX
   5090), and runs matmul, cuDNN conv and ``deform_conv2d`` (TransMVSNet's DCN).
2. pycolmap was built with CUDA and runs GPU SIFT extraction + matching.
3. hloc runs ALIKED + LightGlue + pycolmap incremental mapping end to end,
   which also checks hloc against the installed pycolmap API.

Uses hloc's 10-image Sacre Coeur sample, downloaded once into the cache dir.
Exits non-zero if any check fails.

Usage::

    python scripts/gpu_smoke.py                # inside the GPU container
    python scripts/gpu_smoke.py --device cpu   # logic check on a CPU machine
"""

import argparse
import logging
import os
import sqlite3
import sys
import tempfile
import time
import traceback
import urllib.request
from collections.abc import Callable
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("gpu_smoke")

# Must match the hloc rev pinned in pyproject.toml.
_HLOC_REV = "c13273bd0ecc2917a35910fd843712a1c6243193"
_SAMPLE_URL = (
    f"https://raw.githubusercontent.com/cvg/Hierarchical-Localization/{_HLOC_REV}"
    "/datasets/sacre_coeur/mapping/{name}"
)
_SAMPLE_IMAGES = [
    "02928139_3448003521.jpg",
    "03903474_1471484089.jpg",
    "10265353_3838484249.jpg",
    "17295357_9106075285.jpg",
    "32809961_8274055477.jpg",
    "44120379_8371960244.jpg",
    "51091044_3486849416.jpg",
    "60584745_2207571072.jpg",
    "71295362_4051449754.jpg",
    "93341989_396310999.jpg",
]
_MIN_SIFT_VERIFIED_PAIRS = 5
_MIN_HLOC_REGISTERED = 8


def fetch_sample_images(dest: Path) -> Path:
    """Download the hloc Sacre Coeur mapping images if not already cached.

    Parameters
    ----------
    dest : Path
        Directory to store the images in.

    Returns
    -------
    Path
        ``dest``, containing all sample images.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for name in _SAMPLE_IMAGES:
        path = dest / name
        if not path.exists():
            urllib.request.urlretrieve(_SAMPLE_URL.format(name=name), path)
    return dest


def check_torch(device: str) -> str:
    """Run basic torch ops on ``device`` and verify the GPU architecture.

    Parameters
    ----------
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    str
        Human-readable summary.
    """
    import torch
    from torchvision.ops import deform_conv2d

    summary = f"torch {torch.__version__}"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        if arch not in torch.cuda.get_arch_list():
            raise RuntimeError(
                f"GPU is {arch} but torch was built for {torch.cuda.get_arch_list()}"
            )
        summary += (
            f", CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}, "
            f"{torch.cuda.get_device_name()} ({arch})"
        )

    a = torch.randn(1024, 1024, device=device)
    torch.testing.assert_close(
        (a @ a.T).diagonal(), (a * a).sum(1), rtol=1e-3, atol=1e-2
    )

    conv = torch.nn.Conv2d(3, 8, 3, padding=1).to(device)
    assert conv(torch.randn(1, 3, 64, 64, device=device)).shape == (1, 8, 64, 64)

    x = torch.randn(1, 4, 16, 16, device=device)
    weight = torch.randn(8, 4, 3, 3, device=device)
    offset = torch.zeros(1, 2 * 3 * 3, 16, 16, device=device)
    # Zero offsets must reduce deformable conv to a plain conv.
    torch.testing.assert_close(
        deform_conv2d(x, offset, weight, padding=(1, 1)),
        torch.nn.functional.conv2d(x, weight, padding=1),
        rtol=1e-3,
        atol=1e-3,
    )
    return summary


def check_pycolmap_sift(image_dir: Path, work_dir: Path, device: str) -> str:
    """Run COLMAP SIFT extraction and exhaustive matching.

    Parameters
    ----------
    image_dir : Path
        Directory of input images.
    work_dir : Path
        Scratch directory for the COLMAP database.
    device : str
        ``"cuda"`` or ``"cpu"``.

    Returns
    -------
    str
        Human-readable summary.
    """
    import pycolmap

    if device == "cuda" and not pycolmap.has_cuda:
        raise RuntimeError("pycolmap was built without CUDA")

    colmap_device = pycolmap.Device.cuda if device == "cuda" else pycolmap.Device.cpu
    database_path = work_dir / "sift.db"
    pycolmap.extract_features(
        database_path=database_path, image_path=image_dir, device=colmap_device
    )
    pycolmap.match_exhaustive(database_path=database_path, device=colmap_device)

    # Plain SQL: the COLMAP database schema is more stable than the Python API.
    with sqlite3.connect(database_path) as db:
        num_images = db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        verified = db.execute(
            "SELECT COUNT(*) FROM two_view_geometries WHERE rows >= 15"
        ).fetchone()[0]
    if verified < _MIN_SIFT_VERIFIED_PAIRS:
        raise RuntimeError(
            f"only {verified} verified pairs (< {_MIN_SIFT_VERIFIED_PAIRS})"
        )
    return f"pycolmap {pycolmap.__version__}, {num_images} images, {verified} verified pairs"


def check_hloc_sfm(image_dir: Path, work_dir: Path) -> str:
    """Run the hloc ALIKED + LightGlue + pycolmap mapping pipeline.

    Parameters
    ----------
    image_dir : Path
        Directory of input images.
    work_dir : Path
        Scratch directory for features, matches and the model.

    Returns
    -------
    str
        Human-readable summary.
    """
    from hloc import (
        extract_features,
        match_features,
        pairs_from_exhaustive,
        reconstruction,
    )

    feature_conf = extract_features.confs["aliked-n16"]
    matcher_conf = match_features.confs["aliked+lightglue"]
    names = sorted(p.name for p in image_dir.iterdir())
    features = work_dir / "features.h5"
    matches = work_dir / "matches.h5"
    pairs = work_dir / "pairs.txt"

    extract_features.main(
        feature_conf, image_dir, image_list=names, feature_path=features
    )
    pairs_from_exhaustive.main(pairs, image_list=names)
    match_features.main(matcher_conf, pairs, features=features, matches=matches)
    model = reconstruction.main(
        work_dir / "sfm", image_dir, pairs, features, matches, image_list=names
    )
    if model is None:
        raise RuntimeError("hloc reconstruction returned no model")

    registered = model.num_reg_images()
    if registered < _MIN_HLOC_REGISTERED:
        raise RuntimeError(f"only {registered}/{len(names)} images registered")
    return (
        f"{registered}/{len(names)} registered, {model.num_points3D()} points, "
        f"reproj {model.compute_mean_reprojection_error():.2f} px"
    )


def _run(
    name: str, fn: Callable[[], str], results: list[tuple[str, bool, str]]
) -> None:
    start = time.perf_counter()
    try:
        detail = fn()
        ok = True
    except Exception as exc:  # noqa: BLE001 - report every failure, keep going
        traceback.print_exc()
        detail = f"{type(exc).__name__}: {exc}"
        ok = False
    results.append((name, ok, f"{detail} [{time.perf_counter() - start:.1f}s]"))


def main() -> None:
    """Run all smoke checks and exit non-zero if any fails."""
    parser = argparse.ArgumentParser(
        description="GPU smoke test for the runtime image."
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Image directory to use instead of the downloaded Sacre Coeur sample.",
    )
    args = parser.parse_args()

    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    image_dir = args.images or fetch_sample_images(cache / "smoke" / "sacre_coeur")

    results: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="gpu_smoke_") as tmp:
        work_dir = Path(tmp)
        _run("torch", lambda: check_torch(args.device), results)
        _run(
            "pycolmap SIFT",
            lambda: check_pycolmap_sift(image_dir, work_dir, args.device),
            results,
        )
        _run("hloc SfM", lambda: check_hloc_sfm(image_dir, work_dir), results)

    print("\n=== GPU smoke results ===")
    for name, ok, detail in results:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)


if __name__ == "__main__":
    main()
