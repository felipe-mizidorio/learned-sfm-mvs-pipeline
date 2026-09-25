"""TransMVSNet depth inference over an undistorted MVS workspace.

Input preparation mirrors upstream's BlendedMVS/Tanks-and-Temples evaluation
loader (``datasets/tnt_eval.py``): RGB in [0, 1] without normalization, one
image size for the whole scene (both sides multiples of 32), stage projection
matrices at 1/4, 1/2 and full resolution, 192 evenly spaced depth hypotheses.
The saved confidence is the product of the three stages' confidences, as in
upstream's ``test.py``.

One ``<image_id>.npz`` per reference view holds everything fusion needs, so
fusion can be re-run with other thresholds without inference.
"""

import hashlib
import logging
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from learned_sfm_mvs._vendor.transmvsnet.models import TransMVSNet
from learned_sfm_mvs.mvs.transmvsnet.views import View

logger = logging.getLogger(__name__)

_SIZE_BASE = 32


def target_size(
    width: int, height: int, max_long_side: int, max_short_side: int
) -> tuple[int, int]:
    """Network input size: fit inside the maximum, both sides multiples of 32.

    Upstream's ``scale_mvs_input`` bounds width and height, which assumes
    landscape images; here the bounds apply to the long and short sides, so
    portrait video is not shrunk to fit a landscape box. Never upscales.

    Parameters
    ----------
    width, height : int
        Original image size.
    max_long_side, max_short_side : int
        Upper bounds for the longer and the shorter side.

    Returns
    -------
    tuple[int, int]
        ``(width, height)`` for the network.
    """
    scale = min(
        1.0,
        max_long_side / max(width, height),
        max_short_side / min(width, height),
    )
    new_w = int(scale * width // _SIZE_BASE * _SIZE_BASE)
    new_h = int(scale * height // _SIZE_BASE * _SIZE_BASE)
    if new_w == 0 or new_h == 0:
        raise ValueError(f"Image {width}x{height} too small for TransMVSNet")
    return new_w, new_h


def scale_intrinsics(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """Intrinsics after resizing, for pixel centres at integer coordinates.

    Parameters
    ----------
    K : np.ndarray
        ``(3, 3)`` intrinsics (see ``views.View.K``).
    sx, sy : float
        New over old width and height.

    Returns
    -------
    np.ndarray
        Scaled ``(3, 3)`` intrinsics.
    """
    scaled = K.astype(np.float64).copy()
    scaled[0, 0] *= sx
    scaled[1, 1] *= sy
    scaled[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
    scaled[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return scaled


def depth_hypotheses(depth_min: float, depth_max: float, num_depth: int) -> np.ndarray:
    """Evenly spaced depth hypotheses from ``depth_min``, as upstream.

    Parameters
    ----------
    depth_min, depth_max : float
        Search range.
    num_depth : int
        Number of hypotheses.

    Returns
    -------
    np.ndarray
        ``(num_depth,)`` float32.
    """
    interval = (depth_max - depth_min) / num_depth
    return (depth_min + interval * np.arange(num_depth)).astype(np.float32)


def load_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    """RGB image resized to ``size``, float32 in [0, 1].

    Parameters
    ----------
    path : Path
        Image file.
    size : tuple[int, int]
        ``(width, height)``.

    Returns
    -------
    np.ndarray
        ``(height, width, 3)`` array.

    Raises
    ------
    FileNotFoundError
        If the image cannot be read.
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    if (bgr.shape[1], bgr.shape[0]) != size:
        # INTER_LINEAR: upstream's cv2.resize default.
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def verify_checkpoint(path: Path, expected_sha256: str | None) -> str:
    """SHA-256 of the checkpoint, checked against the expected value.

    Parameters
    ----------
    path : Path
        Checkpoint file.
    expected_sha256 : str or None
        Expected digest; None skips the check.

    Returns
    -------
    str
        The file's SHA-256, for the manifest.

    Raises
    ------
    FileNotFoundError
        If the checkpoint does not exist.
    ValueError
        If the digest does not match.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"TransMVSNet checkpoint not found: {path}. Download model_bld.ckpt "
            "(see README) into models/transmvsnet/."
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError(
            f"Checkpoint {path} has SHA-256 {digest}, expected {expected_sha256}."
        )
    return digest


def load_model(checkpoint: Path, model_cfg: dict, device: torch.device) -> TransMVSNet:
    """Build TransMVSNet and load weights (safe unpickling only).

    Parameters
    ----------
    checkpoint : Path
        Checkpoint with a ``model`` state dict.
    model_cfg : dict
        ``model`` section of ``configs/transmvsnet.yaml``.
    device : torch.device
        Inference device.

    Returns
    -------
    TransMVSNet
        Model in eval mode on ``device``.
    """
    model = TransMVSNet(
        refine=False,
        ndepths=list(model_cfg["ndepths"]),
        depth_interals_ratio=list(model_cfg["depth_interval_ratios"]),
        share_cr=False,
        cr_base_chs=[8] * len(model_cfg["ndepths"]),
        grad_method="detach",
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    return model.eval().to(device)


def build_sample(
    views: list[View],
    images: list[np.ndarray],
    size: tuple[int, int],
    num_depth: int,
) -> dict:
    """Network inputs for one reference view (first) and its sources.

    Parameters
    ----------
    views : list[View]
        Reference view followed by source views.
    images : list[np.ndarray]
        Matching images from ``load_image``.
    size : tuple[int, int]
        Network ``(width, height)``.
    num_depth : int
        Number of depth hypotheses.

    Returns
    -------
    dict
        ``imgs`` ``(1, V, 3, H, W)``, ``proj_matrices`` per stage
        ``(1, V, 2, 4, 4)``, ``depth_values`` ``(1, D)``, and ``K`` the
        reference intrinsics at full network resolution.
    """
    width, height = size
    proj = np.zeros((len(views), 2, 4, 4), dtype=np.float32)
    ref_K = None
    for i, view in enumerate(views):
        K = scale_intrinsics(view.K, width / view.width, height / view.height)
        if i == 0:
            ref_K = K
        proj[i, 0] = view.extrinsic
        # Upstream scales the first two rows by 1/4 for stage 1 and multiplies
        # them back up, so stage 3 equals the full-resolution K.
        quarter = K.copy()
        quarter[:2] /= 4.0
        proj[i, 1, :3, :3] = quarter
        proj[i, 1, 3, 3] = 1.0
    stage2, stage3 = proj.copy(), proj.copy()
    stage2[:, 1, :2, :] *= 2
    stage3[:, 1, :2, :] *= 4
    imgs = np.stack(images).transpose(0, 3, 1, 2)[None]
    ref = views[0]
    return {
        "imgs": torch.from_numpy(np.ascontiguousarray(imgs)),
        "proj_matrices": {
            "stage1": torch.from_numpy(proj[None]),
            "stage2": torch.from_numpy(stage2[None]),
            "stage3": torch.from_numpy(stage3[None]),
        },
        "depth_values": torch.from_numpy(
            depth_hypotheses(ref.depth_min, ref.depth_max, num_depth)[None]
        ),
        "K": ref_K,
    }


def combined_confidence(outputs: dict) -> torch.Tensor:
    """Product of the three stages' confidences at full resolution.

    Parameters
    ----------
    outputs : dict
        TransMVSNet outputs.

    Returns
    -------
    torch.Tensor
        ``(H, W)`` confidence.
    """
    final = outputs["photometric_confidence"][0]
    size = final.shape[-2:]
    conf = final
    for stage in ("stage1", "stage2"):
        stage_conf = outputs[stage]["photometric_confidence"]
        conf = (
            conf
            * F.interpolate(
                stage_conf[:, None], size=size, mode="bilinear", align_corners=False
            )[0, 0]
        )
    return conf


def run_inference(
    workspace: Path,
    views: list[View],
    cfg: dict,
    out_dir: Path,
    device: torch.device,
) -> dict:
    """Estimate and save a depth map for every view with source views.

    Parameters
    ----------
    workspace : Path
        Undistorted MVS workspace (images in ``images/``).
    views : list[View]
        Output of ``views.build_views``.
    cfg : dict
        ``transmvsnet`` section of ``configs/transmvsnet.yaml``.
    out_dir : Path
        Where ``<image_id>.npz`` files are written.
    device : torch.device
        Inference device.

    Returns
    -------
    dict
        Checkpoint digest, network size, view counts and timing.
    """
    checkpoint = Path(cfg["checkpoint"])
    digest = verify_checkpoint(checkpoint, cfg.get("checkpoint_sha256"))
    model = load_model(checkpoint, cfg["model"], device)
    infer_cfg = cfg["inference"]
    by_id = {v.image_id: v for v in views}
    ref = views[0]
    size = target_size(
        ref.width,
        ref.height,
        int(infer_cfg["max_long_side"]),
        int(infer_cfg["max_short_side"]),
    )

    @lru_cache(maxsize=int(infer_cfg["num_view"]) * 4)
    def image(name: str) -> np.ndarray:
        return load_image(workspace / "images" / name, size)

    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    written = skipped = 0
    for i, view in enumerate(views):
        src_ids = view.src_ids[: int(infer_cfg["num_view"]) - 1]
        if not src_ids:
            skipped += 1
            logger.warning("No source views for '%s'; no depth map.", view.name)
            continue
        group = [view] + [by_id[s] for s in src_ids]
        sample = build_sample(
            group, [image(v.name) for v in group], size, int(cfg["model"]["num_depth"])
        )
        with (
            torch.no_grad(),
            torch.autocast(device.type, enabled=bool(infer_cfg["amp"])),
        ):
            outputs = model(
                sample["imgs"].to(device),
                {k: v.to(device) for k, v in sample["proj_matrices"].items()},
                sample["depth_values"].to(device),
            )
        np.savez(
            out_dir / f"{view.image_id:08d}.npz",
            depth=outputs["depth"][0].float().cpu().numpy(),
            confidence=combined_confidence(outputs).float().cpu().numpy(),
            K=sample["K"],
            extrinsic=view.extrinsic,
            src_ids=np.array(view.src_ids, dtype=np.int64),
            name=np.array(view.name),
        )
        written += 1
        if (i + 1) % 20 == 0 or i + 1 == len(views):
            logger.info("TransMVSNet: %d/%d views", i + 1, len(views))

    stats = {
        "checkpoint_sha256": digest,
        "input_size": list(size),
        "depth_maps": written,
        "views_without_sources": skipped,
        "seconds": round(time.perf_counter() - start, 1),
        "device": str(device),
    }
    if device.type == "cuda":
        stats["peak_vram_gb"] = round(
            torch.cuda.max_memory_allocated(device) / 2**30, 2
        )
    logger.info("TransMVSNet inference done: %s", stats)
    return stats
