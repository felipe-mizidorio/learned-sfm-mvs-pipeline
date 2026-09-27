"""Sharpest frame per stride window, saved as PNG (diag/markerless branch only).

D6 of the white-mannequin diagnosis: does better frame selection and lossless
frames help a textureless subject? Reads an existing aruco-frame-preprocessing
session (fixed-stride JPEG frames + SAM2 masks) and its source video, and
writes a new session whose frame ``k`` is the sharpest of the ``stride``
decoded frames starting at the old frame ``k``. Sharpness is the variance of
the Laplacian (after a 3x3 Gaussian, against sensor noise) inside the old
frame's subject mask, dilated so the silhouette edge counts. The new session
has only ``metadata.json`` and the PNG frames, ready for ``aruco-detect``.

Usage::

    python scripts/diag/sharpest_frames.py --session OLD --video V.mp4 --out-session NEW
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _roi(mask_path: Path, grow_px: int = 25) -> np.ndarray | None:
    """Dilated subject mask, or None when missing or full-frame."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or (mask > 0).mean() > 0.99:
        return None
    size = 2 * grow_px + 1
    return cv2.dilate((mask > 0).astype(np.uint8), np.ones((size, size), np.uint8)) > 0


def _sharpness(frame: np.ndarray, roi: np.ndarray | None) -> float:
    """Variance of the Laplacian of the smoothed grey image inside ``roi``."""
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    return float(lap[roi].var() if roi is not None else lap.var())


def sharpest_frames(session: Path, video: Path, out_session: Path) -> dict:
    """Write the sharpest-per-window PNG session.

    Parameters
    ----------
    session : Path
        Existing session (``metadata.json``, ``filtered/masks``).
    video : Path
        The session's source video.
    out_session : Path
        New session directory; must not exist.

    Returns
    -------
    dict
        Selection statistics (offset of the chosen frame in its window,
        sharpness gain over the fixed-stride frame).
    """
    meta = json.loads((session / "metadata.json").read_text())
    stride = int(meta["stride"])
    anchors = {f["frame_index"]: f for f in meta["frames"]}
    mask_dir = session / "filtered" / "masks"
    out_session.mkdir(parents=True)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    pad = len(str(int(meta["total_frames"])))
    best: dict[int, tuple[float, int, np.ndarray]] = {}
    anchor_score: dict[int, float] = {}
    rois: dict[int, np.ndarray | None] = {}
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        anchor = index - index % stride
        if anchor in anchors:
            if anchor not in rois:
                rois[anchor] = _roi(mask_dir / f"{anchors[anchor]['filename']}.png")
            score = _sharpness(frame, rois[anchor])
            if index == anchor:
                anchor_score[anchor] = score
            if anchor not in best or score > best[anchor][0]:
                best[anchor] = (score, index, frame)
        index += 1
    cap.release()

    frames, offsets, gains = [], [], []
    for anchor in sorted(best):
        score, chosen, frame = best[anchor]
        name = f"frame_{str(chosen).zfill(pad)}.png"
        cv2.imwrite(str(out_session / name), frame, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        frames.append(
            {
                "frame_index": chosen,
                "timestamp_s": round(chosen / fps, 4),
                "filename": name,
            }
        )
        offsets.append(chosen - anchor)
        if anchor_score.get(anchor):
            gains.append(score / anchor_score[anchor])

    new_meta = dict(meta)
    new_meta.update(
        frames=frames,
        frames_extracted=len(frames),
        diag_sharpest_frames={
            "source_session": str(session),
            "window": stride,
            "metric": "laplacian_var_gauss3_in_dilated_mask",
        },
    )
    (out_session / "metadata.json").write_text(json.dumps(new_meta, indent=2))
    stats = {
        "frames": len(frames),
        "decoded": index,
        "offset_in_window_median": float(np.median(offsets)),
        "chosen_anchor_itself": int(sum(o == 0 for o in offsets)),
        "sharpness_gain_median": float(np.median(gains)) if gains else None,
        "sharpness_gain_p10": float(np.percentile(gains, 10)) if gains else None,
    }
    (out_session / "sharpest_frames.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out-session", type=Path, required=True)
    args = parser.parse_args()
    sharpest_frames(args.session, args.video, args.out_session)


if __name__ == "__main__":
    main()
