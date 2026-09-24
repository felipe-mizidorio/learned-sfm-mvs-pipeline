"""Input image discovery and camera-intrinsics parsing shared by SfM backends."""

import re
from pathlib import Path

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"})

_DIGITS = re.compile(r"(\d+)")


def natural_sort_key(name: str) -> list[int | str]:
    """Sort key that orders embedded numbers numerically.

    ``frame_9.jpg`` sorts before ``frame_10.jpg``, so capture order survives
    frame names that are not zero-padded — sequential pairing depends on it.

    Parameters
    ----------
    name : str
        Image name.

    Returns
    -------
    list[int | str]
        Alternating text and integer chunks.
    """
    return [int(chunk) if chunk.isdigit() else chunk for chunk in _DIGITS.split(name)]


def list_images(image_dir: Path) -> list[str]:
    """List images under ``image_dir`` recursively, in capture order.

    Parameters
    ----------
    image_dir : Path
        Root image directory.

    Returns
    -------
    list[str]
        POSIX paths relative to ``image_dir``, naturally sorted.

    Raises
    ------
    ValueError
        If ``image_dir`` does not exist or contains no images.
    """
    if not image_dir.is_dir():
        raise ValueError(f"image_dir does not exist: {image_dir}")
    names = [
        p.relative_to(image_dir).as_posix()
        for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not names:
        raise ValueError(f"No images found in {image_dir}")
    return sorted(names, key=natural_sort_key)


def normalize_camera_params(params: str) -> str:
    """Normalize an intrinsics string to the comma-separated form COLMAP parses.

    COLMAP 4.x rejects space-separated values (``"fx fy cx cy"`` fails its
    params check), so both separators are accepted here.

    Parameters
    ----------
    params : str
        Values separated by commas and/or whitespace.

    Returns
    -------
    str
        The same values, comma-separated.

    Raises
    ------
    ValueError
        If the string is empty or a value is not a number.
    """
    values = [v for v in re.split(r"[,\s]+", params.strip()) if v]
    if not values:
        raise ValueError("camera_params is empty")
    for value in values:
        try:
            float(value)
        except ValueError:
            raise ValueError(
                f"camera_params value is not a number: {value!r}"
            ) from None
    return ",".join(values)
