"""Image-pair selection for learned matching.

Pairs are unordered: ``(a, b)`` and ``(b, a)`` are the same pair and only the
first occurrence is kept, so strategies can be combined freely.
"""

from collections.abc import Iterable
from itertools import combinations
from pathlib import Path

Pair = tuple[str, str]


def sequential_pairs(names: list[str], overlap: int) -> list[Pair]:
    """Pair each image with the next ``overlap`` images in capture order.

    Parameters
    ----------
    names : list[str]
        Image names in capture order.
    overlap : int
        Number of following images to pair with each image.

    Returns
    -------
    list[Pair]
        Pairs in order of the first image.

    Raises
    ------
    ValueError
        If ``overlap`` is not positive.
    """
    if overlap < 1:
        raise ValueError(f"overlap must be >= 1, got {overlap}")
    return [
        (names[i], names[j])
        for i in range(len(names))
        for j in range(i + 1, min(i + 1 + overlap, len(names)))
    ]


def exhaustive_pairs(names: list[str]) -> list[Pair]:
    """All unordered pairs of images.

    Parameters
    ----------
    names : list[str]
        Image names.

    Returns
    -------
    list[Pair]
        ``len(names) * (len(names) - 1) / 2`` pairs.
    """
    return list(combinations(names, 2))


def merge_pairs(*pair_lists: Iterable[Pair]) -> list[Pair]:
    """Concatenate pair lists, dropping self-pairs and unordered duplicates.

    Parameters
    ----------
    *pair_lists : Iterable[Pair]
        Pair lists to merge, in priority order.

    Returns
    -------
    list[Pair]
        Unique pairs, first occurrence wins.
    """
    seen: set[frozenset[str]] = set()
    merged: list[Pair] = []
    for pairs in pair_lists:
        for a, b in pairs:
            key = frozenset((a, b))
            if a == b or key in seen:
                continue
            seen.add(key)
            merged.append((a, b))
    return merged


def write_pairs(pairs: Iterable[Pair], path: Path) -> None:
    """Write pairs in hloc/COLMAP format: one ``name0 name1`` line per pair.

    Parameters
    ----------
    pairs : Iterable[Pair]
        Pairs to write.
    path : Path
        Output text file.

    Raises
    ------
    ValueError
        If a name contains whitespace (the format is space-delimited).
    """
    lines = []
    for a, b in pairs:
        if any(c.isspace() for c in a + b):
            raise ValueError(f"image names must not contain whitespace: {a!r}, {b!r}")
        lines.append(f"{a} {b}\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def read_pairs(path: Path) -> list[Pair]:
    """Read a pairs file written by ``write_pairs`` or hloc.

    Parameters
    ----------
    path : Path
        Pairs text file.

    Returns
    -------
    list[Pair]
        Pairs in file order.
    """
    pairs = []
    for line in path.read_text().splitlines():
        if line.strip():
            a, b = line.split()
            pairs.append((a, b))
    return pairs
