"""Colour aligned meshes by their distance to the reference (diag/markerless only).

Takes the ``aligned_<name>.ply`` meshes written by ``cross_align.py`` (already
in the reference frame) and writes ``errmap_<name>.ply``: vertex colours from
blue (0) to red (``--max-mm`` or more), distances in mm via ``--mm-per-unit``
(the reference's scale). Open in any PLY viewer to see where a method departs
from the black reference.

Usage::

    python scripts/diag/spikes/error_map.py --target REF.ply --mm-per-unit 76.1 \\
        --max-mm 3 CROSS/aligned_white_neus.ply [...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def error_map(aligned: Path, target: Path, mm_per_unit: float, max_mm: float) -> dict:
    """Write ``errmap_<name>.ply`` next to ``aligned``.

    Parameters
    ----------
    aligned : Path
        Mesh already in the reference frame.
    target : Path
        Reference mesh.
    mm_per_unit : float
        Millimetres per reference unit.
    max_mm : float
        Distance mapped to full red.

    Returns
    -------
    dict
        Vertex distance quartiles in mm.
    """
    mesh = o3d.io.read_triangle_mesh(str(aligned))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(str(target)))
    )
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    d_mm = scene.compute_distance(o3d.core.Tensor(verts)).numpy() * mm_per_unit
    x = np.clip(d_mm / max_mm, 0, 1)[:, None]
    blue, green, red = (
        np.array([0.1, 0.3, 1.0]),
        np.array([0.2, 0.9, 0.2]),
        np.array([1.0, 0.1, 0.1]),
    )
    colours = np.where(
        x < 0.5, blue + (green - blue) * 2 * x, green + (red - green) * (2 * x - 1)
    )
    mesh.vertex_colors = o3d.utility.Vector3dVector(colours)
    mesh.compute_vertex_normals()
    out = aligned.with_name(aligned.name.replace("aligned_", "errmap_"))
    o3d.io.write_triangle_mesh(str(out), mesh)
    stats = {
        "mesh": out.name,
        "max_mm_red": max_mm,
        "vertex_dist_mm_p25_p50_p75_p90": [
            float(np.percentile(d_mm, q)) for q in (25, 50, 75, 90)
        ],
    }
    print(json.dumps(stats))
    return stats


def main() -> None:
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--mm-per-unit", type=float, required=True)
    parser.add_argument("--max-mm", type=float, default=3.0)
    parser.add_argument("aligned", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.aligned:
        error_map(path, args.target, args.mm_per_unit, args.max_mm)


if __name__ == "__main__":
    main()
