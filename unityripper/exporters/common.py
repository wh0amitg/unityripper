from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..materials import MaterialData
from ..mathutil import Vec3, det3
from ..scene import MeshData
from ..util import object_path_id, sanitize_filename, sanitize_node_path

def renderer_stem(model: object, max_length: int = 110) -> str:
    path = getattr(model, "path", None) or getattr(model, "name", "") or "object"
    stem = sanitize_node_path(str(path))
    path_id = object_path_id(getattr(model, "obj", None))
    if path_id:
        stem = f"{stem}_{path_id}"
    return sanitize_filename(stem, "object", max_length)

@dataclass
class ExportOptions:
    mirror_x: bool = True
    scale: float = 1.0
    textures: bool = True
    normal_mode: str = "auto"
    mr_mode: str = "auto"
    all_meshes: bool = False
    active_only: bool = False

def needs_winding_reverse(world_det: float, mirror_x: bool) -> bool:
    if mirror_x:
        return world_det > 0.0
    return world_det < 0.0

def usable_normals(mesh: MeshData) -> list[Vec3] | None:
    normals = mesh.normals
    if not normals or len(normals) != len(mesh.vertices):
        return None
    zero = sum(1 for n in normals if abs(n[0]) + abs(n[1]) + abs(n[2]) < 1e-8)
    if zero > max(1, len(normals) // 10):
        return None
    return normals

def compute_normals(vertices: Sequence[Vec3], triangles: Iterable[Sequence[int]]) -> list[Vec3]:
    acc: list[list[float]] = [[0.0, 0.0, 0.0] for _ in vertices]
    for tri in triangles:
        if len(tri) < 3:
            continue
        ia, ib, ic = int(tri[0]), int(tri[1]), int(tri[2])
        if max(ia, ib, ic) >= len(vertices):
            continue
        ax, ay, az = vertices[ia]
        bx, by, bz = vertices[ib]
        cx, cy, cz = vertices[ic]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        for idx in (ia, ib, ic):
            acc[idx][0] += nx
            acc[idx][1] += ny
            acc[idx][2] += nz
    out: list[Vec3] = []
    for nx, ny, nz in acc:
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length < 1e-12:
            out.append((0.0, 0.0, 1.0))
        else:
            out.append((nx / length, ny / length, nz / length))
    return out

def flatten_triangles(mesh: MeshData) -> list[tuple[int, int, int]]:
    return [tri for sub in mesh.triangles for tri in sub]

def linear_to_srgb(value: float) -> float:
    value = max(0.0, min(1.0, float(value)))
    if value <= 0.0031308:
        return value * 12.92
    return 1.055 * (value ** (1.0 / 2.4)) - 0.055

def srgb_tuple(color: Sequence[float]) -> tuple[float, float, float]:
    return (linear_to_srgb(color[0]), linear_to_srgb(color[1]), linear_to_srgb(color[2]))

def texture_relpath(texture_file: Path, start: Path) -> str:
    return Path(os.path.relpath(texture_file, start)).as_posix()

def material_stem(index: int, material: MaterialData, prefix: str = "mat") -> str:
    from ..util import sanitize_filename

    name = sanitize_filename(material.name or f"material_{index}", f"material_{index}", 60)
    return f"{prefix}{index}_{name}"

def material_key(material: MaterialData) -> tuple:
    return material.key or (material.name, material.shader)
