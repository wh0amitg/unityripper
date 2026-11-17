from __future__ import annotations

import math
from typing import Iterable, Sequence

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Vec4 = tuple[float, float, float, float]
Mat4 = tuple[float, ...]
Mat3 = tuple[float, ...]

IDENTITY: Mat4 = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)

def scale_matrix(scale: float) -> Mat4:
    return (scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 0.0, 1.0)

def quat_to_mat3(x: float, y: float, z: float, w: float) -> Mat3:
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return (
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
        2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy),
    )

def trs_matrix(position: Sequence[float], rotation: Sequence[float], scale: Sequence[float]) -> Mat4:
    px, py, pz = float(position[0]), float(position[1]), float(position[2])
    qx, qy, qz, qw = (float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3]))
    sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    r = quat_to_mat3(qx, qy, qz, qw)
    return (
        r[0] * sx, r[1] * sy, r[2] * sz, px,
        r[3] * sx, r[4] * sy, r[5] * sz, py,
        r[6] * sx, r[7] * sy, r[8] * sz, pz,
        0.0, 0.0, 0.0, 1.0,
    )

def mat_mul(a: Mat4, b: Mat4) -> Mat4:
    return (
        a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12],
        a[0] * b[1] + a[1] * b[5] + a[2] * b[9] + a[3] * b[13],
        a[0] * b[2] + a[1] * b[6] + a[2] * b[10] + a[3] * b[14],
        a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3] * b[15],
        a[4] * b[0] + a[5] * b[4] + a[6] * b[8] + a[7] * b[12],
        a[4] * b[1] + a[5] * b[5] + a[6] * b[9] + a[7] * b[13],
        a[4] * b[2] + a[5] * b[6] + a[6] * b[10] + a[7] * b[14],
        a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7] * b[15],
        a[8] * b[0] + a[9] * b[4] + a[10] * b[8] + a[11] * b[12],
        a[8] * b[1] + a[9] * b[5] + a[10] * b[9] + a[11] * b[13],
        a[8] * b[2] + a[9] * b[6] + a[10] * b[10] + a[11] * b[14],
        a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11] * b[15],
        a[12] * b[0] + a[13] * b[4] + a[14] * b[8] + a[15] * b[12],
        a[12] * b[1] + a[13] * b[5] + a[14] * b[9] + a[15] * b[13],
        a[12] * b[2] + a[13] * b[6] + a[14] * b[10] + a[15] * b[14],
        a[12] * b[3] + a[13] * b[7] + a[14] * b[11] + a[15] * b[15],
    )

def transform_point(m: Mat4, p: Sequence[float]) -> Vec3:
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    return (
        m[0] * x + m[1] * y + m[2] * z + m[3],
        m[4] * x + m[5] * y + m[6] * z + m[7],
        m[8] * x + m[9] * y + m[10] * z + m[11],
    )

def transform_vector(m: Mat4, v: Sequence[float]) -> Vec3:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return (
        m[0] * x + m[1] * y + m[2] * z,
        m[4] * x + m[5] * y + m[6] * z,
        m[8] * x + m[9] * y + m[10] * z,
    )

def det3(m: Mat4) -> float:
    return (
        m[0] * (m[5] * m[10] - m[6] * m[9])
        - m[1] * (m[4] * m[10] - m[6] * m[8])
        + m[2] * (m[4] * m[9] - m[5] * m[8])
    )

def inverse_transpose3(m: Mat4) -> Mat3:
    a, b, c = m[0], m[1], m[2]
    d, e, f = m[4], m[5], m[6]
    g, h, i = m[8], m[9], m[10]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    inv_det = 1.0 / det
    return (
        (e * i - f * h) * inv_det, (c * h - b * i) * inv_det, (b * f - c * e) * inv_det,
        (f * g - d * i) * inv_det, (a * i - c * g) * inv_det, (c * d - a * f) * inv_det,
        (d * h - e * g) * inv_det, (b * g - a * h) * inv_det, (a * e - b * d) * inv_det,
    )

def transform_normal(m: Mat4, n: Sequence[float]) -> Vec3:
    nm = inverse_transpose3(m)
    x, y, z = float(n[0]), float(n[1]), float(n[2])
    rx = nm[0] * x + nm[1] * y + nm[2] * z
    ry = nm[3] * x + nm[4] * y + nm[5] * z
    rz = nm[6] * x + nm[7] * y + nm[8] * z
    length = math.sqrt(rx * rx + ry * ry + rz * rz)
    if length < 1e-12:
        return (0.0, 0.0, 0.0)
    return (rx / length, ry / length, rz / length)

def normalize3(v: Sequence[float]) -> Vec3:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    length = math.sqrt(x * x + y * y + z * z)
    if length < 1e-12:
        return (0.0, 0.0, 0.0)
    return (x / length, y / length, z / length)

def mirror_x_point(p: Sequence[float]) -> Vec3:
    return (-float(p[0]), float(p[1]), float(p[2]))

def mirror_x_normal(n: Sequence[float]) -> Vec3:
    return (-float(n[0]), float(n[1]), float(n[2]))

def matrix_from_unity(m: object) -> Mat4:
    return tuple(float(getattr(m, f"e{r}{c}")) for r in range(4) for c in range(4))

def mat4_to_gltf(m: Mat4) -> list[float]:
    return [m[0], m[4], m[8], m[12], m[1], m[5], m[9], m[13], m[2], m[6], m[10], m[14], m[3], m[7], m[11], m[15]]

def iter_floats(values: Iterable[float]) -> list[float]:
    out: list[float] = []
    for v in values:
        f = float(v)
        if not math.isfinite(f):
            f = 0.0
        out.append(f)
    return out
