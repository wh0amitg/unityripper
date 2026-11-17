from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ..materials import MaterialData, TextureBinding
from ..mathutil import Mat4, IDENTITY, det3, scale_matrix, transform_normal, transform_point
from ..scene import MeshAsset, MeshData, RendererModel
from ..textures import TextureStore
from ..util import format_size, log, sanitize_filename
from .common import (
    ExportOptions,
    compute_normals,
    flatten_triangles,
    material_key,
    material_stem,
    needs_winding_reverse,
    renderer_stem,
    srgb_tuple,
    texture_relpath,
    usable_normals,
)

Float = float

class ObjExporter:
    def __init__(self, output_dir: Path, textures: TextureStore, options: ExportOptions):
        self.output_dir = output_dir
        self.obj_dir = output_dir / "obj"
        self.textures = textures
        self.options = options

    def _texture_path(self, binding: TextureBinding | None, hint: str, role: str, from_dir: Path) -> str | None:
        if binding is None or binding.texture is None:
            return None
        asset = self.textures.save(binding.texture, hint=hint, role=role)
        if asset is None:
            return None
        return texture_relpath(asset.file, from_dir)

    def _write_material_block(self, lines: list[str], name: str, material: MaterialData, from_dir: Path) -> None:
        lines.append(f"newmtl {name}\n")
        r, g, b = srgb_tuple(material.base_color)
        lines.append(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
        alpha = 1.0 if material.alpha_mode == "OPAQUE" else material.base_color[3]
        lines.append(f"d {alpha:.6f}\n")
        spec = max(0.0, min(1.0, 1.0 - material.roughness))
        lines.append(f"Ns {spec * 900.0 + 10.0:.1f}\n")
        lines.append("illum 2\n")
        if material.emission_color and max(material.emission_color) > 0.001:
            er, eg, eb = srgb_tuple(material.emission_color)
            lines.append(f"Ke {er:.6f} {eg:.6f} {eb:.6f}\n")

        hint = material.name or "material"
        albedo = self._texture_path(material.albedo, f"{hint}_albedo", "albedo", from_dir)
        if albedo:
            scale = material.albedo.scale
            offset = material.albedo.offset
            options = ""
            if scale != (1.0, 1.0) or offset != (0.0, 0.0):
                options = f"-s {scale[0]:.6f} {scale[1]:.6f} -o {offset[0]:.6f} {offset[1]:.6f} "
            lines.append(f"map_Kd {options}{albedo}\n")

        normal = self._texture_path(material.normal, f"{hint}_normal", "normal", from_dir)
        if normal:
            lines.append(f"map_Bump {normal}\n")

        emission = self._texture_path(material.emission_map, f"{hint}_emission", "albedo", from_dir)
        if emission:
            lines.append(f"map_Ke {emission}\n")
        lines.append("\n")

    def _write_mtl(self, mtl_path: Path, materials: Sequence[MaterialData], prefix: str) -> list[str]:
        lines: list[str] = [f"# UnityModelRipper\n"]
        names: list[str] = []
        used: set[str] = set()
        for i, material in enumerate(materials):
            name = material_stem(i, material, prefix)
            base = name
            n = 2
            while name in used:
                name = f"{base}_{n}"
                n += 1
            used.add(name)
            names.append(name)
            self._write_material_block(lines, name, material, mtl_path.parent)
        mtl_path.write_text("".join(lines), encoding="utf-8")
        return names

    def _write_geometry(
        self,
        lines: list[str],
        mesh: MeshData,
        materials: Sequence[MaterialData],
        material_names: Sequence[str] | None,
        world: Mat4,
        vertex_offset: int,
        uv_offset: int,
        normal_offset: int,
        group_name: str,
    ) -> tuple[int, int, int]:
        positions = [transform_point(world, v) for v in mesh.vertices]
        if self.options.mirror_x:
            positions = [(-p[0], p[1], p[2]) for p in positions]
        if self.options.scale != 1.0:
            s = self.options.scale
            positions = [(p[0] * s, p[1] * s, p[2] * s) for p in positions]

        normals = usable_normals(mesh)
        if normals is not None:
            normals = [transform_normal(world, n) for n in normals]
            if self.options.mirror_x:
                normals = [(-n[0], n[1], n[2]) for n in normals]
        else:
            normals = compute_normals(mesh.vertices, flatten_triangles(mesh))
            normals = [transform_normal(world, n) for n in normals]
            if self.options.mirror_x:
                normals = [(-n[0], n[1], n[2]) for n in normals]

        uvs = mesh.uvs
        lines.extend(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for p in positions)
        if uvs:
            lines.extend(f"vt {uv[0]:.6f} {uv[1]:.6f}\n" for uv in uvs)
        lines.extend(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n" for n in normals)

        reverse = needs_winding_reverse(det3(world), self.options.mirror_x)
        for sub_index, tris in enumerate(mesh.triangles):
            if not tris:
                continue
            suffix = f"_{sub_index}" if len(mesh.triangles) > 1 else ""
            lines.append(f"g {group_name}{suffix}\n")
            if material_names:
                mat_index = min(sub_index, len(material_names) - 1)
                lines.append(f"usemtl {material_names[mat_index]}\n")
            for tri in tris:
                a, b, c = tri
                if reverse:
                    b, c = c, b
                va = vertex_offset + a + 1
                vb = vertex_offset + b + 1
                vc = vertex_offset + c + 1
                if uvs:
                    ta = uv_offset + a + 1
                    tb = uv_offset + b + 1
                    tc = uv_offset + c + 1
                    if normals:
                        na = normal_offset + a + 1
                        nb = normal_offset + b + 1
                        nc = normal_offset + c + 1
                        lines.append(f"f {va}/{ta}/{na} {vb}/{tb}/{nb} {vc}/{tc}/{nc}\n")
                    else:
                        lines.append(f"f {va}/{ta} {vb}/{tb} {vc}/{tc}\n")
                elif normals:
                    na = normal_offset + a + 1
                    nb = normal_offset + b + 1
                    nc = normal_offset + c + 1
                    lines.append(f"f {va}//{na} {vb}//{nb} {vc}//{nc}\n")
                else:
                    lines.append(f"f {va} {vb} {vc}\n")
        return len(positions), (len(uvs) if uvs else 0), len(normals)

    def export_assets(self, assets: Sequence[MeshAsset]) -> list[tuple[str, Path]]:
        self.obj_dir.mkdir(parents=True, exist_ok=True)
        written: list[tuple[str, Path]] = []
        used: set[str] = set()
        for asset in assets:
            stem = sanitize_filename(asset.name, "model")
            candidate = stem
            n = 2
            while candidate.lower() in used:
                candidate = f"{stem}_{n}"
                n += 1
            used.add(candidate.lower())
            try:
                written.append((asset.name, self._export_asset(asset, candidate)))
            except Exception as exc:
                log.warning("OBJ export failed for %s: %s: %s", asset.name, type(exc).__name__, exc)
        return written

    def _export_asset(self, asset: MeshAsset, stem: str) -> Path:
        mesh = asset.mesh
        materials = list(asset.materials) or [MaterialData(key=None, name="Default")]
        obj_path = self.obj_dir / f"{stem}.obj"
        mtl_path = obj_path.with_suffix(".mtl")

        lines = [f"# UnityModelRipper - {asset.name}\n", f"# mesh: {mesh.name}\n"]
        material_names: list[str] | None = None
        if materials:
            material_names = self._write_mtl(mtl_path, materials, prefix="m")
            lines.append(f"mtllib {mtl_path.name}\n")

        self._write_geometry(
            lines,
            mesh,
            materials,
            material_names,
            asset.matrix if asset.matrix != IDENTITY else scale_matrix(asset.scale),
            0,
            0,
            0,
            sanitize_filename(mesh.name or asset.name, "model"),
        )
        obj_path.write_text("".join(lines), encoding="utf-8")
        log.info("OBJ %s (%s)", obj_path.name, format_size(obj_path.stat().st_size))
        return obj_path

    def export_scene(self, models: Sequence[RendererModel], name: str = "scene") -> Path:
        self.obj_dir.mkdir(parents=True, exist_ok=True)
        obj_path = self.obj_dir / f"{sanitize_filename(name, 'scene')}.obj"
        mtl_path = obj_path.with_suffix(".mtl")

        materials: list[MaterialData] = []
        material_names: list[str] = []
        material_index: dict[tuple, int] = {}
        for model in models:
            for material in model.materials:
                key = material_key(material)
                if key not in material_index:
                    material_index[key] = len(materials)
                    materials.append(material)

        if materials:
            material_names = self._write_mtl(mtl_path, materials, prefix="mat")
        lines = [f"# UnityModelRipper - {name}\n"]
        if materials:
            lines.append(f"mtllib {mtl_path.name}\n")

        vertex_offset = uv_offset = normal_offset = 0
        for model in models:
            if model.mesh is None:
                continue
            sub_names = []
            for material in model.materials:
                idx = material_index.get(material_key(material), 0)
                sub_names.append(material_names[idx] if material_names else "")
            before_v, before_uv, before_n = vertex_offset, uv_offset, normal_offset
            dv, du, dn = self._write_geometry(
                lines,
                model.mesh,
                model.materials,
                sub_names,
                model.world,
                vertex_offset,
                uv_offset,
                normal_offset,
                sanitize_filename(model.path or model.name, "object"),
            )
            vertex_offset += dv
            uv_offset += du
            normal_offset += dn
            if (vertex_offset, uv_offset, normal_offset) == (before_v, before_uv, before_n):
                continue

        obj_path.write_text("".join(lines), encoding="utf-8")
        log.info("OBJ scene %s (%s)", obj_path.name, format_size(obj_path.stat().st_size))
        return obj_path

    def export_scene_split(self, models: Sequence[RendererModel]) -> list[tuple[RendererModel, Path]]:
        self.obj_dir.mkdir(parents=True, exist_ok=True)
        written: list[tuple[RendererModel, Path]] = []
        used: set[str] = set()
        for model in models:
            if model.mesh is None:
                continue
            stem = renderer_stem(model)
            candidate = stem
            n = 2
            while candidate.lower() in used:
                candidate = f"{stem}_{n}"
                n += 1
            used.add(candidate.lower())

            materials = list(model.materials) or [MaterialData(key=None, name="Default")]
            obj_path = self.obj_dir / f"{candidate}.obj"
            mtl_path = obj_path.with_suffix(".mtl")
            lines = [f"# UnityModelRipper - {model.path or model.name}\n", f"# mesh: {model.mesh.name}\n"]
            material_names = self._write_mtl(mtl_path, materials, prefix="m")
            lines.append(f"mtllib {mtl_path.name}\n")
            self._write_geometry(
                lines,
                model.mesh,
                materials,
                material_names,
                model.world,
                0,
                0,
                0,
                sanitize_filename(model.path or model.name or model.mesh.name, "object"),
            )
            obj_path.write_text("".join(lines), encoding="utf-8")
            written.append((model, obj_path))
        log.info("OBJ scene split into %d file(s)", len(written))
        return written
