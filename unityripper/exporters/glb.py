from __future__ import annotations

import json
import math
import struct
import sys
from array import array
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..materials import MaterialData, TextureBinding
from ..mathutil import Mat4, IDENTITY, mat4_to_gltf, transform_normal, transform_point
from ..scene import MeshAsset, MeshData, RendererModel
from ..textures import TextureAsset, TextureStore
from ..util import asset_file_of, format_size, log, object_path_id, sanitize_filename
from .common import (
    ExportOptions,
    compute_normals,
    flatten_triangles,
    material_key,
    usable_normals,
)

MAGIC = 0x46546C67
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

def _f32(values: Iterable[float]) -> bytes:
    data = array("f", (float(v) for v in values))
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()

def _u32(values: Iterable[int]) -> bytes:
    data = array("I", (int(v) for v in values))
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()

def _u16(values: Iterable[int]) -> bytes:
    data = array("H", (int(v) for v in values))
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()

def _u8(values: Iterable[int]) -> bytes:
    return bytes(max(0, min(255, int(v))) for v in values)

def _finite(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0

WRAP_MODES = {0: 10497, 1: 33071, 2: 33648, 3: 33648}

class GlbWriter:
    def __init__(self, textures: TextureStore, options: ExportOptions):
        self.textures = textures
        self.options = options
        self.bin = bytearray()
        self.views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []
        self.meshes: list[dict[str, Any]] = []
        self.materials: list[dict[str, Any]] = []
        self.textures_json: list[dict[str, Any]] = []
        self.images: list[dict[str, Any]] = []
        self.samplers: list[dict[str, Any]] = []
        self.nodes: list[dict[str, Any]] = []
        self.skins: list[dict[str, Any]] = []
        self.extensions_used: set[str] = set()
        self._image_cache: dict[Path, int] = {}
        self._sampler_cache: dict[tuple[int, int], int] = {}
        self._texture_cache: dict[tuple[int, int], int] = {}
        self._material_cache: dict[tuple, int] = {}
        self._mesh_cache: dict[tuple, int] = {}
        self._material_names: set[str] = set()
        self._mesh_names: set[str] = set()

    def _add_view(self, data: bytes, target: int | None = None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        offset = len(self.bin)
        self.bin.extend(data)
        view: dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        return len(self.views) - 1

    def _add_accessor(
        self,
        data: bytes,
        component_type: int,
        count: int,
        type_: str,
        target: int | None,
        minimum: Sequence[float] | None = None,
        maximum: Sequence[float] | None = None,
        normalized: bool = False,
        no_view: bool = False,
    ) -> int:
        accessor: dict[str, Any] = {
            "componentType": component_type,
            "count": count,
            "type": type_,
        }
        if not no_view:
            accessor["bufferView"] = self._add_view(data, target)
        if minimum is not None:
            accessor["min"] = [round(float(v), 6) for v in minimum]
        if maximum is not None:
            accessor["max"] = [round(float(v), 6) for v in maximum]
        if normalized:
            accessor["normalized"] = True
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def _add_image(self, asset: TextureAsset) -> tuple[int, int]:
        image_index = self._image_cache.get(asset.file)
        if image_index is None:
            view = self._add_view(asset.png_bytes())
            self.images.append(
                {"bufferView": view, "mimeType": "image/png", "name": asset.name or asset.file.stem}
            )
            image_index = len(self.images) - 1
            self._image_cache[asset.file] = image_index

        sampler_key = (asset.wrap_u, asset.wrap_v)
        sampler_index = self._sampler_cache.get(sampler_key)
        if sampler_index is None:
            sampler = {
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": WRAP_MODES.get(asset.wrap_u, 10497),
                "wrapT": WRAP_MODES.get(asset.wrap_v, 10497),
            }
            self.samplers.append(sampler)
            sampler_index = len(self.samplers) - 1
            self._sampler_cache[sampler_key] = sampler_index

        texture_key = (image_index, sampler_index)
        texture_index = self._texture_cache.get(texture_key)
        if texture_index is None:
            self.textures_json.append(
                {"sampler": sampler_index, "source": image_index, "name": asset.name or asset.file.stem}
            )
            texture_index = len(self.textures_json) - 1
            self._texture_cache[texture_key] = texture_index
        return texture_index, sampler_index

    def _texture_info(self, binding: TextureBinding | None, hint: str, role: str) -> dict[str, Any] | None:
        if binding is None or binding.texture is None or not self.textures.enabled:
            return None
        asset = self.textures.save(binding.texture, hint=hint, role=role)
        if asset is None:
            return None
        index, _ = self._add_image(asset)
        info: dict[str, Any] = {"index": index, "texCoord": 0}
        scale = binding.scale
        offset = binding.offset
        if scale != (1.0, 1.0) or offset != (0.0, 0.0):
            self.extensions_used.add("KHR_texture_transform")
            info["extensions"] = {
                "KHR_texture_transform": {
                    "offset": [round(offset[0], 6), round(1.0 - offset[1] - scale[1], 6)],
                    "scale": [round(scale[0], 6), round(scale[1], 6)],
                }
            }
        return info

    def _unique_material_name(self, material: MaterialData) -> str:
        from ..util import sanitize_filename

        base = sanitize_filename(material.name or "material", "material", 64)
        name = base
        i = 2
        while name in self._material_names:
            name = f"{base}_{i}"
            i += 1
        self._material_names.add(name)
        return name

    def add_material(self, material: MaterialData) -> int:
        key = material_key(material)
        cached = self._material_cache.get(key)
        if cached is not None:
            return cached

        name = self._unique_material_name(material)
        pbr: dict[str, Any] = {
            "baseColorFactor": [round(max(0.0, min(1.0, float(c))), 6) for c in material.base_color],
        }
        albedo = self._texture_info(material.albedo, f"{name}_albedo", "albedo")
        if albedo is not None:
            pbr["baseColorTexture"] = albedo

        if material.metallic_map is not None and self.options.mr_mode != "off" and self.textures.enabled:
            asset = self.textures.save_metallic_roughness(
                material.metallic_map.texture, hint=name
            )
            if asset is not None:
                index, _ = self._add_image(asset)
                pbr["metallicRoughnessTexture"] = {"index": index, "texCoord": 0}
                pbr["metallicFactor"] = 1.0
                pbr["roughnessFactor"] = 1.0
            else:
                pbr["metallicFactor"] = material.metallic
                pbr["roughnessFactor"] = material.roughness
        else:
            pbr["metallicFactor"] = round(material.metallic, 6)
            pbr["roughnessFactor"] = round(material.roughness, 6)

        result: dict[str, Any] = {"name": name, "pbrMetallicRoughness": pbr}
        if material.alpha_mode != "OPAQUE":
            result["alphaMode"] = material.alpha_mode
            if material.alpha_mode == "MASK":
                result["alphaCutoff"] = round(material.alpha_cutoff, 6)
        if material.double_sided:
            result["doubleSided"] = True
        if material.unlit:
            self.extensions_used.add("KHR_materials_unlit")
            result.setdefault("extensions", {})["KHR_materials_unlit"] = {}

        normal = self._texture_info(material.normal, f"{name}_normal", "normal")
        if normal is not None:
            bump = material.floats.get("_BumpScale", 1.0)
            normal["scale"] = round(float(bump), 6)
            result["normalTexture"] = normal

        occlusion = self._texture_info(material.occlusion, f"{name}_occlusion", "albedo")
        if occlusion is not None:
            strength = material.floats.get("_OcclusionStrength", 1.0)
            occlusion["strength"] = round(float(strength), 6)
            result["occlusionTexture"] = occlusion

        emission_info = self._texture_info(material.emission_map, f"{name}_emission", "albedo")
        if emission_info is not None:
            result["emissiveTexture"] = emission_info
        emission = [max(0.0, min(1.0, float(c))) for c in material.emission_color]
        if max(emission) > 0.001:
            peak = max(emission)
            if peak > 1.0:
                emission = [c / peak for c in emission]
                self.extensions_used.add("KHR_materials_emissive_strength")
                result.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                    "emissiveStrength": round(peak, 6)
                }
            result["emissiveFactor"] = [round(c, 6) for c in emission]

        self.materials.append(result)
        index = len(self.materials) - 1
        self._material_cache[key] = index
        return index

    def _add_attribute(
        self, values: Iterable[float], count: int, type_: str, target: int, minmax: bool = False
    ) -> int:
        flat = [float(v) for v in values]
        components = int(type_[-1]) if type_.startswith("VEC") else 1
        minimum = maximum = None
        if minmax:
            minimum = [min(flat[i::components]) for i in range(components)]
            maximum = [max(flat[i::components]) for i in range(components)]
        return self._add_accessor(_f32(flat), 5126, count, type_, target, minimum, maximum)

    def add_mesh(
        self,
        mesh: MeshData,
        materials: Sequence[MaterialData],
        name: str,
        *,
        skin: bool = False,
        bone_count: int = 0,
    ) -> int | None:
        key = (
            mesh.key,
            tuple(material_key(m) for m in materials),
            bool(skin),
        )
        cached = self._mesh_cache.get(key)
        if cached is not None:
            return cached

        if not mesh.vertices or mesh.triangle_count == 0:
            return None

        uv_flipped = [(uv[0], 1.0 - uv[1]) for uv in mesh.uvs] if mesh.uvs else None
        attributes: dict[str, int] = {}
        attributes["POSITION"] = self._add_attribute(
            (c for v in mesh.vertices for c in v), len(mesh.vertices), "VEC3", ARRAY_BUFFER, minmax=True
        )
        normals = usable_normals(mesh)
        if normals is None:
            normals = compute_normals(mesh.vertices, flatten_triangles(mesh))
        attributes["NORMAL"] = self._add_attribute(
            (c for n in normals for c in n), len(normals), "VEC3", ARRAY_BUFFER
        )
        if uv_flipped:
            attributes["TEXCOORD_0"] = self._add_attribute(
                (c for uv in uv_flipped for c in uv), len(uv_flipped), "VEC2", ARRAY_BUFFER
            )
        if mesh.colors:
            attributes["COLOR_0"] = self._add_attribute(
                (c for col in mesh.colors for c in col), len(mesh.colors), "VEC4", ARRAY_BUFFER
            )

        if skin and mesh.joints and mesh.weights and bone_count > 0:
            joint_values: list[int] = []
            weight_values: list[float] = []
            valid = True
            for j, w in zip(mesh.joints, mesh.weights):
                if any((int(x) < 0 or int(x) >= bone_count) for x in j):
                    valid = False
                    break
                total = sum(max(0.0, float(x)) for x in w)
                if total <= 0.0:
                    total = 1.0
                joint_values.extend(int(x) for x in j)
                weight_values.extend(max(0.0, float(x)) / total for x in w)
            if valid:
                max_joint = max(joint_values) if joint_values else 0
                if max_joint < 256:
                    jdata = _u8(joint_values)
                    jtype = 5121
                else:
                    jdata = _u16(joint_values)
                    jtype = 5123
                attributes["JOINTS_0"] = self._add_accessor(
                    jdata, jtype, len(mesh.joints), "VEC4", ARRAY_BUFFER
                )
                attributes["WEIGHTS_0"] = self._add_accessor(
                    _f32(weight_values), 5126, len(mesh.joints), "VEC4", ARRAY_BUFFER
                )

        primitives: list[dict[str, Any]] = []
        for sub_index, tris in enumerate(mesh.triangles):
            if not tris:
                continue
            indices = [index for tri in tris for index in tri]
            if max(indices) >= len(mesh.vertices):
                continue
            if len(mesh.vertices) <= 65535:
                index_data = _u16(indices)
                component = 5123
            else:
                index_data = _u32(indices)
                component = 5125
            index_accessor = self._add_accessor(
                index_data, component, len(indices), "SCALAR", ELEMENT_ARRAY_BUFFER
            )
            material = materials[min(sub_index, len(materials) - 1)] if materials else None
            primitive: dict[str, Any] = {
                "attributes": attributes,
                "indices": index_accessor,
                "mode": 4,
            }
            if material is not None:
                primitive["material"] = self.add_material(material)
            primitives.append(primitive)

        if not primitives:
            return None

        base = sanitize_filename(name, "mesh", 80)
        unique = base
        i = 2
        while unique in self._mesh_names:
            unique = f"{base}_{i}"
            i += 1
        self._mesh_names.add(unique)
        self.meshes.append({"name": unique, "primitives": primitives})
        index = len(self.meshes) - 1
        self._mesh_cache[key] = index
        return index

    def add_node(
        self,
        name: str,
        *,
        translation: Sequence[float] | None = None,
        rotation: Sequence[float] | None = None,
        scale: Sequence[float] | None = None,
        matrix: Mat4 | None = None,
        mesh: int | None = None,
        skin: int | None = None,
        children: Sequence[int] | None = None,
    ) -> int:
        node: dict[str, Any] = {"name": name}
        if matrix is not None:
            node["matrix"] = [round(_finite(v), 6) for v in mat4_to_gltf(matrix)]
        if translation is not None:
            node["translation"] = [round(_finite(v), 6) for v in translation]
        if rotation is not None:
            q = [round(_finite(v), 6) for v in rotation]
            length = math.sqrt(sum(v * v for v in q))
            if length > 1e-9:
                q = [v / length for v in q]
            else:
                q = [0.0, 0.0, 0.0, 1.0]
            node["rotation"] = q
        if scale is not None:
            node["scale"] = [round(_finite(v), 6) for v in scale]
        if mesh is not None:
            node["mesh"] = mesh
        if skin is not None:
            node["skin"] = skin
        if children:
            node["children"] = list(children)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def add_skin(self, joints: Sequence[int], bindposes: Sequence[Mat4]) -> int:
        flat: list[float] = []
        for matrix in bindposes:
            flat.extend(mat4_to_gltf(matrix))
        accessor = self._add_accessor(_f32(flat), 5126, len(bindposes), "MAT4", None)
        self.skins.append({"joints": list(joints), "inverseBindMatrices": accessor})
        return len(self.skins) - 1

    def finish(self, path: Path, scene_name: str, roots: Sequence[int]) -> Path:
        while len(self.bin) % 4:
            self.bin.append(0)
        document: dict[str, Any] = {
            "asset": {"version": "2.0", "generator": "UnityModelRipper"},
            "scene": 0,
            "scenes": [{"name": scene_name, "nodes": list(roots)}],
            "nodes": self.nodes,
        }
        if self.meshes:
            document["meshes"] = self.meshes
        if self.materials:
            document["materials"] = self.materials
        if self.textures_json:
            document["textures"] = self.textures_json
        if self.images:
            document["images"] = self.images
        if self.samplers:
            document["samplers"] = self.samplers
        if self.skins:
            document["skins"] = self.skins
        if self.accessors:
            document["accessors"] = self.accessors
        if self.views:
            document["bufferViews"] = self.views
        if self.bin:
            document["buffers"] = [{"byteLength": len(self.bin)}]
        if self.extensions_used:
            document["extensionsUsed"] = sorted(self.extensions_used)

        json_bytes = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        while len(json_bytes) % 4:
            json_bytes += b" "

        total = 12 + 8 + len(json_bytes) + (8 + len(self.bin) if self.bin else 0)
        out = bytearray()
        out.extend(struct.pack("<III", MAGIC, 2, total))
        out.extend(struct.pack("<II", len(json_bytes), CHUNK_JSON))
        out.extend(json_bytes)
        if self.bin:
            out.extend(struct.pack("<II", len(self.bin), CHUNK_BIN))
            out.extend(self.bin)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(out)
        log.info("GLB %s (%s, %d mesh(es), %d material(s))", path.name, format_size(len(out)), len(self.meshes), len(self.materials))
        return path

def _root_scale(options: ExportOptions) -> list[float] | None:
    scale = float(options.scale)
    values = [-scale, scale, scale] if options.mirror_x else [scale, scale, scale]
    if values == [1.0, 1.0, 1.0]:
        return None
    return values

def _add_asset_node(writer: GlbWriter, asset: MeshAsset, mesh_index: int) -> int:
    if asset.matrix != IDENTITY:
        return writer.add_node(asset.name, mesh=mesh_index, matrix=asset.matrix)
    scale = (asset.scale, asset.scale, asset.scale) if asset.scale != 1.0 else None
    return writer.add_node(asset.name, mesh=mesh_index, scale=scale)

def export_glb_assets(path: Path, assets: Sequence[MeshAsset], textures: TextureStore, options: ExportOptions) -> Path | None:
    writer = GlbWriter(textures, options)
    roots: list[int] = []
    for asset in assets:
        mesh_index = writer.add_mesh(asset.mesh, asset.materials, asset.name)
        if mesh_index is None:
            continue
        roots.append(_add_asset_node(writer, asset, mesh_index))
    if not roots:
        return None
    root_scale = _root_scale(options)
    if root_scale is not None:
        root = writer.add_node("UnityRoot", scale=root_scale, children=roots)
        roots = [root]
    return writer.finish(path, "models", roots)

def export_glb_assets_split(
    directory: Path,
    assets: Sequence[MeshAsset],
    textures: TextureStore,
    options: ExportOptions,
) -> list[tuple[str, Path]]:
    directory.mkdir(parents=True, exist_ok=True)
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
            writer = GlbWriter(textures, options)
            mesh_index = writer.add_mesh(asset.mesh, asset.materials, asset.name)
            if mesh_index is None:
                continue
            roots = [_add_asset_node(writer, asset, mesh_index)]
            root_scale = _root_scale(options)
            if root_scale is not None:
                roots = [writer.add_node("UnityRoot", scale=root_scale, children=roots)]
            written.append((asset.name, writer.finish(directory / f"{candidate}.glb", asset.name, roots)))
        except Exception as exc:
            log.warning("GLB export failed for %s: %s: %s", asset.name, type(exc).__name__, exc)
    return written

class _SceneNodes:

    def __init__(self, writer: GlbWriter):
        self.writer = writer
        self.by_transform: dict[tuple[int, int], int] = {}
        self.roots: list[int] = []

    @staticmethod
    def _key(transform: Any) -> tuple[int, int]:
        reader = getattr(transform, "object_reader", None)
        if reader is not None:
            return (id(asset_file_of(reader)), object_path_id(reader))
        return (id(asset_file_of(transform)), object_path_id(transform))

    @staticmethod
    def _vec(value: Any, size: int) -> tuple[float, ...]:
        try:
            if size == 3:
                return (float(value.x), float(value.y), float(value.z))
            return (float(value.x), float(value.y), float(value.z), float(value.w))
        except Exception:
            return tuple(1.0 if size == 4 and i == 3 else 0.0 for i in range(size))

    def ensure(self, transform: Any, depth: int = 0) -> int | None:
        if transform is None or depth > 512:
            return None
        if not hasattr(transform, "m_LocalScale") and hasattr(transform, "read"):
            try:
                transform = transform.read()
            except Exception:
                return None
        key = self._key(transform)
        if key in self.by_transform:
            return self.by_transform[key]

        gameobject = None
        father_ptr = getattr(transform, "m_Father", None)
        try:
            if father_ptr is not None and getattr(father_ptr, "m_PathID", 0):
                gameobject_ptr = getattr(transform, "m_GameObject", None)
            else:
                gameobject_ptr = getattr(transform, "m_GameObject", None)
            gameobject = gameobject_ptr.read() if gameobject_ptr is not None and getattr(gameobject_ptr, "m_PathID", 0) else None
        except Exception:
            gameobject = None
        name = str(getattr(gameobject, "m_Name", "") or "node") if gameobject is not None else "node"

        index = self.writer.add_node(
            name,
            translation=self._vec(getattr(transform, "m_LocalPosition", None), 3),
            rotation=self._vec(getattr(transform, "m_LocalRotation", None), 4),
            scale=self._vec(getattr(transform, "m_LocalScale", None), 3),
        )
        self.by_transform[key] = index

        father = None
        if father_ptr is not None and getattr(father_ptr, "m_PathID", 0):
            try:
                father = father_ptr.read()
            except Exception:
                father = None
        father_index = self.ensure(father, depth + 1) if father is not None else None
        if father_index is None:
            self.roots.append(index)
        else:
            children = self.writer.nodes[father_index].setdefault("children", [])
            children.append(index)
        return index

def export_glb_scene(
    path: Path,
    models: Sequence[RendererModel],
    textures: TextureStore,
    options: ExportOptions,
    scene_name: str = "scene",
) -> Path | None:
    writer = GlbWriter(textures, options)
    graph = _SceneNodes(writer)
    attached = 0

    for model in models:
        if model.mesh is None:
            continue
        transform = getattr(model.gameobject, "m_Transform", None)
        node_index = graph.ensure(transform) if transform is not None else None
        if node_index is None:
            continue

        skin_index: int | None = None
        if model.is_skinned and model.bones and model.bindposes and model.mesh.joints and model.mesh.weights:
            joints: list[int] = []
            valid = True
            for bone in model.bones:
                joint_node = graph.ensure(bone)
                if joint_node is None:
                    valid = False
                    break
                joints.append(joint_node)
            if valid and len(joints) == len(model.bindposes):
                skin_index = writer.add_skin(joints, model.bindposes)

        mesh_index = writer.add_mesh(
            model.mesh,
            model.materials,
            model.path or model.name,
            skin=skin_index is not None,
            bone_count=len(model.bones),
        )
        if mesh_index is None:
            continue

        target = node_index
        if "mesh" in writer.nodes[target]:
            target = writer.add_node(f"{model.name}_mesh")
            writer.nodes[node_index].setdefault("children", []).append(target)
        writer.nodes[target]["mesh"] = mesh_index
        if skin_index is not None:
            writer.nodes[target]["skin"] = skin_index
        attached += 1

    if not attached:
        return None

    roots = list(graph.roots)
    root_scale = _root_scale(options)
    if root_scale is not None:
        roots = [writer.add_node("UnityRoot", scale=root_scale, children=roots)]
    return writer.finish(path, scene_name, roots)
