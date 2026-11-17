from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..materials import MaterialData
from ..mathutil import Mat4, IDENTITY, det3, scale_matrix, transform_normal, transform_point
from ..scene import MeshAsset, MeshData, RendererModel
from ..textures import TextureAsset, TextureStore
from ..util import format_size, log, sanitize_filename
from .common import (
    ExportOptions,
    compute_normals,
    flatten_triangles,
    material_key,
    needs_winding_reverse,
    renderer_stem,
    srgb_tuple,
    texture_relpath,
    usable_normals,
)

FBX_VERSION = 7400
_HEAD_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
_FOOT_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
_FOOT_MAGIC = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"

@dataclass
class Node:
    name: str
    props: list[Any] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)

    def add(self, name: str, *props: Any) -> "Node":
        child = Node(name, list(props))
        self.children.append(child)
        return child

    def child(self, name: str) -> "Node":
        node = Node(name)
        self.children.append(node)
        return node

@dataclass
class Array:

    kind: str
    values: Sequence[float] | Sequence[int]

    def pack(self) -> bytes:
        values = self.values
        count = len(values)
        if self.kind == "d":
            raw = struct.pack(f"<{count}d", *values)
        elif self.kind == "f":
            raw = struct.pack(f"<{count}f", *values)
        elif self.kind == "i":
            raw = struct.pack(f"<{count}i", *values)
        elif self.kind == "l":
            raw = struct.pack(f"<{count}q", *values)
        elif self.kind == "b":
            raw = bytes(1 if value else 0 for value in values)
        else:
            raise ValueError(f"unsupported FBX array kind {self.kind!r}")
        data = zlib.compress(raw, 6)
        if len(data) < len(raw):
            return self.kind.encode() + struct.pack("<III", count, 1, len(data)) + data
        return self.kind.encode() + struct.pack("<III", count, 0, len(raw)) + raw

class Long(int):
    pass

def pack_prop(value: Any) -> bytes:
    if isinstance(value, bool):
        return b"B" + (b"\x01" if value else b"\x00")
    if isinstance(value, Long):
        return b"L" + struct.pack("<q", value)
    if isinstance(value, int):
        if -(2**31) <= value < 2**31:
            return b"I" + struct.pack("<i", value)
        return b"L" + struct.pack("<q", value)
    if isinstance(value, float):
        return b"D" + struct.pack("<d", value)
    if isinstance(value, str):
        raw = value.encode("utf-8")
        return b"S" + struct.pack("<I", len(raw)) + raw
    if isinstance(value, Array):
        return value.pack()
    raise TypeError(f"cannot encode FBX property {type(value).__name__}")

def _write_node(fp, node: Node) -> None:
    start = fp.tell()
    name = node.name.encode("ascii")
    fp.write(b"\x00" * (13 + len(name)))
    body = b"".join(pack_prop(prop) for prop in node.props)
    fp.write(body)
    for child in node.children:
        _write_node(fp, child)
    if node.children:
        fp.write(b"\x00" * 13)
    end = fp.tell()
    fp.seek(start)
    fp.write(struct.pack("<IIIB", end, len(node.props), len(body), len(name)))
    fp.write(name)
    fp.seek(end)

def write_fbx(path: Path, root: list[Node]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        fp.write(_HEAD_MAGIC)
        fp.write(struct.pack("<I", FBX_VERSION))
        for node in root:
            _write_node(fp, node)
        fp.write(b"\x00" * 13)

        fp.write(_FOOT_ID)
        fp.write(b"\x00" * 4)
        offset = fp.tell()
        pad = ((offset + 15) & ~15) - offset
        fp.write(b"\x00" * (pad if pad else 16))
        fp.write(struct.pack("<I", FBX_VERSION))
        fp.write(b"\x00" * 120)
        fp.write(_FOOT_MAGIC)
    return path

@dataclass
class _MaterialIds:
    material_id: int
    textures: list[int] = field(default_factory=list)

class FbxBuilder:
    def __init__(self, output_file: Path, textures: TextureStore, options: ExportOptions):
        self.output_file = output_file
        self.base_dir = output_file.parent
        self.textures = textures
        self.options = options
        self.objects: list[Node] = []
        self.connections: list[Node] = []
        self._materials: dict[tuple, _MaterialIds] = {}
        self._used_models: set[str] = set()
        self._next_id = 1_000_000
        self._counts = {"Model": 0, "Geometry": 0, "Material": 0, "Texture": 0}

    def _id(self) -> Long:
        value = Long(self._next_id)
        self._next_id += 1
        return value

    def _connect(self, kind: str, child: int, parent: int, prop: str | None = None) -> None:
        props: list[Any] = [kind, Long(child), Long(parent)]
        if prop is not None:
            props.append(prop)
        self.connections.append(Node("C", props))

    def _add_texture(self, asset: TextureAsset) -> int | None:
        rel = texture_relpath(asset.file, self.base_dir)
        name = sanitize_filename(asset.file.stem, "texture", 60)
        texture_id = self._id()

        texture = Node("Texture", [texture_id, f"{name}\x00\x01Texture", ""])
        texture.add("Type", "TextureVideoClip")
        texture.add("Version", 202)
        texture.add("TextureName", f"Texture::{name}")
        texture.add("FileName", rel)
        texture.add("RelativeFilename", rel)
        texture.add("ModelUVTranslation", Array("d", [0.0, 0.0]))
        texture.add("ModelUVScaling", Array("d", [1.0, 1.0]))
        texture.add("Texture_Alpha_Source", "None")
        texture.add("Cropping", Array("d", [0.0, 0.0, 0.0, 0.0]))
        self.objects.append(texture)
        self._counts["Texture"] += 1
        return texture_id

    def add_material(self, material: MaterialData) -> _MaterialIds:
        key = material_key(material)
        cached = self._materials.get(key)
        if cached is not None:
            return cached

        mat_id = self._id()
        name = sanitize_filename(material.name or "material", "material", 60)
        r, g, b = srgb_tuple(material.base_color)
        alpha = 1.0 if material.alpha_mode == "OPAQUE" else material.base_color[3]
        er, eg, eb = srgb_tuple(material.emission_color)

        node = Node("Material", [mat_id, f"{name}\x00\x01Material", ""])
        node.add("Version", 102)
        node.add("ShadingModel", "phong")
        node.add("MultiLayer", 0)
        props = node.child("Properties70")
        props.add("P", "DiffuseColor", "Color", "", "A", float(r), float(g), float(b))
        props.add("P", "DiffuseFactor", "Number", "", "A", 1.0)
        props.add("P", "TransparencyFactor", "Number", "", "A", float(1.0 - alpha))
        props.add("P", "Opacity", "Number", "", "A", float(alpha))
        props.add("P", "EmissiveColor", "Color", "", "A", float(er), float(eg), float(eb))
        props.add("P", "AmbientColor", "Color", "", "A", 0.0, 0.0, 0.0)
        props.add("P", "SpecularColor", "Color", "", "A", 1.0, 1.0, 1.0)
        props.add("P", "Shininess", "Number", "", "A", float(max(0.0, min(1.0, 1.0 - material.roughness)) * 100.0))
        self.objects.append(node)
        self._counts["Material"] += 1

        ids = _MaterialIds(mat_id)
        if self.textures.enabled:
            for binding, role, link in (
                (material.albedo, "albedo", "DiffuseColor"),
                (material.normal, "normal", "NormalMap"),
                (material.emission_map, "emission", "EmissiveColor"),
            ):
                if binding is None or binding.texture is None:
                    continue
                hint = f"{name}_{role}"
                try:
                    asset = self.textures.save(binding.texture, hint=hint, role=role)
                except Exception as exc:
                    log.warning("texture %s failed: %s: %s", hint, type(exc).__name__, exc)
                    asset = None
                if asset is None:
                    continue
                texture_id = self._add_texture(asset)
                if texture_id is None:
                    continue
                ids.textures.append(texture_id)
                self._connect("OP", texture_id, mat_id, link)

        self._materials[key] = ids
        return ids

    def add_mesh(
        self,
        mesh: MeshData,
        materials: Sequence[MaterialData],
        name: str,
        world: Mat4 = IDENTITY,
    ) -> str | None:
        if not mesh.vertices or mesh.triangle_count == 0:
            return None
        reverse = needs_winding_reverse(det3(world), self.options.mirror_x)
        mirror = self.options.mirror_x
        scale = self.options.scale

        positions: list[tuple[float, float, float]] = []
        for vertex in mesh.vertices:
            p = transform_point(world, vertex)
            if mirror:
                p = (-p[0], p[1], p[2])
            positions.append((p[0] * scale, p[1] * scale, p[2] * scale))

        normals = usable_normals(mesh)
        if normals is None:
            normals = compute_normals(mesh.vertices, flatten_triangles(mesh))
        transformed_normals: list[tuple[float, float, float]] = []
        for normal in normals:
            n = transform_normal(world, normal)
            if mirror:
                n = (-n[0], n[1], n[2])
            transformed_normals.append(n)

        polygon_indices: list[int] = []
        polygon_normals: list[float] = []
        polygon_uvs: list[int] = []
        polygon_materials: list[int] = []

        for sub_index, triangles in enumerate(mesh.triangles):
            if not triangles:
                continue
            material_index = min(sub_index, len(materials) - 1) if materials else 0
            for tri in triangles:
                a, b, c = tri
                if reverse:
                    b, c = c, b
                polygon_indices.extend([a, b, -(c + 1)])
                polygon_normals.extend(transformed_normals[a])
                polygon_normals.extend(transformed_normals[b])
                polygon_normals.extend(transformed_normals[c])
                if mesh.uvs:
                    polygon_uvs.extend([a, b, c])
                polygon_materials.append(material_index)

        if not polygon_indices:
            return None

        geom_id = self._id()
        model_id = self._id()
        model_name = sanitize_filename(name, "mesh", 70)
        base = model_name
        i = 2
        while model_name.lower() in self._used_models:
            model_name = f"{base}_{i}"
            i += 1
        self._used_models.add(model_name.lower())

        geometry = Node("Geometry", [geom_id, f"{model_name}\x00\x01Geometry", "Mesh"])
        geometry.add("Vertices", Array("d", [value for point in positions for value in point]))
        geometry.add("PolygonVertexIndex", Array("i", polygon_indices))
        geometry.add("GeometryVersion", 124)

        normal_layer = geometry.child("LayerElementNormal")
        normal_layer.add("Version", 101)
        normal_layer.add("Name", "")
        normal_layer.add("MappingInformationType", "ByPolygonVertex")
        normal_layer.add("ReferenceInformationType", "Direct")
        normal_layer.add("Normals", Array("d", polygon_normals))

        if mesh.uvs and polygon_uvs:
            uv_layer = geometry.child("LayerElementUV")
            uv_layer.add("Version", 101)
            uv_layer.add("Name", "UVMap")
            uv_layer.add("MappingInformationType", "ByPolygonVertex")
            uv_layer.add("ReferenceInformationType", "IndexToDirect")
            uv_layer.add("UV", Array("d", [value for uv in mesh.uvs for value in uv]))
            uv_layer.add("UVIndex", Array("i", polygon_uvs))

        if materials:
            mat_layer = geometry.child("LayerElementMaterial")
            mat_layer.add("Version", 101)
            mat_layer.add("Name", "")
            mat_layer.add("MappingInformationType", "ByPolygon")
            mat_layer.add("ReferenceInformationType", "IndexToDirect")
            mat_layer.add("Materials", Array("i", polygon_materials))

        layer = geometry.child("Layer")
        layer.add("Version", 100)
        layer_types = ["LayerElementNormal"]
        if mesh.uvs and polygon_uvs:
            layer_types.append("LayerElementUV")
        if materials:
            layer_types.append("LayerElementMaterial")
        for layer_type in layer_types:
            item = layer.child("LayerElement")
            item.add("Type", layer_type)
            item.add("TypedIndex", 0)

        self.objects.append(geometry)
        self._counts["Geometry"] += 1

        model = Node("Model", [model_id, f"{model_name}\x00\x01Model", "Mesh"])
        model.add("Version", 232)
        model_props = model.child("Properties70")
        model_props.add("P", "Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0)
        model_props.add("P", "Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0)
        model_props.add("P", "Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0)
        model_props.add("P", "DefaultAttributeIndex", "int", "Integer", "", 0)
        model.add("Culling", "CullingOff")
        self.objects.append(model)
        self._counts["Model"] += 1

        self._connect("OO", geom_id, model_id)
        self._connect("OO", model_id, 0)
        for material in materials:
            self._connect("OO", self.add_material(material).material_id, model_id)
        return model_name

    def write(self) -> Path:
        header = Node("FBXHeaderExtension")
        header.add("FBXHeaderVersion", 1003)
        header.add("FBXVersion", FBX_VERSION)
        stamp = header.child("CreationTimeStamp")
        for key, value in (
            ("Version", 1000),
            ("Year", 2026),
            ("Month", 11),
            ("Day", 15),
            ("Hour", 0),
            ("Minute", 0),
            ("Second", 0),
            ("Millisecond", 0),
        ):
            stamp.add(key, value)
        header.add("Creator", "UnityModelRipper")

        settings = Node("GlobalSettings")
        settings.add("Version", 1000)
        props = settings.child("Properties70")
        for name, value in (
            ("UpAxis", 1),
            ("UpAxisSign", 1),
            ("FrontAxis", 2),
            ("FrontAxisSign", 1),
            ("CoordAxis", 0),
            ("CoordAxisSign", 1),
        ):
            props.add("P", name, "int", "Integer", "", value)
        props.add("P", "UnitScaleFactor", "double", "Number", "", 100.0)

        documents = Node("Documents")
        documents.add("Count", 1)
        document = documents.child("Document")
        document.props = [1_000_000_000, "Scene", "Scene"]
        doc_props = document.child("Properties70")
        doc_props.add("P", "SourceObject", "object", "", "")
        doc_props.add("P", "ActiveAnimStackName", "KString", "", "", "")
        document.add("RootNode", 0)

        references = Node("References")

        definitions = Node("Definitions")
        definitions.add("Version", 100)
        definitions.add("Count", 1 + sum(self._counts.values()))
        global_type = definitions.child("ObjectType")
        global_type.props = ["GlobalSettings"]
        global_type.add("Count", 1)
        for kind in ("Model", "Geometry", "Material", "Texture"):
            count = self._counts[kind]
            if not count:
                continue
            object_type = definitions.child("ObjectType")
            object_type.props = [kind]
            object_type.add("Count", count)

        objects = Node("Objects")
        objects.children = list(self.objects)

        connections = Node("Connections")
        connections.children = list(self.connections)

        takes = Node("Takes")
        takes.add("Current", "")

        root = [header, settings, documents, references, definitions, objects, connections, takes]
        write_fbx(self.output_file, root)
        log.info("FBX %s (%s)", self.output_file.name, format_size(self.output_file.stat().st_size))
        return self.output_file

def export_fbx_assets(
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
            builder = FbxBuilder(directory / f"{candidate}.fbx", textures, options)
            matrix = asset.matrix if asset.matrix != IDENTITY else scale_matrix(asset.scale)
            if builder.add_mesh(asset.mesh, asset.materials, asset.name, matrix) is None:
                continue
            written.append((asset.name, builder.write()))
        except Exception as exc:
            log.warning("FBX export failed for %s: %s: %s", asset.name, type(exc).__name__, exc)
    return written

def export_fbx_scene(
    path: Path,
    models: Iterable[RendererModel],
    textures: TextureStore,
    options: ExportOptions,
    scene_name: str = "scene",
) -> Path | None:
    builder = FbxBuilder(path, textures, options)
    exported = 0
    for model in models:
        if model.mesh is None:
            continue
        if builder.add_mesh(model.mesh, model.materials, model.path or model.name, model.world) is not None:
            exported += 1
    if not exported:
        log.warning("FBX scene skipped: no geometry")
        return None
    return builder.write()

def export_fbx_scene_split(
    directory: Path,
    models: Iterable[RendererModel],
    textures: TextureStore,
    options: ExportOptions,
) -> list[tuple[RendererModel, Path]]:
    directory.mkdir(parents=True, exist_ok=True)
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
        try:
            builder = FbxBuilder(directory / f"{candidate}.fbx", textures, options)
            if builder.add_mesh(model.mesh, model.materials, model.path or model.name, model.world) is None:
                continue
            written.append((model, builder.write()))
        except Exception as exc:
            log.warning("FBX export failed for %s: %s: %s", model.path, type(exc).__name__, exc)
    log.info("FBX scene split into %d file(s)", len(written))
    return written
