from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler

from .discovery import InputSet
from .materials import MaterialData, read_material
from .mathutil import Mat4, Vec2, Vec3, Vec4, IDENTITY, mat_mul, matrix_from_unity, trs_matrix
from .util import asset_file_of, log, object_identity, object_path_id, sanitize_filename, unique_name

RENDERER_TYPES = {"MeshRenderer", "SkinnedMeshRenderer"}
MESH_TOPOLOGY_TRIANGLES = 0
MESH_TOPOLOGY_TRIANGLE_STRIP = 1
MESH_TOPOLOGY_QUADS = 2

@dataclass
class MeshData:
    key: tuple[str, int]
    name: str
    source: str
    vertices: list[Vec3]
    uvs: list[Vec2] | None
    normals: list[Vec3] | None
    colors: list[Vec4] | None
    triangles: list[list[tuple[int, int, int]]]
    submesh_count: int
    vertex_count: int
    triangle_count: int
    joints: list[tuple[int, int, int, int]] | None = None
    weights: list[tuple[float, float, float, float]] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.vertices) and self.triangle_count > 0

@dataclass
class MeshAsset:
    mesh: MeshData
    name: str
    materials: list[MaterialData]
    instances: int = 1
    source: str = ""
    scale: float = 1.0
    matrix: Mat4 = IDENTITY

@dataclass
class _AssetGroup:

    mesh: MeshData
    materials: list[MaterialData]
    instances: int = 0
    worlds: list[Mat4] = field(default_factory=list)

@dataclass
class RendererModel:
    obj: Any
    gameobject: Any
    path: str
    world: Mat4
    mesh: MeshData | None
    materials: list[MaterialData]
    is_skinned: bool
    bones: list[Any] = field(default_factory=list)
    bindposes: list[Mat4] = field(default_factory=list)
    enabled: bool = True
    active: bool = True
    source: str = ""

    @property
    def name(self) -> str:
        if self.mesh is not None and not self.is_skinned:
            return self.mesh.name
        return self.path.split("/")[-1] if self.path else self.mesh.name if self.mesh else "model"

def _reader_key(obj: Any, base: str | None = None) -> tuple[str, int]:
    reader = getattr(obj, "object_reader", None)
    target = reader if reader is not None else obj
    return object_identity(target, base)

def _alive(ptr: Any) -> bool:
    if ptr is None:
        return False
    try:
        return object_path_id(ptr) != 0
    except Exception:
        return False

def _read(ptr: Any) -> Any:
    return ptr.read() if _alive(ptr) else None

def _vec(value: Any, size: int) -> tuple[float, ...]:
    if value is None:
        return tuple(0.0 for _ in range(size))
    try:
        if size == 2:
            return (float(value.x), float(value.y))
        if size == 3:
            return (float(value.x), float(value.y), float(value.z))
        if size == 4:
            return (float(value.x), float(value.y), float(value.z), float(value.w))
    except Exception:
        pass
    try:
        return tuple(float(value[i]) for i in range(size))
    except Exception:
        return tuple(0.0 for _ in range(size))

def load_environment(input_set: InputSet, decrypt_key: str | None = None) -> UnityPy.Environment:
    if decrypt_key:
        try:
            UnityPy.set_assetbundle_decrypt_key(bytes.fromhex(decrypt_key))
        except ValueError as exc:
            raise ValueError(f"invalid --key (expected hex): {exc}") from exc

    env = UnityPy.Environment()
    roots = [*input_set.data_dirs, *input_set.game_roots]
    if roots:
        env.path = str(roots[0])
    elif input_set.files:
        env.path = str(input_set.files[0].parent)

    errors: list[str] = []
    for path in input_set.files:
        try:
            env.load_file(str(path))
        except Exception as exc:
            message = f"{path.name}: {type(exc).__name__}: {exc}"
            errors.append(message)
            log.warning("failed to load %s", message)
    for archive in input_set.archives:
        try:
            env.load_zip_file(str(archive))
        except Exception as exc:
            message = f"{archive.name}: {type(exc).__name__}: {exc}"
            errors.append(message)
            log.warning("failed to load %s", message)

    if not env.files:
        detail = "; ".join(errors[:3]) if errors else "no Unity data"
        raise RuntimeError(f"nothing could be loaded ({detail})")

    input_set.notes.extend(f"load error: {e}" for e in errors)
    log.info("loaded %d file(s), %d object(s)", len(env.files), len(env.objects))
    return env

class UnityScene:
    def __init__(self, env: UnityPy.Environment, sources: Sequence[str] = ()):
        self.env = env
        self.sources = list(sources)
        self.renderers: list[RendererModel] = []
        self.meshes: dict[tuple[str, int], MeshData] = {}
        self.materials: dict[tuple[str, int], MaterialData] = {}
        self.issues: list[dict[str, str]] = []
        self._world_cache: dict[tuple[str, int], Mat4] = {}
        self._path_cache: dict[tuple[str, int], str] = {}
        self._active_cache: dict[tuple[str, int], bool] = {}
        self._mesh_by_key: dict[tuple[str, int], Any] = {}
        self._asset_groups: dict[tuple[str, int], _AssetGroup] = {}
        self._assets_scanned = False
        self._object_counts: Counter[str] = Counter()

    def _issue(self, message: str, obj: Any = None) -> None:
        entry = {"message": message}
        if obj is not None:
            entry["object"] = f"{obj.type.name}#{getattr(obj, 'path_id', 0)}"
        self.issues.append(entry)
        log.warning(message)

    def _source_of(self, obj: Any) -> str:
        assets_file = asset_file_of(obj)
        name = getattr(assets_file, "name", "") if assets_file is not None else ""
        return Path(str(name)).name if name else ""

    @staticmethod
    def _resolve_transform(transform: Any) -> Any:
        if transform is None:
            return None
        if not hasattr(transform, "m_LocalScale") and hasattr(transform, "read"):
            try:
                return transform.read()
            except Exception:
                return None
        return transform

    def _transform_of(self, gameobject: Any) -> Any:
        return self._resolve_transform(getattr(gameobject, "m_Transform", None))

    def _local_matrix(self, transform: Any) -> Mat4:
        position = _vec(getattr(transform, "m_LocalPosition", None), 3)
        rotation = _vec(getattr(transform, "m_LocalRotation", None), 4)
        scale = _vec(getattr(transform, "m_LocalScale", None), 3)
        return trs_matrix(position, rotation, scale)

    def world_matrix(self, transform: Any, _depth: int = 0) -> Mat4:
        if transform is None:
            return IDENTITY
        key = _reader_key(transform, getattr(self.env, "path", None))
        cached = self._world_cache.get(key)
        if cached is not None:
            return cached
        if _depth > 512:
            self._issue("transform hierarchy deeper than 512, truncating")
            return self._local_matrix(transform)
        local = self._local_matrix(transform)
        father_ptr = getattr(transform, "m_Father", None)
        if _alive(father_ptr):
            father = _read(father_ptr)
            world = mat_mul(self.world_matrix(father, _depth + 1), local)
        else:
            world = local
        self._world_cache[key] = world
        return world

    def node_path(self, gameobject: Any) -> str:
        transform = self._transform_of(gameobject)
        if transform is None:
            return str(getattr(gameobject, "m_Name", "object") or "object")
        key = _reader_key(transform, getattr(self.env, "path", None))
        cached = self._path_cache.get(key)
        if cached is not None:
            return cached
        names: list[str] = []
        current = transform
        for _ in range(512):
            go = _read(getattr(current, "m_GameObject", None))
            names.append(str(getattr(go, "m_Name", "") or "object"))
            father_ptr = getattr(current, "m_Father", None)
            if not _alive(father_ptr):
                break
            current = _read(father_ptr)
            if current is None:
                break
        path = "/".join(reversed(names))
        self._path_cache[key] = path
        return path

    def is_active(self, gameobject: Any) -> bool:
        transform = self._transform_of(gameobject)
        if transform is None:
            return bool(getattr(gameobject, "m_IsActive", 1))
        key = _reader_key(transform, getattr(self.env, "path", None))
        if key in self._active_cache:
            return self._active_cache[key]
        active = True
        current_go = gameobject
        current_tr = transform
        for _ in range(512):
            if not bool(getattr(current_go, "m_IsActive", 1)):
                active = False
                break
            father_ptr = getattr(current_tr, "m_Father", None)
            if not _alive(father_ptr):
                break
            father = _read(father_ptr)
            if father is None:
                break
            current_tr = father
            current_go = _read(getattr(father, "m_GameObject", None))
            if current_go is None:
                break
        self._active_cache[key] = active
        return active

    def get_mesh(self, mesh_ptr: Any) -> MeshData | None:
        if not _alive(mesh_ptr):
            return None
        key = _reader_key(mesh_ptr, getattr(self.env, "path", None))
        cached = self.meshes.get(key)
        if cached is not None:
            return cached
        try:
            mesh = mesh_ptr.read()
        except Exception as exc:
            self._issue(f"mesh #{key[1]} could not be parsed: {type(exc).__name__}: {exc}", mesh_ptr)
            return None

        name = str(getattr(mesh, "m_Name", "") or f"mesh_{key[1]}")
        source = self._source_of(getattr(mesh, "object_reader", mesh_ptr))
        error: str | None = None
        vertices: list[Vec3] = []
        uvs: list[Vec2] | None = None
        normals: list[Vec3] | None = None
        colors: list[Vec4] | None = None
        joints: list[tuple[int, int, int, int]] | None = None
        weights: list[tuple[float, float, float, float]] | None = None
        triangles: list[list[tuple[int, int, int]]] = []
        submesh_count = 0

        try:
            handler = MeshHandler(mesh)
            handler.process()
            raw_vertices = handler.m_Vertices or []
            vertices = [_vec(v, 3) for v in raw_vertices]
            if handler.m_UV0:
                uvs = [_vec(v, 2) for v in handler.m_UV0]
            if handler.m_Normals:
                normals = [_vec(v, 3) for v in handler.m_Normals]
            if handler.m_Colors:
                colors = [_vec(v, 4) for v in handler.m_Colors]
            if handler.m_BoneIndices and handler.m_BoneWeights:
                joints = [tuple(int(x) for x in v[:4]) for v in handler.m_BoneIndices]
                weights = [tuple(float(x) for x in v[:4]) for v in handler.m_BoneWeights]
            submeshes = list(getattr(mesh, "m_SubMeshes", None) or [])
            submesh_count = len(submeshes)
            triangles = self._extract_triangles(handler, submeshes, len(vertices))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._issue(f"mesh '{name}' failed: {error}", mesh_ptr)

        if not vertices:
            raw = getattr(mesh, "m_Vertices", None) or []
            vertices = [_vec(v, 3) for v in raw]
        if normals is not None and len(normals) < len(vertices):
            normals = None
        if uvs is not None and len(uvs) < len(vertices):
            uvs = None
        if colors is not None and len(colors) < len(vertices):
            colors = None
        if joints is not None and len(joints) < len(vertices):
            joints = None
        if weights is not None and len(weights) < len(vertices):
            weights = None
        if joints is None or weights is None:
            joints = weights = None

        triangle_count = sum(len(t) for t in triangles)
        data = MeshData(
            key=key,
            name=name,
            source=source,
            vertices=vertices,
            uvs=uvs,
            normals=normals,
            colors=colors,
            triangles=triangles,
            submesh_count=submesh_count or 1,
            vertex_count=len(vertices),
            triangle_count=triangle_count,
            joints=joints,
            weights=weights,
            error=error,
        )
        self.meshes[key] = data
        log.debug("mesh %s: %d verts, %d tris, %d submeshes", name, len(vertices), triangle_count, data.submesh_count)
        return data

    @staticmethod
    def _extract_triangles(handler: MeshHandler, submeshes: list[Any], vertex_count: int) -> list[list[tuple[int, int, int]]]:
        if not submeshes:
            indices = handler.m_IndexBuffer
            if not indices:
                return []
            return [
                (indices[i], indices[i + 1], indices[i + 2])
                for i in range(0, len(indices) - 2, 3)
                if all(0 <= indices[i + k] < vertex_count for k in range(3))
            ]

        index_buffer = handler.m_IndexBuffer
        result: list[list[tuple[int, int, int]]] = []
        for submesh in submeshes:
            if not index_buffer:
                result.append([])
                continue
            first = int(getattr(submesh, "firstByte", 0)) // 2
            if not handler.m_Use16BitIndices:
                first //= 2
            count = int(getattr(submesh, "indexCount", 0))
            topology = int(getattr(submesh, "topology", 0))
            indices = index_buffer[first : first + count]
            tris: list[tuple[int, int, int]] = []
            if topology == MESH_TOPOLOGY_TRIANGLES:
                for i in range(0, len(indices) - 2, 3):
                    a, b, c = indices[i], indices[i + 1], indices[i + 2]
                    if a != b and b != c and a != c and max(a, b, c) < vertex_count:
                        tris.append((a, b, c))
            elif topology == MESH_TOPOLOGY_TRIANGLE_STRIP:
                for i in range(len(indices) - 2):
                    a, b, c = indices[i], indices[i + 1], indices[i + 2]
                    if a == b or b == c or a == c or max(a, b, c) >= vertex_count:
                        continue
                    tris.append((b, a, c) if i & 1 else (a, b, c))
            elif topology == MESH_TOPOLOGY_QUADS:
                for i in range(0, len(indices) - 3, 4):
                    a, b, c, d = indices[i], indices[i + 1], indices[i + 2], indices[i + 3]
                    if max(a, b, c, d) < vertex_count:
                        tris.append((a, b, c))
                        tris.append((a, c, d))
            result.append(tris)
        return result

    def get_material(self, material_ptr: Any) -> MaterialData:
        if not _alive(material_ptr):
            return MaterialData(key=None, name="Default", shader="")
        key = _reader_key(material_ptr, getattr(self.env, "path", None))
        cached = self.materials.get(key)
        if cached is not None:
            return cached
        try:
            data = read_material(material_ptr, getattr(self.env, "path", None))
        except Exception as exc:
            name = f"material_{key[1]}"
            self._issue(f"material #{key[1]} failed: {type(exc).__name__}: {exc}", material_ptr)
            data = MaterialData(key=key, name=name)
        self.materials[key] = data
        return data

    def _materials_of(self, renderer: Any) -> list[MaterialData]:
        result = []
        for ptr in getattr(renderer, "m_Materials", None) or []:
            result.append(self.get_material(ptr))
        return result or [MaterialData(key=None, name="Default")]

    def _mesh_filter_of(self, gameobject: Any) -> Any:
        for pair in getattr(gameobject, "m_Component", None) or []:
            component_ptr = getattr(pair, "component", None)
            if not _alive(component_ptr):
                continue
            try:
                if int(getattr(component_ptr, "type", 0)) == 33:
                    return _read(component_ptr)
            except Exception:
                continue
        return None

    def collect(self, active_only: bool = False) -> None:
        renderers: list[RendererModel] = []
        for obj in self.env.objects:
            type_name = obj.type.name
            self._object_counts[type_name] += 1
            if type_name not in RENDERER_TYPES:
                continue
            try:
                renderer = obj.read()
                gameobject = _read(getattr(renderer, "m_GameObject", None))
                if gameobject is None:
                    continue
                transform = self._transform_of(gameobject)
                if transform is None:
                    continue

                is_skinned = type_name == "SkinnedMeshRenderer"
                if is_skinned:
                    mesh_ptr = getattr(renderer, "m_Mesh", None)
                else:
                    mesh_filter = self._mesh_filter_of(gameobject)
                    mesh_ptr = getattr(mesh_filter, "m_Mesh", None)

                mesh = self.get_mesh(mesh_ptr) if _alive(mesh_ptr) else None
                if mesh is None:
                    continue

                active = self.is_active(gameobject)
                enabled = bool(getattr(renderer, "m_Enabled", 1))
                if active_only and not (active and enabled):
                    continue

                bones: list[Any] = []
                bindposes: list[Mat4] = []
                if is_skinned:
                    for bone_ptr in getattr(renderer, "m_Bones", None) or []:
                        bone = _read(bone_ptr)
                        if bone is not None:
                            bones.append(bone)
                    mesh_obj = mesh_ptr.read()
                    for matrix in getattr(mesh_obj, "m_BindPose", None) or []:
                        try:
                            bindposes.append(matrix_from_unity(matrix))
                        except Exception:
                            break
                    if len(bindposes) != len(bones):
                        bindposes = []

                model = RendererModel(
                    obj=obj,
                    gameobject=gameobject,
                    path=self.node_path(gameobject),
                    world=self.world_matrix(transform),
                    mesh=mesh,
                    materials=self._materials_of(renderer),
                    is_skinned=is_skinned,
                    bones=bones,
                    bindposes=bindposes,
                    enabled=enabled,
                    active=active,
                    source=self._source_of(obj),
                )
                renderers.append(model)
            except Exception as exc:
                self._issue(f"renderer #{getattr(obj, 'path_id', 0)} failed: {type(exc).__name__}: {exc}", obj)

        renderers.sort(key=lambda r: (r.path.lower(), object_path_id(r.obj)))
        self.renderers = renderers
        log.info(
            "collected %d renderer(s), %d unique mesh(es), %d material(s)",
            len(renderers), len(self.meshes), len(self.materials),
        )

    def scan_assets(self, active_only: bool = False) -> None:
        if self._assets_scanned:
            return
        self._assets_scanned = True
        seen_renderers = 0
        started = time.monotonic()
        for obj in self.env.objects:
            type_name = obj.type.name
            self._object_counts[type_name] += 1
            if type_name not in RENDERER_TYPES:
                continue
            seen_renderers += 1
            if seen_renderers % 25000 == 0:
                log.info(
                    "asset scan: %d renderer(s), %.1f s, %d unique mesh(es)",
                    seen_renderers, time.monotonic() - started, len(self._asset_groups),
                )
            try:
                renderer = obj.read()
                gameobject = _read(getattr(renderer, "m_GameObject", None))
                if gameobject is None:
                    continue
                transform = self._transform_of(gameobject)
                if transform is None:
                    continue
                if active_only and (
                    not self.is_active(gameobject)
                    or not bool(getattr(renderer, "m_Enabled", 1))
                ):
                    continue

                if type_name == "SkinnedMeshRenderer":
                    mesh_ptr = getattr(renderer, "m_Mesh", None)
                else:
                    mesh_filter = self._mesh_filter_of(gameobject)
                    mesh_ptr = getattr(mesh_filter, "m_Mesh", None)

                mesh = self.get_mesh(mesh_ptr) if _alive(mesh_ptr) else None
                if mesh is None:
                    continue
                group = self._mesh_group_key(mesh)
                world = self.world_matrix(transform)
                entry = self._asset_groups.get(group)
                if entry is None:
                    entry = _AssetGroup(
                        mesh=mesh,
                        materials=self._materials_of(renderer),
                        worlds=[world],
                    )
                    self._asset_groups[group] = entry
                else:
                    entry.worlds.append(world)
                entry.instances += 1
            except Exception as exc:
                log.warning(
                    "asset scan failed for %s: %s: %s",
                    type_name, type(exc).__name__, exc,
                )
        log.info(
            "asset scan complete: %d renderer(s), %d unique mesh(es), %.1f s",
            seen_renderers, len(self._asset_groups), time.monotonic() - started,
        )

    @staticmethod
    def _mesh_group_key(mesh: MeshData) -> tuple[str, int]:
        path_id = mesh.key[1]
        return (mesh.name.lower(), path_id) if path_id else (mesh.name.lower(), id(mesh))

    @staticmethod
    def _matrix_scale(m: Mat4) -> float | None:
        sx = math.hypot(m[0], m[4], m[8])
        sy = math.hypot(m[1], m[5], m[9])
        sz = math.hypot(m[2], m[6], m[10])
        if not all(math.isfinite(v) and v > 1e-8 for v in (sx, sy, sz)):
            return None
        return (sx * sy * sz) ** (1.0 / 3.0)

    @staticmethod
    def _without_translation(m: Mat4) -> Mat4:
        return (
            m[0], m[1], m[2], 0.0,
            m[4], m[5], m[6], 0.0,
            m[8], m[9], m[10], 0.0,
            0.0, 0.0, 0.0, 1.0,
        )

    @classmethod
    def _representative_matrix(cls, worlds: Sequence[Mat4]) -> tuple[Mat4, float]:
        ranked = sorted(
            (scale, index)
            for index, world in enumerate(worlds)
            if (scale := cls._matrix_scale(world)) is not None
        )
        if not ranked:
            return IDENTITY, 1.0
        median_scale, index = ranked[len(ranked) // 2]
        return cls._without_translation(worlds[index]), median_scale

    @classmethod
    def _asset_transform(cls, instances: Sequence[RendererModel]) -> tuple[Mat4, float]:
        return cls._representative_matrix([model.world for model in instances])

    def mesh_assets(
        self,
        all_meshes: bool = False,
        active_only: bool = False,
        fast: bool = False,
    ) -> list[MeshAsset]:
        used_names: set[str] = set()
        assets: list[MeshAsset] = []

        def add(
            mesh: MeshData | None,
            materials: list[MaterialData],
            instances: int,
            matrix: Mat4,
            scale: float,
            group: tuple[str, int],
        ) -> None:
            if mesh is None or not mesh.ok:
                return
            stem = sanitize_filename(mesh.name or "mesh")
            name = unique_name(stem, used_names, suffix=str(group[1]) if group[1] else "")
            assets.append(
                MeshAsset(
                    mesh=mesh,
                    name=name,
                    materials=materials,
                    instances=instances,
                    source=mesh.source,
                    scale=scale,
                    matrix=matrix,
                )
            )

        if fast:
            self.scan_assets(active_only)
            for group, entry in self._asset_groups.items():
                matrix, scale = self._representative_matrix(entry.worlds)
                add(
                    entry.mesh,
                    entry.materials,
                    entry.instances,
                    matrix,
                    scale,
                    group,
                )
            if all_meshes:
                known = set(self._asset_groups)
                for obj in self.env.objects:
                    if obj.type.name != "Mesh":
                        continue
                    data = self.get_mesh(obj)
                    if data is None:
                        continue
                    group = self._mesh_group_key(data)
                    if group in known:
                        continue
                    known.add(group)
                    add(data, [MaterialData(key=None, name="Default")], 0, IDENTITY, 1.0, group)
        else:
            by_group: dict[tuple[str, int], list[RendererModel]] = defaultdict(list)
            mesh_by_group: dict[tuple[str, int], MeshData] = {}
            for model in self.renderers:
                if model.mesh is None:
                    continue
                group = self._mesh_group_key(model.mesh)
                by_group[group].append(model)
                mesh_by_group.setdefault(group, model.mesh)

            if all_meshes:
                for obj in self.env.objects:
                    if obj.type.name != "Mesh":
                        continue
                    data = self.get_mesh(obj)
                    if data is None:
                        continue
                    group = self._mesh_group_key(data)
                    by_group.setdefault(group, [])
                    mesh_by_group.setdefault(group, data)

            for group, instances in by_group.items():
                mesh = mesh_by_group.get(group)
                first = instances[0] if instances else None
                materials = first.materials if first else [MaterialData(key=None, name="Default")]
                matrix, scale = self._asset_transform(instances)
                add(
                    mesh,
                    materials,
                    len(instances),
                    matrix,
                    scale,
                    group,
                )

        assets.sort(key=lambda a: a.name.lower())
        return assets

    def scene_models(self, active_only: bool = False) -> list[RendererModel]:
        return [
            r for r in self.renderers
            if r.mesh is not None and r.mesh.ok and (not active_only or (r.active and r.enabled))
        ]

    def object_counts(self) -> dict[str, int]:
        return dict(self._object_counts)

    def unity_versions(self) -> list[str]:
        versions: set[str] = set()
        for assets_file in self.env.assets:
            version = getattr(assets_file, "version", None)
            if version:
                versions.add(".".join(str(v) for v in version[:3]))
        return sorted(versions)
