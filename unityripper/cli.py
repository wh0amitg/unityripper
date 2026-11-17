from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .discovery import InputSet, collect
from .exporters.common import ExportOptions
from .exporters.glb import export_glb_assets, export_glb_assets_split, export_glb_scene
from .exporters.obj import ObjExporter
from .materials import MaterialData
from .scene import MeshAsset, RENDERER_TYPES, RendererModel, UnityScene, load_environment
from .textures import TextureStore, texture_format_name
from .util import asset_file_of, format_size, log, object_identity, object_path_id, plural, sanitize_filename

def _add_common_load_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("inputs", nargs="+", help="game folder, *_Data folder, .assets file, bundle, .apk or .zip")
    parser.add_argument("--active-only", action="store_true", help="skip inactive GameObjects and disabled renderers")
    parser.add_argument("--key", default=None, help="hex AES key for encrypted AssetBundles")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="more logging (-vv for debug)")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unityripper",
        description="Extract 3D models, materials and textures from Unity games (Mono and IL2CPP builds).",
    )
    parser.add_argument("--version", action="version", version=f"unityripper {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="export models to OBJ / GLB / FBX")
    _add_common_load_options(extract)
    extract.add_argument("-o", "--output", required=True, help="output directory")
    extract.add_argument(
        "--format",
        default="obj,glb",
        help="comma separated list: obj, glb, fbx (default: obj,glb)",
    )
    extract.add_argument(
        "--mode",
        choices=("assets", "scene"),
        default="assets",
        help="assets: one model per unique mesh at the origin; scene: every renderer with transforms and skinning",
    )
    extract.add_argument("--all-meshes", action="store_true", help="also export meshes that no renderer references")
    extract.add_argument(
        "--split",
        action="store_true",
        help="scene mode: one OBJ and one FBX per renderer instead of merged scene files "
        "(assets mode already writes one file per model)",
    )
    extract.add_argument("--no-textures", action="store_true", help="skip texture decoding")
    extract.add_argument(
        "--all-textures",
        action="store_true",
        help="also export every Texture2D in the game, not only textures used by exported materials",
    )
    extract.add_argument(
        "--normal-maps",
        choices=("auto", "raw", "off"),
        default="auto",
        help="auto: rebuild DXT5nm normal maps; raw: save as stored; off: no normal maps",
    )
    extract.add_argument(
        "--metallic-roughness",
        choices=("auto", "off"),
        default="auto",
        help="repack Unity metallic/smoothness maps for glTF",
    )
    extract.add_argument("--keep-axes", action="store_true", help="do not mirror X (keep Unity's left-handed axes)")
    extract.add_argument("--scale", type=float, default=1.0, help="uniform export scale (default: 1.0)")
    extract.add_argument("--no-manifest", action="store_true", help="do not write manifest.json")
    extract.add_argument("--scene-name", default=None, help="name for scene mode output files")

    listing = sub.add_parser("list", help="list meshes, renderers and textures")
    _add_common_load_options(listing)
    listing.add_argument("--type", default=None, help="filter object types (Mesh, Texture2D, Material, ...)")
    listing.add_argument("--json", action="store_true", help="machine readable output")
    listing.add_argument("--limit", type=int, default=500, help="maximum rows (default: 500)")

    info = sub.add_parser("info", help="show files, Unity versions and object counts")
    _add_common_load_options(info)
    info.add_argument("--json", action="store_true", help="machine readable output")

    return parser

def _setup_logging(verbose: int) -> None:
    level = logging.WARNING
    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)

def _load(
    inputs: Sequence[str],
    key: str | None,
    active_only: bool,
    collect_scene: bool = True,
) -> tuple[InputSet, UnityScene]:
    input_set = collect(list(inputs))
    for note in input_set.notes:
        log.warning(note)
    if not input_set.all_sources:
        raise RuntimeError("no Unity assets found in the given paths")
    env = load_environment(input_set, decrypt_key=key)
    scene = UnityScene(env, [str(p) for p in input_set.all_sources])
    if collect_scene:
        scene.collect(active_only=active_only)
    return input_set, scene

def _unique_object_counts(scene: Any) -> Counter[str]:
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str, int]] = set()
    for obj in scene.env.objects:
        type_name = obj.type.name
        identity = (type_name, *object_identity(obj, getattr(scene.env, "path", None)))
        if identity in seen:
            continue
        seen.add(identity)
        counts[type_name] += 1
    return counts

def _unique_file_count(env: Any) -> int:
    base = getattr(env, "path", None)
    seen: set[str] = set()
    for entry in env.files:
        name = str(getattr(entry, "name", None) or entry)
        path = Path(name)
        if not path.is_absolute() and base:
            path = Path(str(base)) / path
        seen.add(os.path.normcase(os.path.normpath(str(path))))
    return len(seen)

def _cmd_info(args: argparse.Namespace) -> int:
    input_set, scene = _load(args.inputs, args.key, args.active_only, collect_scene=False)
    counts = _unique_object_counts(scene)
    renderers = counts.get("MeshRenderer", 0) + counts.get("SkinnedMeshRenderer", 0)
    data = {
        "tool": f"unityripper {__version__}",
        "inputs": [str(p) for p in input_set.all_sources],
        "data_dirs": [str(p) for p in input_set.data_dirs],
        "unity_versions": scene.unity_versions(),
        "files": _unique_file_count(scene.env),
        "objects": sum(counts.values()),
        "renderers": renderers,
        "meshes": counts.get("Mesh", 0),
        "textures": counts.get("Texture2D", 0),
        "materials": counts.get("Material", 0),
        "types": dict(counts.most_common()),
    }
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    print(f"unityripper {__version__}")
    print(f"inputs:        {len(data['inputs'])}")
    for path in data["inputs"][:20]:
        print(f"  {path}")
    if len(data["inputs"]) > 20:
        print(f"  ... and {len(data['inputs']) - 20} more")
    print(f"unity version: {', '.join(data['unity_versions']) or 'unknown'}")
    print(f"files:         {data['files']}")
    print(f"objects:       {data['objects']}")
    print(f"renderers:     {data['renderers']}")
    print(f"meshes:        {data['meshes']}")
    print(f"materials:     {data['materials']}")
    print(f"textures:      {data['textures']}")
    print("object types:")
    for name, count in counts.most_common(25):
        print(f"  {name:<24} {count}")
    return 0

def _cmd_list(args: argparse.Namespace) -> int:
    wanted = {t.strip() for t in args.type.split(",")} if args.type else None
    needs_scene = wanted is None or bool(wanted & RENDERER_TYPES)
    _, scene = _load(
        args.inputs,
        args.key,
        args.active_only,
        collect_scene=needs_scene,
    )
    rows: list[dict[str, Any]] = []
    for obj in scene.env.objects:
        type_name = obj.type.name
        if wanted and type_name not in wanted:
            continue
        if type_name == "Mesh":
            mesh = scene.get_mesh(obj)
            if mesh is None:
                continue
            rows.append(
                {
                    "type": "Mesh",
                    "name": mesh.name,
                    "vertices": mesh.vertex_count,
                    "triangles": mesh.triangle_count,
                    "submeshes": mesh.submesh_count,
                    "source": mesh.source,
                }
            )
        elif type_name in {"Texture2D", "Sprite"}:
            try:
                texture = obj.read()
                fmt_name = texture_format_name(getattr(texture, "m_TextureFormat", None))
                rows.append(
                    {
                        "type": type_name,
                        "name": str(getattr(texture, "m_Name", "") or ""),
                        "width": int(getattr(texture, "m_Width", 0) or 0),
                        "height": int(getattr(texture, "m_Height", 0) or 0),
                        "format": fmt_name,
                        "source": Path(str(getattr(asset_file_of(obj), "name", ""))).name,
                    }
                )
            except Exception:
                continue
        elif type_name in {"MeshRenderer", "SkinnedMeshRenderer"}:
            model = next(
                (
                    m
                    for m in scene.renderers
                    if m.obj.path_id == obj.path_id and asset_file_of(m.obj) is asset_file_of(obj)
                ),
                None,
            )
            if model is None:
                continue
            rows.append(
                {
                    "type": type_name,
                    "name": model.path,
                    "mesh": model.mesh.name if model.mesh else "",
                    "vertices": model.mesh.vertex_count if model.mesh else 0,
                    "triangles": model.mesh.triangle_count if model.mesh else 0,
                    "materials": [m.name for m in model.materials],
                    "skinned": model.is_skinned,
                    "active": model.active,
                    "source": model.source,
                }
            )
        elif wanted:
            name = ""
            try:
                name = str(getattr(obj.read(), "m_Name", "") or "")
            except Exception:
                name = ""
            rows.append(
                {
                    "type": type_name,
                    "name": name,
                    "source": Path(str(getattr(asset_file_of(obj), "name", ""))).name,
                }
            )

    rows = rows[: args.limit]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    if not rows:
        print("nothing found")
        return 1
    if wanted and wanted == {"Mesh"}:
        print(f"{'name':<44} {'verts':>8} {'tris':>8} {'sub':>4}  source")
        for row in rows:
            print(
                f"{row['name'][:44]:<44} {row['vertices']:>8} {row['triangles']:>8} "
                f"{row['submeshes']:>4}  {row['source']}"
            )
    elif wanted and wanted <= {"Texture2D", "Sprite"}:
        print(f"{'name':<44} {'size':>12} {'format':>16}  source")
        for row in rows:
            size = f"{row['width']}x{row['height']}"
            print(f"{row['name'][:44]:<44} {size:>12} {row['format']:>16}  {row['source']}")
    else:
        for row in rows:
            if row["type"] == "Mesh":
                print(f"Mesh {row['name']}  {row['vertices']}v {row['triangles']}t  [{row['source']}]")
            elif row["type"] in {"MeshRenderer", "SkinnedMeshRenderer"}:
                kind = "Skinned" if row["skinned"] else "Mesh"
                print(
                    f"{kind} {row['name']}  mesh={row['mesh']}  {row['vertices']}v {row['triangles']}t  "
                    f"materials={', '.join(row['materials'])}"
                )
            else:
                print(f"{row['type']} {row['name']}  {row.get('width', '')}x{row.get('height', '')} {row.get('format', '')}")
    return 0

def _asset_manifest(asset: MeshAsset, files: dict[str, str]) -> dict[str, Any]:
    return {
        "name": asset.name,
        "mesh": asset.mesh.name,
        "vertices": asset.mesh.vertex_count,
        "triangles": asset.mesh.triangle_count,
        "submeshes": asset.mesh.submesh_count,
        "instances": asset.instances,
        "scale": round(asset.scale, 6),
        "materials": [m.name for m in asset.materials],
        "source": asset.source,
        "files": files,
    }

def _scene_manifest(model: RendererModel, files: dict[str, str]) -> dict[str, Any]:
    return {
        "name": model.name,
        "path": model.path,
        "mesh": model.mesh.name if model.mesh else "",
        "vertices": model.mesh.vertex_count if model.mesh else 0,
        "triangles": model.mesh.triangle_count if model.mesh else 0,
        "skinned": model.is_skinned,
        "bones": len(model.bones),
        "materials": [m.name for m in model.materials],
        "active": model.active,
        "enabled": model.enabled,
        "source": model.source,
        "files": files,
    }

def _export_all_textures(scene: UnityScene, textures: TextureStore) -> int:
    exported = 0
    seen: set[tuple[str, int]] = set()
    for _ in range(4):
        pending = []
        for obj in scene.env.objects:
            if obj.type.name != "Texture2D":
                continue
            identity = object_identity(obj, getattr(scene.env, "path", None))
            if identity in seen:
                continue
            seen.add(identity)
            pending.append(obj)
        if not pending:
            break
        for obj in pending:
            path_id = object_path_id(obj)
            try:
                name = str(getattr(obj.read(), "m_Name", "") or f"texture_{path_id}")
            except Exception:
                name = f"texture_{path_id}"
            if textures.save(obj, hint=name, role="albedo") is not None:
                exported += 1
    log.info("exported %d texture object(s), %d PNG file(s)", exported, len(textures.assets))
    return exported

def _parse_formats(value: str) -> list[str]:
    formats = [part.strip().lower() for part in value.split(",") if part.strip()]
    valid = {"obj", "glb", "fbx"}
    unknown = [f for f in formats if f not in valid]
    if unknown:
        raise ValueError(f"unknown format(s): {', '.join(unknown)} (valid: obj, glb, fbx)")
    if not formats:
        raise ValueError("no export format selected")
    return formats

def _rel(path: Path, output: Path) -> str:
    return path.relative_to(output).as_posix()

def _cmd_extract(args: argparse.Namespace) -> int:
    formats = _parse_formats(args.format)
    input_set, scene = _load(
        args.inputs,
        args.key,
        args.active_only,
        collect_scene=args.mode != "assets",
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    options = ExportOptions(
        mirror_x=not args.keep_axes,
        scale=float(args.scale),
        textures=not args.no_textures,
        normal_mode=args.normal_maps,
        mr_mode=args.metallic_roughness,
        all_meshes=args.all_meshes,
        active_only=args.active_only,
    )
    textures = TextureStore(
        output,
        normal_mode=args.normal_maps,
        enabled=not args.no_textures,
        base_path=getattr(scene.env, "path", None),
    )

    scene_name = args.scene_name
    if not scene_name:
        if input_set.game_roots:
            scene_name = sanitize_filename(input_set.game_roots[0].name, "scene")
        elif input_set.data_dirs:
            scene_name = sanitize_filename(input_set.data_dirs[0].parent.name or "scene", "scene")
        else:
            scene_name = "scene"
    scene_stem = sanitize_filename(scene_name, "scene")

    exported_models: list[dict[str, Any]] = []
    written_files: list[str] = []
    renderer_count = 0

    if args.mode == "assets":
        assets = scene.mesh_assets(
            all_meshes=args.all_meshes,
            active_only=args.active_only,
            fast=True,
        )
        renderer_count = sum(asset.instances for asset in assets)
        if not assets:
            if not (args.all_textures and textures.enabled):
                log.error("no exportable meshes found")
                return 1
            log.warning("no exportable meshes found, exporting textures only")
        log.info("exporting %d unique mesh(es)", len(assets))
        exported_models = [_asset_manifest(asset, {}) for asset in assets]
        by_name = {model["name"]: model for model in exported_models}

        if assets and "obj" in formats:
            obj = ObjExporter(output, textures, options)
            for name, path in obj.export_assets(assets):
                rel = _rel(path, output)
                written_files.append(rel)
                model = by_name.get(name)
                if model is not None:
                    model["files"]["obj"] = rel
                    model["files"]["mtl"] = _rel(path.with_suffix(".mtl"), output)
        if assets and "glb" in formats:
            for name, path in export_glb_assets_split(output / "glb", assets, textures, options):
                rel = _rel(path, output)
                written_files.append(rel)
                model = by_name.get(name)
                if model is not None:
                    model["files"]["glb"] = rel
        if assets and "fbx" in formats:
            from .exporters.fbx import export_fbx_assets

            for name, path in export_fbx_assets(output / "fbx", assets, textures, options):
                rel = _rel(path, output)
                written_files.append(rel)
                model = by_name.get(name)
                if model is not None:
                    model["files"]["fbx"] = rel
    else:
        models = scene.scene_models(active_only=args.active_only)
        renderer_count = len(scene.renderers)
        if not models:
            if not (args.all_textures and textures.enabled):
                log.error("no exportable renderers found")
                return 1
            log.warning("no exportable renderers found, exporting textures only")
        log.info("exporting scene with %d renderer(s)", len(models))
        files: dict[str, str] = {}
        obj_files: dict[int, str] = {}
        fbx_files: dict[int, str] = {}

        if models and "obj" in formats:
            obj = ObjExporter(output, textures, options)
            if args.split:
                for model, path in obj.export_scene_split(models):
                    rel = _rel(path, output)
                    obj_files[id(model)] = rel
                    written_files.append(rel)
                    written_files.append(_rel(path.with_suffix(".mtl"), output))
            else:
                obj_path = obj.export_scene(models, name=scene_stem)
                rel = _rel(obj_path, output)
                written_files.append(rel)
                written_files.append(_rel(obj_path.with_suffix(".mtl"), output))
                for model in models:
                    obj_files[id(model)] = rel
        if models and "glb" in formats:
            glb_path = export_glb_scene(
                output / f"{scene_stem}.glb", models, textures, options, scene_name=scene_name
            )
            if glb_path is not None:
                files["glb"] = _rel(glb_path, output)
                written_files.append(files["glb"])
        if models and "fbx" in formats:
            from .exporters.fbx import export_fbx_scene, export_fbx_scene_split

            if args.split:
                for model, path in export_fbx_scene_split(output / "fbx", models, textures, options):
                    rel = _rel(path, output)
                    fbx_files[id(model)] = rel
                    written_files.append(rel)
            else:
                fbx_path = export_fbx_scene(
                    output / f"{scene_stem}.fbx", models, textures, options, scene_name=scene_name
                )
                if fbx_path is not None:
                    rel = _rel(fbx_path, output)
                    written_files.append(rel)
                    for model in models:
                        fbx_files[id(model)] = rel

        exported_models = []
        for model in models:
            model_files = dict(files)
            if id(model) in obj_files:
                model_files["obj"] = obj_files[id(model)]
                model_files["mtl"] = obj_files[id(model)].rsplit(".", 1)[0] + ".mtl"
            if id(model) in fbx_files:
                model_files["fbx"] = fbx_files[id(model)]
            exported_models.append(_scene_manifest(model, model_files))

    if args.all_textures and textures.enabled:
        _export_all_textures(scene, textures)

    if not args.no_manifest:
        type_counts = _unique_object_counts(scene)
        manifest = {
            "tool": f"unityripper {__version__}",
            "generated": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "unity_versions": scene.unity_versions(),
            "inputs": [str(p) for p in input_set.all_sources],
            "options": {
                "formats": formats,
                "mirror_x": options.mirror_x,
                "scale": options.scale,
                "textures": options.textures,
                "all_textures": bool(args.all_textures),
                "normal_maps": options.normal_mode,
            },
            "counts": {
                "renderers": renderer_count,
                "meshes": type_counts.get("Mesh", 0),
                "materials": type_counts.get("Material", 0),
                "textures": type_counts.get("Texture2D", 0),
                "exported_models": len(exported_models),
                "exported_textures": len(textures.assets),
            },
            "models": exported_models,
            "textures": [
                {
                    "name": t.name,
                    "file": str(t.file.relative_to(output)),
                    "width": t.width,
                    "height": t.height,
                    "format": t.format_name,
                }
                for t in textures.assets
            ],
            "materials": [
                {"name": m.name, "shader": m.shader, "alpha": m.alpha_mode}
                for m in scene.materials.values()
            ],
            "issues": [*input_set.notes, *scene.issues, *textures.errors],
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        written_files.append("manifest.json")

    texture_count = len(textures.assets)
    print(
        f"done: {len(exported_models)} {plural(len(exported_models), 'model', 'models', 'models')}, "
        f"{texture_count} {plural(texture_count, 'texture', 'textures', 'textures')}, "
        f"{len(written_files) + texture_count} file(s) -> {output}"
    )
    if scene.issues or textures.errors:
        print(f"warnings: {len(scene.issues) + len(textures.errors)} (details in manifest.json)")
    return 0

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(getattr(args, "verbose", 0))
    try:
        if args.command == "extract":
            return _cmd_extract(args)
        if args.command == "list":
            return _cmd_list(args)
        if args.command == "info":
            return _cmd_info(args)
        parser.print_help()
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        if getattr(args, "verbose", 0) >= 2:
            traceback.print_exc()
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
