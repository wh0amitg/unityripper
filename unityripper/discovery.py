from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .util import log, walk_files

BUNDLE_MAGICS = (b"UnityFS", b"UnityWeb", b"UnityRaw")
BUNDLE_EXTS = {".bundle", ".unity3d", ".ab", ".assetbundle", ".assetsbundle", ".unitypackage"}
ARCHIVE_EXTS = {".apk", ".zip", ".obb", ".xapk"}
ASSET_EXTS = {".assets", ".asset"}
SKIP_EXTS = {
    ".dll", ".exe", ".pdb", ".json", ".xml", ".txt", ".md", ".log", ".ini", ".cfg",
    ".png", ".jpg", ".jpeg", ".dds", ".tga", ".wav", ".mp3", ".ogg", ".mp4", ".bk2",
    ".resS", ".resource", ".config", ".hash", ".info", ".manifest", ".csv", ".bytes",
    ".shader", ".cginc", ".hlsl", ".compute", ".meta", ".pdb", ".mdb", ".so", ".dylib",
}
CORE_NAMES = {"globalgamemanagers", "globalgamemanagers.assets", "resources.assets", "data.unity3d"}
_SPLIT_RE = re.compile(r"^(.*)\.split\d+$")

@dataclass
class InputSet:
    files: list[Path] = field(default_factory=list)
    game_roots: list[Path] = field(default_factory=list)
    data_dirs: list[Path] = field(default_factory=list)
    archives: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def all_sources(self) -> list[Path]:
        return [*self.files, *self.archives]

def sniff_magic(path: Path) -> str | None:
    try:
        with path.open("rb") as fh:
            head = fh.read(8)
    except OSError:
        return None
    if any(head.startswith(m) for m in BUNDLE_MAGICS):
        return "bundle"
    return None

def _is_data_dir(path: Path) -> bool:
    return (path / "globalgamemanagers").is_file() or (path / "globalgamemanagers.assets").is_file()

def find_data_dirs(root: Path) -> list[Path]:
    found: list[Path] = []
    if _is_data_dir(root):
        found.append(root)
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and _is_data_dir(entry):
            found.append(entry)
    if not found:
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            if _is_data_dir(entry):
                found.append(entry)
                continue
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and _is_data_dir(sub):
                    found.append(sub)
    return found

def _core_assets(data_dir: Path) -> Iterable[Path]:
    for entry in data_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        lower = name.lower()
        if name in CORE_NAMES:
            yield entry
        elif lower.startswith("sharedassets") and lower.endswith(".assets"):
            yield entry
        elif lower.startswith("level") and not lower.endswith((".resS", ".resource")):
            yield entry
        elif lower.endswith(".assets"):
            yield entry
        elif lower == "data.unity3d":
            yield entry

def _bundles_under(root: Path) -> Iterable[Path]:
    if not root.is_dir():
        return
    for path in walk_files(root):
        lower = path.name.lower()
        if lower.endswith((".manifest", ".json", ".txt", ".hash", ".info")):
            continue
        if path.suffix.lower() in BUNDLE_EXTS or sniff_magic(path) == "bundle":
            yield path

def collect(inputs: list[str], recursive: bool = True) -> InputSet:
    result = InputSet()
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen or not path.exists():
            return
        seen.add(resolved)
        result.files.append(path)

    for raw in inputs:
        path = Path(raw)
        if not path.exists():
            result.notes.append(f"input not found: {raw}")
            continue

        if path.is_file():
            ext = path.suffix.lower()
            if ext in ARCHIVE_EXTS:
                result.archives.append(path)
            else:
                add(path)
            continue

        data_dirs = find_data_dirs(path)
        if data_dirs:
            result.game_roots.append(path)
            for data_dir in data_dirs:
                result.data_dirs.append(data_dir)
                for asset in _core_assets(data_dir):
                    add(asset)
                for bundle_dir in (data_dir / "StreamingAssets", data_dir / "AssetBundles", data_dir / "Bundles"):
                    for bundle in _bundles_under(bundle_dir):
                        add(bundle)
            continue

        if recursive:
            for candidate in walk_files(path):
                ext = candidate.suffix.lower()
                if ext in ARCHIVE_EXTS:
                    result.archives.append(candidate)
                elif ext in ASSET_EXTS or ext in BUNDLE_EXTS:
                    add(candidate)
                elif sniff_magic(candidate) == "bundle":
                    add(candidate)
        else:
            for candidate in path.iterdir():
                if candidate.is_file() and (
                    candidate.suffix.lower() in ASSET_EXTS | BUNDLE_EXTS or sniff_magic(candidate) == "bundle"
                ):
                    add(candidate)

    normalized: list[Path] = []
    for path in result.files:
        match = _SPLIT_RE.match(str(path))
        if match:
            base = Path(match.group(1))
            path = base
        if path not in normalized:
            normalized.append(path)
    result.files = normalized

    if not result.files and not result.archives:
        result.notes.append("no Unity .assets files, bundles or archives were found")
    else:
        log.debug("discovered %d asset file(s), %d archive(s)", len(result.files), len(result.archives))
    return result
