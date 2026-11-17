from __future__ import annotations

import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterable

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_SPACES = re.compile(r"\s+")
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

log = logging.getLogger("unityripper")

def sanitize_filename(name: str, fallback: str = "unnamed", max_length: int = 100) -> str:
    name = unicodedata.normalize("NFKC", str(name or "")).strip()
    name = _INVALID.sub("_", name)
    name = _SPACES.sub(" ", name).strip(" .")
    if not name:
        name = fallback
    if len(name) > max_length:
        name = name[:max_length].rstrip(" .")
    if not name:
        name = fallback
    if name.split(".")[0].upper() in _RESERVED:
        name = f"_{name}"
    return name

def sanitize_node_path(path: str) -> str:
    parts = [sanitize_filename(p, "node", 48) for p in path.split("/") if p]
    return "__".join(parts) if parts else "model"

def unique_name(base: str, used: set[str], suffix: str = "") -> str:
    if suffix:
        candidate = f"{base}_{suffix}"
    else:
        candidate = base
    if candidate not in used:
        used.add(candidate)
        return candidate
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    candidate = f"{base}_{i}"
    used.add(candidate)
    return candidate

def relative_posix(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path, start)).as_posix()

def asset_file_of(obj: object) -> object | None:
    assets_file = getattr(obj, "assets_file", None)
    if assets_file is None:
        assets_file = getattr(obj, "assetsfile", None)
    return assets_file

def object_path_id(obj: object) -> int:
    for attribute in ("path_id", "m_PathID"):
        value = getattr(obj, attribute, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0

def object_identity(obj: object, base: str | Path | None = None) -> tuple[str, int]:
    assets_file = asset_file_of(obj)
    name = str(getattr(assets_file, "name", "") or "") if assets_file is not None else ""
    if name:
        path = Path(name)
        if not path.is_absolute() and base is not None:
            path = Path(base) / path
        name = os.path.normcase(os.path.normpath(str(path)))
    return name, object_path_id(obj)

def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"

def plural(count: int, one: str, few: str, many: str) -> str:
    n = abs(count) % 100
    n1 = n % 10
    if 11 <= n <= 19:
        return many
    if n1 == 1:
        return one
    if 2 <= n1 <= 4:
        return few
    return many

def walk_files(root: Path) -> Iterable[Path]:
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            yield Path(dirpath) / name
