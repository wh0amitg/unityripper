from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from .util import log, object_identity, object_path_id, sanitize_filename, unique_name

try:
    from UnityPy.enums import TextureFormat

    _FORMAT_NAMES = {int(value): value.name for value in TextureFormat}
except Exception:
    _FORMAT_NAMES = {}

def texture_format_name(value: Any) -> str:
    if value is None:
        return ""
    name = getattr(value, "name", None)
    if name:
        return str(name)
    try:
        return _FORMAT_NAMES.get(int(value), str(int(value)))
    except Exception:
        return str(value)

try:
    import numpy as _np
except Exception:
    _np = None

NORMAL_FORMATS = {"DXT5", "DXT5Crunched", "BC3"}
_FLOAT_FORMATS = {
    "RHalf": (1, "<f2"),
    "RGHalf": (2, "<f2"),
    "RGBAHalf": (4, "<f2"),
    "RFloat": (1, "<f4"),
    "RGFloat": (2, "<f4"),
    "RGBAFloat": (4, "<f4"),
    "RGBFloat": (3, "<f4"),
}

@dataclass
class TextureAsset:
    key: tuple[int, int]
    name: str
    file: Path
    width: int
    height: int
    image: Image.Image | None
    format_name: str = ""
    wrap_u: int = 0
    wrap_v: int = 0
    variant: str = ""
    _png: bytes | None = field(default=None, repr=False)

    def png_bytes(self) -> bytes:
        if self._png is None:
            if self.image is None:
                self._png = self.file.read_bytes()
            else:
                buffer = io.BytesIO()
                self.image.save(buffer, format="PNG")
                self._png = buffer.getvalue()
        return self._png

    @property
    def relative_path(self) -> str:
        return f"textures/{self.file.name}"

def _get(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default

def _wrap_mode(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0

def _unpack_dxt5nm(image: Image.Image) -> Image.Image:
    if _np is None:
        return image.convert("RGB")
    rgba = _np.asarray(image.convert("RGBA"), dtype=_np.float32) / 255.0
    x = rgba[..., 3]
    y = rgba[..., 1]
    z = _np.sqrt(_np.clip(1.0 - x * x - y * y, 0.0, 1.0))
    out = _np.stack([x, y, z], axis=-1)
    return Image.fromarray((_np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(_np.uint8), "RGB")

def _has_pixel_data(texture: Any) -> bool:
    try:
        inline = getattr(texture, "image_data", b"") or b""
    except Exception:
        inline = b""
    if inline:
        return True
    stream = getattr(texture, "m_StreamData", None)
    if stream is None:
        return False
    try:
        path = str(getattr(stream, "path", "") or "")
        size = int(getattr(stream, "size", 0) or 0)
    except Exception:
        return False
    return bool(path or size)

def _decode_float_texture(texture: Any, fmt: str) -> Image.Image:
    if _np is None:
        raise ValueError("numpy is required to decode float textures")
    if fmt not in _FLOAT_FORMATS:
        raise ValueError(f"no fallback decoder for {fmt}")
    channels, dtype = _FLOAT_FORMATS[fmt]
    width = int(getattr(texture, "m_Width", 0) or 0)
    height = int(getattr(texture, "m_Height", 0) or 0)
    if width <= 0 or height <= 0:
        raise ValueError("texture has zero size")
    data = texture.get_image_data()
    itemsize = _np.dtype(dtype).itemsize
    expected = width * height * channels * itemsize
    if len(data) < expected:
        raise ValueError(f"truncated {fmt} texture: {len(data)} < {expected} bytes")
    values = _np.frombuffer(data[:expected], dtype=dtype).reshape(height, width, channels)
    values = _np.flipud(values).astype(_np.float32)

    rgb = _np.clip(values[..., : min(channels, 3)], 0.0, 1.0)
    rgb = _np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * _np.power(rgb, 1.0 / 2.4) - 0.055)
    rgb8 = (_np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(_np.uint8)

    if channels == 1:
        return Image.fromarray(rgb8[..., 0], "L")
    if channels == 2:
        alpha = (_np.clip(values[..., 1], 0.0, 1.0) * 255.0 + 0.5).astype(_np.uint8)
        return Image.fromarray(_np.stack([rgb8[..., 0], alpha], axis=-1), "LA")
    if channels == 3:
        return Image.fromarray(rgb8, "RGB")
    alpha = (_np.clip(values[..., 3], 0.0, 1.0) * 255.0 + 0.5).astype(_np.uint8)
    rgba = _np.dstack([rgb8, alpha])
    return Image.fromarray(rgba, "RGBA")

def _alpha_to_gray(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"}:
        return image.getchannel("A")
    if image.mode == "P":
        return image.convert("RGBA").getchannel("A")
    return image.convert("L")

def _ensure_png_mode(image: Image.Image) -> Image.Image:
    if image.mode == "F":
        if _np is None:
            return image.convert("L")
        values = _np.clip(_np.asarray(image, dtype=_np.float32), 0.0, 1.0)
        return Image.fromarray((values * 255.0 + 0.5).astype(_np.uint8), "L")
    if image.mode.startswith("I;16") or image.mode == "I":
        if _np is None:
            return image.convert("L")
        values = _np.clip(_np.asarray(image, dtype=_np.float64) / 257.0, 0.0, 255.0)
        return Image.fromarray(values.astype(_np.uint8), "L")
    if image.mode in {"CMYK", "YCbCr", "HSV"}:
        return image.convert("RGB")
    return image

class TextureStore:

    def __init__(
        self,
        output_dir: Path,
        normal_mode: str = "auto",
        enabled: bool = True,
        base_path: str | Path | None = None,
        retain_images: bool = True,
    ):
        self.output_dir = output_dir
        self.directory = output_dir / "textures"
        self.normal_mode = normal_mode
        self.enabled = enabled
        self.base_path = base_path
        self.retain_images = retain_images
        self._cache: dict[tuple[str, int, str], TextureAsset | None] = {}
        self._used_names: set[str] = set()
        self.assets: list[TextureAsset] = []
        self.errors: list[str] = []

    def _key(self, obj: Any) -> tuple[str, int]:
        return object_identity(obj, self.base_path)

    def _decode(self, obj: Any) -> tuple[Image.Image, str, int, int]:
        texture = obj.read()
        fmt = texture_format_name(_get(texture, "m_TextureFormat", None))
        settings = _get(texture, "m_TextureSettings", None)
        wrap_u = _wrap_mode(_get(settings, "m_WrapU", 0))
        wrap_v = _wrap_mode(_get(settings, "m_WrapV", 0))
        try:
            image = texture.image
        except Exception as exc:
            if not _has_pixel_data(texture):
                raise ValueError("no pixel data (runtime-generated or empty stream)") from exc
            if fmt not in _FLOAT_FORMATS:
                raise
            image = _decode_float_texture(texture, fmt)
        image = _ensure_png_mode(image)
        if fmt in {"Alpha8", "A8"}:
            image = _alpha_to_gray(image)
        if image.width == 0 or image.height == 0:
            raise ValueError("texture has zero size")
        return image, fmt, wrap_u, wrap_v

    def save(self, obj: Any, hint: str = "texture", role: str = "albedo") -> TextureAsset | None:
        if obj is None or not self.enabled:
            return None
        if object_path_id(obj) == 0:
            return None
        if role == "normal" and self.normal_mode == "off":
            return None

        key = self._key(obj)
        variant = ""
        try:
            texture = obj.read()
            fmt = texture_format_name(_get(texture, "m_TextureFormat", None))
            if role == "normal" and self.normal_mode == "auto" and fmt in NORMAL_FORMATS:
                variant = "n"
        except Exception:
            variant = ""

        cache_key = (key[0], key[1], variant)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            image, fmt, wrap_u, wrap_v = self._decode(obj)
            name = _get(texture, "m_Name", "") or hint
            if variant == "n":
                image = _unpack_dxt5nm(image)
            elif role == "normal" and self.normal_mode == "raw":
                image = image.copy()

            stem = sanitize_filename(name, fallback=sanitize_filename(hint, "texture"))
            if variant:
                stem = f"{stem}_n"
            stem = unique_name(stem, self._used_names, suffix=str(key[1]) if key[1] else "")
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{stem}.png"
            image.save(path, format="PNG")

            asset = TextureAsset(
                key=key,
                name=name,
                file=path,
                width=image.width,
                height=image.height,
                image=image if self.retain_images else None,
                format_name=fmt,
                wrap_u=wrap_u,
                wrap_v=wrap_v,
                variant=variant,
            )
            self._cache[cache_key] = asset
            self.assets.append(asset)
            log.debug("texture %s -> %s (%dx%d %s)", name, path.name, image.width, image.height, fmt)
            return asset
        except Exception as exc:
            self._cache[cache_key] = None
            message = f"texture '{hint}' failed: {type(exc).__name__}: {exc}"
            self.errors.append(message)
            log.warning(message)
            return None

    def save_metallic_roughness(self, obj: Any, hint: str = "material") -> TextureAsset | None:
        if obj is None or not self.enabled:
            return None
        key = self._key(obj)
        cache_key = (key[0], key[1], "mr")
        if cache_key in self._cache:
            return self._cache[cache_key]
        try:
            source = self.save(obj, hint=hint, role="albedo")
            if source is None:
                self._cache[cache_key] = None
                return None
            if source.image is not None:
                rgba = source.image.convert("RGBA")
            else:
                with Image.open(source.file) as handle:
                    rgba = handle.convert("RGBA")
            r, _g, _b, a = rgba.split()
            from PIL import ImageChops

            rough = ImageChops.invert(a)
            out = Image.merge("RGBA", (r, rough, r, Image.new("L", rgba.size, 255)))
            stem = f"{sanitize_filename(hint, 'material')}_mr"
            stem = unique_name(stem, self._used_names, suffix=str(key[1]) if key[1] else "")
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{stem}.png"
            out.save(path, format="PNG")
            asset = TextureAsset(
                key=key,
                name=f"{hint}_mr",
                file=path,
                width=out.width,
                height=out.height,
                image=out if self.retain_images else None,
                format_name="RGBA32",
                wrap_u=source.wrap_u,
                wrap_v=source.wrap_v,
                variant="mr",
            )
            self._cache[cache_key] = asset
            self.assets.append(asset)
            return asset
        except Exception as exc:
            self._cache[cache_key] = None
            message = f"metallic/roughness repack for '{hint}' failed: {type(exc).__name__}: {exc}"
            self.errors.append(message)
            log.warning(message)
            return None
