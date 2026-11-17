from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .util import log, object_identity

ALBEDO_SLOTS = (
    "_BaseMap", "_MainTex", "_AlbedoMap", "_Albedo", "_Diffuse", "_DiffuseMap",
    "_BaseColorMap", "_Atlas", "_Tex", "_Texture", "_MainTexture",
)
NORMAL_SLOTS = ("_BumpMap", "_NormalMap", "_NormalTex", "_DetailNormalMap")
METALLIC_SLOTS = ("_MetallicGlossMap", "_MetallicMap", "_MaskMap", "_SpecGlossMap")
OCCLUSION_SLOTS = ("_OcclusionMap", "_Occlusion", "_AO")
EMISSION_SLOTS = ("_EmissionMap", "_EmissiveMap", "_Emission")
COLOR_SLOTS = ("_BaseColor", "_Color", "_TintColor", "_MainColor", "_AlbedoColor", "_Tint")
METALLIC_FLOATS = ("_Metallic",)
SMOOTHNESS_FLOATS = ("_Glossiness", "_Smoothness")
ROUGHNESS_FLOATS = ("_Roughness",)

@dataclass
class TextureBinding:
    slot: str
    texture: Any
    scale: tuple[float, float] = (1.0, 1.0)
    offset: tuple[float, float] = (0.0, 0.0)

@dataclass
class MaterialData:
    key: tuple[str, int] | None
    name: str
    shader: str = ""
    albedo: TextureBinding | None = None
    normal: TextureBinding | None = None
    metallic_map: TextureBinding | None = None
    occlusion: TextureBinding | None = None
    emission_map: TextureBinding | None = None
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    emission_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metallic: float = 0.0
    roughness: float = 0.5
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    unlit: bool = False
    render_queue: int = 0
    floats: dict[str, float] = field(default_factory=dict)
    colors: dict[str, tuple[float, float, float, float]] = field(default_factory=dict)

    @property
    def has_albedo(self) -> bool:
        return self.albedo is not None and self.albedo.texture is not None

def _pairs(value: Any):
    if not value:
        return
    if isinstance(value, dict):
        yield from value.items()
        return
    for item in value:
        try:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                yield item[0], item[1]
            else:
                yield item[0], item[1]
        except Exception:
            continue

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

def _color_tuple(value: Any) -> tuple[float, float, float, float]:
    try:
        return (float(value.r), float(value.g), float(value.b), float(value.a))
    except Exception:
        pass
    try:
        packed = int(value.rgba)
        return (
            ((packed >> 24) & 0xFF) / 255.0,
            ((packed >> 16) & 0xFF) / 255.0,
            ((packed >> 8) & 0xFF) / 255.0,
            (packed & 0xFF) / 255.0,
        )
    except Exception:
        return (1.0, 1.0, 1.0, 1.0)

def _texture_binding(value: Any, slot: str) -> TextureBinding | None:
    texture = getattr(value, "m_Texture", None)
    if texture is None or not getattr(texture, "m_PathID", 0):
        return None
    scale = getattr(value, "m_Scale", None)
    offset = getattr(value, "m_Offset", None)
    sx = _as_float(getattr(scale, "x", 1.0), 1.0)
    sy = _as_float(getattr(scale, "y", 1.0), 1.0)
    ox = _as_float(getattr(offset, "x", 0.0), 0.0)
    oy = _as_float(getattr(offset, "y", 0.0), 0.0)
    return TextureBinding(slot=slot, texture=texture, scale=(sx, sy), offset=(ox, oy))

def _pick(mapping: dict[str, Any], names: tuple[str, ...]) -> tuple[str, Any] | None:
    lowered = {k.lower(): (k, v) for k, v in mapping.items()}
    for name in names:
        hit = lowered.get(name.lower())
        if hit is not None:
            return hit
    return None

def _shader_name(material: Any) -> str:
    shader_ptr = getattr(material, "m_Shader", None)
    if shader_ptr is None or not getattr(shader_ptr, "m_PathID", 0):
        return ""
    try:
        shader = shader_ptr.read()
        parsed = getattr(shader, "m_ParsedForm", None)
        return str(getattr(parsed, "m_Name", "") or getattr(shader, "m_Name", "") or "")
    except Exception:
        return ""

def read_material(obj: Any, base_path: str | None = None) -> MaterialData:
    material = obj.read()
    name = str(getattr(material, "m_Name", "material") or "material")
    props = getattr(material, "m_SavedProperties", None)
    texenvs: dict[str, Any] = {}
    floats: dict[str, float] = {}
    colors: dict[str, tuple[float, float, float, float]] = {}

    if props is not None:
        for slot, value in _pairs(getattr(props, "m_TexEnvs", None)):
            if value is not None:
                texenvs[str(slot)] = value
        for key, value in _pairs(getattr(props, "m_Floats", None)):
            floats[str(key)] = _as_float(value)
        for key, value in _pairs(getattr(props, "m_Colors", None)):
            colors[str(key)] = _color_tuple(value)

    shader = _shader_name(material)

    def tex(names: tuple[str, ...]) -> TextureBinding | None:
        hit = _pick(texenvs, names)
        if hit is None:
            return None
        slot, value = hit
        return _texture_binding(value, slot)

    albedo = tex(ALBEDO_SLOTS)
    normal = tex(NORMAL_SLOTS)
    metallic_map = tex(METALLIC_SLOTS)
    occlusion = tex(OCCLUSION_SLOTS)
    emission_map = tex(EMISSION_SLOTS)

    base_color = (1.0, 1.0, 1.0, 1.0)
    color_hit = _pick(colors, COLOR_SLOTS)
    if color_hit is not None:
        base_color = color_hit[1]

    emission = (0.0, 0.0, 0.0)
    emission_hit = _pick(colors, ("_EmissionColor", "_EmissiveColor"))
    if emission_hit is not None:
        emission = emission_hit[1][:3]

    metallic = 0.0
    metallic_hit = _pick(floats, METALLIC_FLOATS)
    if metallic_hit is not None:
        metallic = max(0.0, min(1.0, metallic_hit[1]))

    roughness = 0.5
    roughness_hit = _pick(floats, ROUGHNESS_FLOATS)
    if roughness_hit is not None:
        roughness = max(0.0, min(1.0, roughness_hit[1]))
    else:
        smoothness_hit = _pick(floats, SMOOTHNESS_FLOATS)
        if smoothness_hit is not None:
            roughness = max(0.0, min(1.0, 1.0 - smoothness_hit[1]))

    alpha_mode = "OPAQUE"
    cutoff = 0.5
    mode_hit = _pick(floats, ("_Mode",))
    alpha_clip_hit = _pick(floats, ("_AlphaClip",))
    surface_hit = _pick(floats, ("_Surface",))
    cull_hit = _pick(floats, ("_Cull",))
    cutoff_hit = _pick(floats, ("_Cutoff", "_AlphaCutoff"))
    if cutoff_hit is not None:
        cutoff = max(0.0, min(1.0, cutoff_hit[1]))

    mode = int(mode_hit[1]) if mode_hit is not None else None
    if mode_hit is not None and mode in (2, 3):
        alpha_mode = "BLEND"
    elif alpha_clip_hit is not None and alpha_clip_hit[1] >= 0.5:
        alpha_mode = "MASK"
    elif surface_hit is not None and int(surface_hit[1]) == 1:
        alpha_mode = "BLEND"
    elif mode_hit is not None and mode == 1:
        alpha_mode = "MASK"

    render_queue = int(_as_float(getattr(material, "m_CustomRenderQueue", 0)))
    if alpha_mode == "OPAQUE" and render_queue >= 3000:
        alpha_mode = "BLEND"
    elif alpha_mode == "OPAQUE" and 2450 <= render_queue < 3000:
        alpha_mode = "MASK"

    double_sided = cull_hit is not None and int(cull_hit[1]) == 0
    shader_lower = shader.lower()
    unlit = "unlit" in shader_lower or "sprite" in shader_lower or shader_lower.startswith("ui/")

    data = MaterialData(
        key=object_identity(obj, base_path),
        name=name,
        shader=shader,
        albedo=albedo,
        normal=normal,
        metallic_map=metallic_map,
        occlusion=occlusion,
        emission_map=emission_map,
        base_color=base_color,
        emission_color=emission,
        metallic=metallic,
        roughness=roughness,
        alpha_mode=alpha_mode,
        alpha_cutoff=cutoff,
        double_sided=double_sided,
        unlit=unlit,
        render_queue=render_queue,
        floats=floats,
        colors=colors,
    )
    log.debug("material %s shader=%s", name, shader)
    return data
