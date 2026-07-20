"""Helpers for creating a free-floating DMBot MuJoCo scene."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


TERRAIN_FRICTION = "1.0 0.02 0.001"


def _has_named_child(parent: ET.Element, tag: str, name: str) -> bool:
    return any(child.tag == tag and child.get("name") == name for child in parent)


def _get_or_create(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.Element(tag)
        parent.append(child)
    return child


def _add_named(parent: ET.Element, tag: str, attrs: dict[str, str]) -> ET.Element:
    name = attrs.get("name")
    if name:
        existing = parent.find(f"./{tag}[@name='{name}']")
        if existing is not None:
            return existing
    child = ET.Element(tag, attrs)
    parent.append(child)
    return child


def _add_scene_visuals(root: ET.Element) -> None:
    visual = _get_or_create(root, "visual")
    if visual.find("headlight") is None:
        visual.append(
            ET.Element(
                "headlight",
                {
                    "diffuse": "0.6 0.6 0.6",
                    "ambient": "0.3 0.3 0.3",
                    "specular": "0 0 0",
                },
            )
        )
    if visual.find("rgba") is None:
        visual.append(ET.Element("rgba", {"haze": "0.15 0.25 0.35 1"}))
    quality = visual.find("quality")
    if quality is None:
        quality = ET.Element("quality")
        visual.append(quality)
    quality.set("shadowsize", "0")
    if visual.find("global") is None:
        visual.append(ET.Element("global", {"azimuth": "-130", "elevation": "-20"}))

    asset = _get_or_create(root, "asset")
    _add_named(
        asset,
        "texture",
        {
            "name": "training_skybox",
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "3072",
        },
    )
    _add_named(
        asset,
        "texture",
        {
            "name": "training_groundplane_texture",
            "type": "2d",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.28 0.33 0.35",
            "rgb2": "0.16 0.19 0.21",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    _add_named(
        asset,
        "material",
        {
            "name": "training_groundplane",
            "texture": "training_groundplane_texture",
            "texuniform": "true",
            "texrepeat": "8 8",
            "reflectance": "0.15",
        },
    )


def _terrain_geom(name: str, **attrs: str) -> ET.Element:
    data = {
        "name": name,
        "friction": TERRAIN_FRICTION,
        "contype": "1",
        "conaffinity": "1",
        "condim": "3",
    }
    data.update(attrs)
    return ET.Element("geom", data)


def _add_floor_and_light(worldbody: ET.Element) -> None:
    if not _has_named_child(worldbody, "geom", "floor"):
        worldbody.insert(
            0,
            _terrain_geom(
                "floor",
                type="plane",
                pos="0 0 0",
                size="24 8 0.05",
                material="training_groundplane",
            ),
        )

    if not _has_named_child(worldbody, "light", "main_light"):
        worldbody.insert(
            1,
            ET.Element(
                "light",
                {
                    "name": "main_light",
                    "pos": "0 0 3.5",
                    "dir": "0 0 -1",
                    "directional": "true",
                    "diffuse": "0.8 0.8 0.8",
                },
            ),
        )


def _add_training_mixed_terrain(worldbody: ET.Element) -> None:
    # Representative MuJoCo geometry for the active IsaacLab training terrains:
    # wave, slope_up, slope_down, rough_slope, stairs_up, stairs_down, obstacles, flat.
    terrain_geoms = [
        _terrain_geom(
            "training_slope_up",
            type="box",
            size="1.2 1.2 0.045",
            pos="2.3 0 0.22",
            euler="0 -10 0",
            rgba="0.72 0.72 0.72 1",
        ),
        _terrain_geom(
            "training_slope_down",
            type="box",
            size="1.2 1.2 0.045",
            pos="4.6 0 0.22",
            euler="0 10 0",
            rgba="0.72 0.72 0.72 1",
        ),
    ]

    step_height = 0.085
    step_width = 0.31
    # Adjacent course segments overlap slightly so rotated ramp boxes do not
    # leave small collision gaps that can catch the robot feet.
    start_x = 5.93
    for i in range(5):
        height = (i + 1) * step_height
        terrain_geoms.append(
            _terrain_geom(
                f"training_stairs_up_{i}",
                type="box",
                size=f"{step_width / 2:.3f} 1.15 {height / 2:.3f}",
                pos=f"{start_x + i * step_width:.3f} 0 {height / 2:.3f}",
                rgba="0.86 0.86 0.82 1",
            )
        )

    down_start_x = start_x + 5 * step_width
    for i in range(5):
        height = (5 - i) * step_height
        terrain_geoms.append(
            _terrain_geom(
                f"training_stairs_down_{i}",
                type="box",
                size=f"{step_width / 2:.3f} 1.15 {height / 2:.3f}",
                pos=f"{down_start_x + i * step_width:.3f} 0 {height / 2:.3f}",
                rgba="0.86 0.86 0.82 1",
            )
        )

    obstacle_specs = [
        (9.2, -0.45, 0.055, 0.20, 0.20),
        (9.8, 0.30, 0.110, 0.25, 0.18),
        (10.4, -0.10, 0.155, 0.22, 0.22),
        (11.0, 0.50, 0.070, 0.28, 0.20),
    ]
    for i, (x, y, height, sx, sy) in enumerate(obstacle_specs):
        terrain_geoms.append(
            _terrain_geom(
                f"training_obstacle_{i}",
                type="box",
                size=f"{sx:.3f} {sy:.3f} {height:.3f}",
                pos=f"{x:.3f} {y:.3f} {height:.3f}",
                rgba="0.78 0.62 0.38 1",
            )
        )

    def add_stair_segment(prefix: str, start_center_x: float, heights: list[float]) -> None:
        tread_depth = 0.30
        platform_length = 0.95
        half_tread = tread_depth / 2.0
        for i, height in enumerate(heights):
            terrain_geoms.append(
                _terrain_geom(
                    f"{prefix}_step_{i}",
                    type="box",
                    size=f"{half_tread:.3f} 0.900 {height / 2:.3f}",
                    pos=f"{start_center_x + i * tread_depth:.3f} 0 {height / 2:.3f}",
                    rgba="0.64 0.72 0.42 1",
                )
            )

        platform_start_x = start_center_x + (len(heights) - 0.5) * tread_depth
        platform_center_x = platform_start_x + platform_length / 2.0
        platform_height = heights[-1]
        terrain_geoms.append(
            _terrain_geom(
                f"{prefix}_platform",
                type="box",
                size=f"{platform_length / 2.0:.3f} 0.900 {platform_height / 2:.3f}",
                pos=f"{platform_center_x:.3f} 0 {platform_height / 2:.3f}",
                rgba="0.58 0.70 0.38 1",
            )
        )

    add_stair_segment("training_stairs_15cm", 11.75, [0.15 * (i + 1) for i in range(10)])
    add_stair_segment("training_stairs_20cm", 20.00, [0.20 * (i + 1) for i in range(10)])

    for i, height in enumerate([0.03, 0.065, 0.04, 0.08, 0.05, 0.095, 0.035, 0.07]):
        terrain_geoms.append(
            _terrain_geom(
                f"training_rough_slope_{i}",
                type="box",
                size="0.22 0.22 0.030",
                pos=f"{1.2 + i * 0.38:.3f} -1.75 {height:.3f}",
                rgba="0.50 0.50 0.48 1",
            )
        )

    for i, height in enumerate([0.04, 0.08, 0.12, 0.08, 0.04, 0.00, 0.04, 0.08]):
        terrain_geoms.append(
            _terrain_geom(
                f"training_wave_{i}",
                type="box",
                size="0.20 0.45 0.025",
                pos=f"{1.0 + i * 0.40:.3f} 1.75 {height + 0.025:.3f}",
                rgba="0.42 0.55 0.68 1",
            )
        )

    for geom in terrain_geoms:
        if not _has_named_child(worldbody, "geom", geom.get("name", "")):
            worldbody.append(geom)


def build_scene_with_training_terrain_xml(
    source_xml: str | Path,
    output_xml: str | Path,
    *,
    terrain_profile: str = "flat",
) -> Path:
    """Create a temporary MuJoCo scene with the shared deployment terrain added."""
    source_xml = Path(source_xml).resolve()
    output_xml = Path(output_xml).resolve()

    tree = ET.parse(source_xml)
    root = tree.getroot()

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", "0.005")
    option.set("gravity", "0 0 -9.81")

    for model_asset in root.findall("./asset/model"):
        model_file = model_asset.get("file")
        if model_file:
            model_path = Path(model_file)
            if not model_path.is_absolute():
                model_asset.set("file", str((source_xml.parent / model_path).resolve()))

    _add_scene_visuals(root)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing <worldbody> in MuJoCo scene: {source_xml}")

    _add_floor_and_light(worldbody)
    if terrain_profile == "training_mixed":
        _add_training_mixed_terrain(worldbody)
    elif terrain_profile != "flat":
        raise ValueError(f"Unsupported MuJoCo terrain profile: {terrain_profile}")

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return output_xml


def build_dmbot_scene_xml(
    source_xml: str | Path,
    output_xml: str | Path,
    *,
    terrain_profile: str = "flat",
) -> Path:
    """Create a deployable DMBot scene from the fixed-base robot MJCF.

    The RoboGauge DMBot MJCF is useful as a robot asset, but it has no freejoint
    and no ground plane. MuJoCo deployment needs a floating base so the policy can
    control locomotion, so this helper writes a generated scene without modifying
    the source asset.
    """
    source_xml = Path(source_xml).resolve()
    output_xml = Path(output_xml).resolve()

    tree = ET.parse(source_xml)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str(source_xml.parent / "assets"))

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", "0.005")
    option.set("gravity", "0 0 -9.81")

    _add_scene_visuals(root)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"Missing <worldbody> in DMBot MJCF: {source_xml}")

    _add_floor_and_light(worldbody)
    if terrain_profile == "training_mixed":
        _add_training_mixed_terrain(worldbody)
    elif terrain_profile != "flat":
        raise ValueError(f"Unsupported DMBot MuJoCo terrain profile: {terrain_profile}")

    base_body = worldbody.find("./body[@name='F_base']")
    if base_body is None:
        raise ValueError(f"Missing F_base body in DMBot MJCF: {source_xml}")

    if not _has_named_child(base_body, "freejoint", "root"):
        base_body.insert(0, ET.Element("freejoint", {"name": "root"}))

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)
    return output_xml
