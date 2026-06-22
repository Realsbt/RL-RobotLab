from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

import mujoco
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
DEPLOY_DIR = ROOT_DIR / "deploy" / "deploy_mujoco"
DMBOT_JOINT_NAMES = [
    "Jfl1_hipr",
    "Jfl2_hipp",
    "Jfl3_knee",
    "Jfr1_hipr",
    "Jfr2_hipp",
    "Jfr3_knee",
    "Jrl1_hipr",
    "Jrl2_hipp",
    "Jrl3_knee",
    "Jrr1_hipr",
    "Jrr2_hipp",
    "Jrr3_knee",
]

if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))


def _torchscript_data_sizes(policy_path: Path) -> list[int]:
    with zipfile.ZipFile(policy_path) as archive:
        return [
            info.file_size
            for info in archive.infolist()
            if "/data/" in info.filename
        ]


def _xml_box_x_range(scene_xml: Path, geom_name: str) -> tuple[float, float]:
    pos, size = _xml_box_pos_size(scene_xml, geom_name)
    return pos[0] - size[0], pos[0] + size[0]


def _xml_box_pos_size(scene_xml: Path, geom_name: str) -> tuple[list[float], list[float]]:
    tree = ET.parse(scene_xml)
    geom = tree.getroot().find(f".//geom[@name='{geom_name}']")
    assert geom is not None

    pos = [float(value) for value in geom.get("pos", "0 0 0").split()]
    size = [float(value) for value in geom.get("size", "0 0 0").split()]
    return pos, size


def _xml_box_height(scene_xml: Path, geom_name: str) -> float:
    pos, size = _xml_box_pos_size(scene_xml, geom_name)
    assert pos[2] == size[2]
    return size[2] * 2.0


def test_go2_deploy_history_length_matches_exported_policy_buffer():
    cfg_path = ROOT_DIR / "deploy" / "deploy_mujoco" / "configs" / "go2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    policy_path = Path(cfg["policy_path"].replace("{ROOT_DIR}", str(ROOT_DIR)))
    expected_buffer_bytes = int(cfg["num_obs"]) * int(cfg["history_len"]) * 4
    data_sizes = _torchscript_data_sizes(policy_path)

    assert data_sizes[0] == expected_buffer_bytes


def test_dmbot_deploy_history_length_matches_exported_policy_buffer():
    cfg_path = DEPLOY_DIR / "configs" / "dmbot.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    policy_path = Path(cfg["policy_path"].replace("{ROOT_DIR}", str(ROOT_DIR)))
    expected_buffer_bytes = int(cfg["num_obs"]) * int(cfg["history_len"]) * 4
    data_sizes = _torchscript_data_sizes(policy_path)

    assert data_sizes[0] == expected_buffer_bytes


def test_dmbot_deploy_config_matches_mjcf_joint_order():
    cfg_path = DEPLOY_DIR / "configs" / "dmbot.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    xml_path = Path(cfg["xml_path"].replace("{ROOT_DIR}", str(ROOT_DIR)))

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    mjcf_joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]

    assert cfg["model_joint_names"] == DMBOT_JOINT_NAMES
    assert cfg["mujoco_joint_names"] == DMBOT_JOINT_NAMES
    assert mjcf_joint_names == DMBOT_JOINT_NAMES
    assert cfg["default_angles"] == [
        0.0,
        -0.8,
        -1.5,
        0.0,
        0.8,
        1.5,
        0.0,
        -1.0,
        -1.5,
        0.0,
        1.0,
        1.5,
    ]


def test_dmbot_scene_builder_adds_freejoint_and_floor(tmp_path):
    from dmbot_scene import build_dmbot_scene_xml

    source_xml = ROOT_DIR / "external" / "RoboGauge" / "resources" / "robots" / "dmbot" / "dmbot.xml"
    scene_xml = tmp_path / "dmbot_floating.xml"

    build_dmbot_scene_xml(source_xml, scene_xml)
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    geom_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
    }

    assert model.nq == 19
    assert model.nv == 18
    assert joint_names == ["root", *DMBOT_JOINT_NAMES]
    assert "floor" in geom_names


def test_dmbot_deploy_config_uses_training_mixed_terrain_profile():
    from utils import load_config

    cfg = load_config("dmbot.yaml")

    assert cfg.terrain_profile == "training_mixed"


def test_dmbot_mujoco_starts_with_zero_command_for_manual_control():
    from utils import load_config

    cfg = load_config("dmbot.yaml")

    assert cfg.cmd_init.tolist() == [0.0, 0.0, 0.0]


def test_dmbot_training_mixed_scene_contains_training_terrain_subset(tmp_path):
    from dmbot_scene import build_dmbot_scene_xml

    source_xml = ROOT_DIR / "external" / "RoboGauge" / "resources" / "robots" / "dmbot" / "dmbot.xml"
    scene_xml = tmp_path / "dmbot_training_mixed.xml"

    build_dmbot_scene_xml(source_xml, scene_xml, terrain_profile="training_mixed")
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    geom_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
    }

    expected_names = {
        "floor",
        "training_wave_0",
        "training_slope_up",
        "training_slope_down",
        "training_rough_slope_0",
        "training_stairs_up_0",
        "training_stairs_down_0",
        "training_obstacle_0",
        "training_stairs_15cm_step_0",
        "training_stairs_15cm_platform",
        "training_stairs_20cm_step_0",
        "training_stairs_20cm_platform",
    }
    assert expected_names.issubset(geom_names)
    assert model.nq == 19
    assert model.nv == 18


def test_dmbot_training_mixed_main_path_has_no_terrain_gaps(tmp_path):
    from dmbot_scene import build_dmbot_scene_xml

    source_xml = ROOT_DIR / "external" / "RoboGauge" / "resources" / "robots" / "dmbot" / "dmbot.xml"
    scene_xml = tmp_path / "dmbot_training_mixed.xml"

    build_dmbot_scene_xml(source_xml, scene_xml, terrain_profile="training_mixed")

    _, slope_up_end = _xml_box_x_range(scene_xml, "training_slope_up")
    slope_down_start, slope_down_end = _xml_box_x_range(scene_xml, "training_slope_down")
    stairs_up_start, _ = _xml_box_x_range(scene_xml, "training_stairs_up_0")
    _, stairs_up_end = _xml_box_x_range(scene_xml, "training_stairs_up_4")
    stairs_down_start, _ = _xml_box_x_range(scene_xml, "training_stairs_down_0")

    assert slope_down_start <= slope_up_end + 1e-6
    assert stairs_up_start <= slope_down_end + 1e-6
    assert stairs_down_start <= stairs_up_end + 1e-6


def test_dmbot_training_mixed_includes_two_10_step_stair_segments_with_platforms(tmp_path):
    from dmbot_scene import build_dmbot_scene_xml

    source_xml = ROOT_DIR / "external" / "RoboGauge" / "resources" / "robots" / "dmbot" / "dmbot.xml"
    scene_xml = tmp_path / "dmbot_training_mixed.xml"

    build_dmbot_scene_xml(source_xml, scene_xml, terrain_profile="training_mixed")

    expected_segments = {
        "training_stairs_15cm": [round(0.15 * (i + 1), 3) for i in range(10)],
        "training_stairs_20cm": [round(0.20 * (i + 1), 3) for i in range(10)],
    }
    segment_ranges = {}
    for prefix, heights in expected_segments.items():
        previous_end = None
        for i, height in enumerate(heights):
            name = f"{prefix}_step_{i}"
            assert round(_xml_box_height(scene_xml, name), 3) == height
            start, end = _xml_box_x_range(scene_xml, name)
            segment_ranges.setdefault(prefix, [start, end])
            segment_ranges[prefix][1] = end
            if previous_end is not None:
                assert start <= previous_end + 1e-6
            previous_end = end

        platform_name = f"{prefix}_platform"
        platform_start, platform_end = _xml_box_x_range(scene_xml, platform_name)
        _, platform_size = _xml_box_pos_size(scene_xml, platform_name)
        assert round(_xml_box_height(scene_xml, platform_name), 3) == heights[-1]
        assert platform_start <= previous_end + 1e-6
        assert platform_size[0] * 2.0 >= 0.8
        segment_ranges[prefix][1] = platform_end

    gap_between_stair_segments = segment_ranges["training_stairs_20cm"][0] - segment_ranges["training_stairs_15cm"][1]
    assert gap_between_stair_segments >= 4.0
