from pathlib import Path
import zipfile

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


def _torchscript_data_sizes(policy_path: Path) -> list[int]:
    with zipfile.ZipFile(policy_path) as archive:
        return [
            info.file_size
            for info in archive.infolist()
            if "/data/" in info.filename
        ]


def test_go2_deploy_history_length_matches_exported_policy_buffer():
    cfg_path = ROOT_DIR / "deploy" / "deploy_mujoco" / "configs" / "go2.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    policy_path = Path(cfg["policy_path"].replace("{ROOT_DIR}", str(ROOT_DIR)))
    expected_buffer_bytes = int(cfg["num_obs"]) * int(cfg["history_len"]) * 4
    data_sizes = _torchscript_data_sizes(policy_path)

    assert data_sizes[0] == expected_buffer_bytes
