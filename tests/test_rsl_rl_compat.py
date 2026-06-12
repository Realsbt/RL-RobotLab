from pathlib import Path
import sys
from types import SimpleNamespace


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts" / "rsl_rl"))

import cli_args  # noqa: E402


def test_update_rsl_rl_cfg_removes_legacy_unsupported_optimizer_field():
    cfg = SimpleNamespace(
        algorithm=SimpleNamespace(class_name="MoECTS", optimizer="adam", share_cnn_encoders=False),
        logger=None,
    )
    args = SimpleNamespace(
        seed=None,
        experiment_name=None,
        resume=None,
        load_run=None,
        checkpoint=None,
        run_name=None,
        logger=None,
        log_project_name=None,
    )

    updated_cfg = cli_args.update_rsl_rl_cfg(cfg, args)

    assert not hasattr(updated_cfg.algorithm, "optimizer")
    assert not hasattr(updated_cfg.algorithm, "share_cnn_encoders")
