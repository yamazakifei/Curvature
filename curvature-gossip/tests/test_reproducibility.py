"""验证同种子结果一致、策略顺序不改变外生实现以及输出防覆盖。"""

import json
from pathlib import Path

import pytest
import yaml

from curvature_gossip.experiments import run_experiment


def _config(output_root, policies):
    return {
        "experiment": {
            "id": "repro", "slots": 20, "warmup_slots": 2, "trace_stride": 2,
            "master_seed": 123, "topology_seeds": [0], "channel_seeds": [0], "update_seeds": [0],
        },
        "topology": {"type": "grid_2d", "params": {"rows": 2, "columns": 3, "spacing_m": 1.0}},
        "source": {"update_probability": 0.2},
        "channel": {
            "tx_power_dbm": 20.0, "pathloss_reference_db": 20.0, "pathloss_exponent": 2.0,
            "shadowing_std_db": 0.0, "rayleigh_fading": True,
            "noise_psd_dbm_per_hz": -174.0, "bandwidth_hz": 1000000.0,
            "noise_figure_db": 0.0, "sinr_threshold_db": -5.0,
        },
        "curvature": {"method": "global_orc", "alpha": 0.5, "normalization": "global_negative_max"},
        "policies": policies,
        "output": {"root": str(output_root), "save_per_slot": True},
    }


POLICIES = [
    {"name": "random", "type": "random", "params": {"tx_probability": 0.2}},
    {"name": "fresh", "type": "freshness", "params": {"q_min": 0.01, "q_max": 0.4}},
]


def _write(path, content):
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(content, stream, sort_keys=False)


def test_same_seed_and_policy_order_produce_identical_summaries(tmp_path):
    first_config = tmp_path / "first.yaml"
    second_config = tmp_path / "second.yaml"
    _write(first_config, _config(tmp_path / "out1", POLICIES))
    _write(second_config, _config(tmp_path / "out2", list(reversed(POLICIES))))
    first = run_experiment(str(first_config), progress=False)
    second = run_experiment(str(second_config), progress=False)
    first_by_policy = {row["policy"]: row for row in first.run_summaries}
    second_by_policy = {row["policy"]: row for row in second.run_summaries}
    assert first_by_policy == second_by_policy


def test_existing_run_is_not_overwritten_without_flag(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write(config_path, _config(tmp_path / "out", POLICIES[:1]))
    run_experiment(str(config_path), progress=False)
    with pytest.raises(FileExistsError):
        run_experiment(str(config_path), progress=False)
