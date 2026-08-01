"""Verify that batch comparison applies one external scenario YAML to each model."""

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "compare_best_models.py"


def _comparison_module():
    """Load the standalone comparison script without invoking its CLI entry point."""
    spec = importlib.util.spec_from_file_location("compare_best_models_test", str(SCRIPT_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_common_evaluation_yaml_overlays_only_evaluation_sections():
    """The common YAML changes simulator inputs but preserves the restored Actor setup."""
    module = _comparison_module()
    evaluation = module.read_yaml(PROJECT_ROOT / "configs" / "compare" / "compare_n100_heuristic_channel.yaml")
    model_config = {
        "experiment": {"id": "model_a", "master_seed": 7},
        "actor": {"residual": {"architecture": "mpnn"}},
        "training": {"consecutive_tx_scale": 9.0},
        "topology": {"type": "different"},
    }

    merged = module.with_evaluation_settings(model_config, evaluation)

    assert merged["actor"] == model_config["actor"]
    assert merged["training"] == model_config["training"]
    assert merged["experiment"]["id"] == "model_a"
    assert merged["experiment"]["master_seed"] == 20260723
    assert merged["topology"] == evaluation["topology"]
    assert len(merged["validation"]["scenarios"]) == 10
