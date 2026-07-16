"""验证 YAML 配置、CLI 骨架以及拓扑图片输出。"""

from pathlib import Path

from curvature_gossip.cli import main
from curvature_gossip.config import load_config
from curvature_gossip.topology import registered_topologies


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_smoke_config_loads():
    config = load_config(str(PROJECT_ROOT / "configs" / "smoke_test.yaml"))
    assert config.topology.type in registered_topologies()
    assert config.experiment["master_seed"] == 20260713


def test_topology_cli_creates_nonempty_png(tmp_path):
    output = tmp_path / "topology.png"
    code = main([
        "topology", "--config", str(PROJECT_ROOT / "configs" / "smoke_test.yaml"),
        "--output", str(output),
    ])
    assert code == 0
    assert output.stat().st_size > 0
