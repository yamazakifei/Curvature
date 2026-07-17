"""从 YAML 配置训练局部 AF3、双层广播约束的共享 CTDE 神经策略。"""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvature_gossip.learning.trainer import train_ctde


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the distributed AF3 CTDE policy")
    parser.add_argument("--config", required=True, help="training YAML path")
    args = parser.parse_args()
    output = train_ctde(args.config)
    print("NN artifacts saved to {}".format(output))


if __name__ == "__main__":
    main()
