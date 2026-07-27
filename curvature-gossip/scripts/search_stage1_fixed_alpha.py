"""Run the fixed-global-alpha Stage-1 validation grid and save CSV artifacts."""

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from curvature_gossip.learning.fixed_alpha_grid import (
    DEFAULT_ALPHA_VALUES,
    DEFAULT_UPDATE_PROBABILITIES,
    save_fixed_alpha_grid,
)


def main() -> None:
    """Parse search options and run the paired fixed-alpha evaluation grid."""
    parser = argparse.ArgumentParser(description="Evaluate fixed global Stage-1 alpha candidates")
    parser.add_argument("--config", required=True, help="base Stage-1 YAML with validation.scenarios")
    parser.add_argument(
        "--output", default="result_2layers/fixed_alpha_grid",
        help="directory for the aggregate and per-scenario CSV files",
    )
    parser.add_argument(
        "--update-probabilities", type=float, nargs="+", default=DEFAULT_UPDATE_PROBABILITIES,
        help="node update probabilities; default: 0.05 0.10 0.20",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=DEFAULT_ALPHA_VALUES,
        help="fixed global alpha values; default: 0.1 0.2 0.5 1 1.5 2 3",
    )
    args = parser.parse_args()
    output = save_fixed_alpha_grid(
        args.config, args.output, args.update_probabilities, args.alphas
    )
    print("Fixed-alpha grid artifacts saved to {}".format(output))


if __name__ == "__main__":
    main()
