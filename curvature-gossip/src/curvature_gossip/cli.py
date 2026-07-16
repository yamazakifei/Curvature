"""提供配置校验、拓扑生成/绘图及后续仿真命令的统一 CLI 入口。"""

import argparse
import sys

import numpy as np

from .config import ConfigError, load_config
from .plotting import plot_topology
from .topology import get_topology_generator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="curvature-gossip")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="校验 YAML 配置")
    validate.add_argument("--config", required=True)
    topology = subparsers.add_parser("topology", help="生成并绘制一个拓扑")
    topology.add_argument("--config", required=True)
    topology.add_argument("--output", required=True)
    annotate = subparsers.add_parser(
        "annotate-curvature", help="add bottleneck importance to a saved curvature CSV"
    )
    annotate.add_argument("--run-dir", required=True)
    for name in ("run", "plot"):
        help_text = "运行配对多策略实验" if name == "run" else "从已保存结果生成图片"
        future = subparsers.add_parser(name, help=help_text)
        future.add_argument("--config" if name == "run" else "--results", required=True)
        if name == "run":
            future.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "annotate-curvature":
        from .experiments import annotate_bottleneck_importance
        try:
            output = annotate_bottleneck_importance(args.run_dir)
            print("Curvature CSV updated: {}".format(output))
            return 0
        except (FileNotFoundError, ValueError, OSError) as error:
            print("Error: {}".format(error), file=sys.stderr)
            return 2
    if args.command == "run":
        from .experiments import run_experiment
        try:
            run_experiment(args.config, overwrite=args.overwrite, progress=True)
            return 0
        except (ConfigError, ValueError, FileExistsError) as error:
            print("Error: {}".format(error), file=sys.stderr)
            return 2
    if args.command == "plot":
        from .plotting import plot_experiment_results
        try:
            output = plot_experiment_results(args.results)
            print("Plots saved: {}".format(output))
            return 0
        except ValueError as error:
            print("Error: {}".format(error), file=sys.stderr)
            return 2
    try:
        config = load_config(args.config)
        if args.command == "validate":
            print("Configuration valid: topology={}".format(config.topology.type))
            return 0
        seed = int(config.experiment.get("master_seed", 0))
        generator = get_topology_generator(config.topology.type)
        generated = generator.generate(np.random.default_rng(seed), config.topology.params)
        figure = plot_topology(generated, args.output)
        import matplotlib.pyplot as plt
        plt.close(figure)
        print("Topology saved: {} (nodes={}, edges={})".format(args.output, generated.graph.number_of_nodes(), generated.graph.number_of_edges()))
        return 0
    except (ConfigError, ValueError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
