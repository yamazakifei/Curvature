# Cross-N 结果整合说明

该目录是 `mpnnV3.2_ch1_cross_n80_120` 的规范结果目录，按一次逻辑训练整理：

- `stage1_*.json/csv`：来自带 SearchBase 的跨 N 校准与搜索。
- `training_history.*`：来自完整的 300 episode PPO 训练。
- `validation_history.csv` 与 `validation_per_scenario.csv`：来自完整训练过程的跨 N 验证。
- `checkpoints/best_validation`：完整训练得到的最佳验证模型。
- `integration_manifest.json`：记录来源、参数和整合规则。

实际执行分成 SearchBase 和固定参数 PPO 两个阶段；为了保持结果语义一致，未把失败的早期 partial history 与完整 history 直接拼接。
