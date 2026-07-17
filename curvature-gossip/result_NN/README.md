# Neural-network results

本目录只保存局部 AF3、双层广播约束和 CTDE 神经策略产生的配置副本、训练历史与模型 checkpoint。
原有启发式结果继续保存在 `results/`，训练脚本不会写入或覆盖该目录。

快速验证：

```powershell
conda run -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/nn_ctde_smoke.yaml
```

正式训练：

```powershell
conda run -n GRL_AoI_cpu37 python scripts/train_nn_ctde.py --config configs/nn_ctde_train.yaml
```

每次训练会生成配置副本、训练历史、模型元数据和 checkpoint。标准 runner 的
NN 评估结果写入 `result_NN/evaluation/`。
