# Nero：6 × A100 80GB 微调配置

入口：`configs/train_nero_6gpu.yaml`、`scripts/train_nero_6gpu.sh`。
继承 8 卡配方中相同的官方模型、数据读取、动作表示、归一化与优化器设置。
默认选用物理 GPU 2–7，可通过 `CUDA_VISIBLE_DEVICES` 显式覆盖。

| 设置 | 6 卡 | 8 卡 |
|---|---:|---:|
| 每卡 batch / 梯度累积 | 16 / 1 | 16 / 1 |
| 全局 batch | 96 | 128 |
| 每 epoch 更新数 | 8556 | 6417 |
| epochs | 5 | 5 |
| 总更新数 | 42780 | 32085 |
| 初始 warmup 后的峰值 LR | 1e-4 | 1e-4 |
| warmup 更新数 | 2139 | 1604 |
| 最低 LR | 1e-6 | 1e-6 |
| Loader workers | 4/卡，共 24 | 4/卡，共 32 |
| 精度 / 分片 | bf16 / ZeRO-2 | bf16 / ZeRO-2 |
| CPU optimizer offload | 关闭 | 关闭 |

保留官方每卡 batch 和学习率，没有按卡数自动缩放 LR。
6 卡的全局 batch 更小，同样 5 epochs 有约 33.3% 更多更新，因此不是与 8 卡严格等价的优化过程。
6 个相同大小的 rank batch 配合整数梯度累积不能恰好构成 global batch=128；
本配方明确使用96，不通过不等长 rank batch 改写官方训练循环。

全部 821344 个完整窗口都用于训练；6 卡每个 epoch 取样 821376 次，尾部补齐重复32次。
5 epochs 总计 4106880 次取样（含160次补齐）。不新增验证集、任务均衡或统计文件。

6 路 ZeRO-2 的单卡 optimizer 分片比8路更大，静态模型状态约49.3 GiB/卡，
训练整体此前规划约60–70 GiB/卡；这仍是估算，不是6卡实测。
启动器要求恰好6张可见80GB级卡、每张至少75 GiB空闲，以保留额外余量。
不因使用更少的卡而提高每卡 batch。

## 使用

```bash
cd /data2/xieweize/project/OpenWAM
source scripts/activate_openwam.sh

# 只解析配置，不启动 GPU。
bash scripts/train_nero_6gpu.sh --print-config

# 100 步诊断：恒定LR、关闭权重保存、记录各rank峰值显存和吞吐。
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash scripts/train_nero_6gpu.sh --benchmark

# 正式训练，约42780次更新。
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash scripts/train_nero_6gpu.sh
```

当前已有 `openwam-nero-8gpu` 的自动等待任务。本次只新增配置，不切换该任务。
如果决定改跑6卡，应先取消8卡等待任务，避免之后它自动启动并竞争相同GPU。
本次未执行上述6卡GPU诊断或正式训练命令。

正式输出在 `outputs/nero_multitask_6gpu/<timestamp>/`；WandB 默认 online，
project=`openwam-nero`，run name=`nero-multitask-6gpu`。
测速输出单独位于 `outputs/nero_multitask_6gpu_benchmark`。
测速复用 `benchmark_nero_8gpu.py`，该记录器根据实际 `accelerator.num_processes`
收集数据、计算吞吐，内部不固定8卡。

每2000步保存权重及完整训练状态，保留最近2个保存点；沿用官方行为，
正常训练结束时会删除完整续训状态、保留权重。
恢复6卡中断任务时应继续使用6卡启动器：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash scripts/train_nero_6gpu.sh \
  training.finetune_ckpt_path=null \
  training.resume_ckpt_path=/absolute/path/to/interrupted/6gpu/run
```

## 配置检查

`outputs/nero_6gpu_recipe_audit/` 保存解析后的配置、与8卡配置的字段差异，
以及6个CPU/Gloo rank的真实Accelerate数据分片和完整LR调度验证。
已通过：每rank每epoch 8556步、全量窗口覆盖、尾部32次补齐、下一epoch重新打乱；
LR在第2139次更新后达到1e-4，第42780次更新后到达1e-6。
配置解析、命令行覆盖、无GPU时拒绝启动和shell语法检查也已通过。
CPU验证不包含NCCL、模型前反向或ZeRO显存行为，实际6卡性能仍需GPU诊断确认。
