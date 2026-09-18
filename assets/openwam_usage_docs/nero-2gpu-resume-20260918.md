# Nero 两卡恢复训练（历史记录，兼容配方已撤回）

按用户要求，已移除双卡专用CPU优化器卸载、为保持global batch96而设置的梯度累积、
CPU AdamW兼容分支及相关启动/转换脚本。以下内容仅记录过去的运行，不再是可执行配方。
旧权重、完整状态和日志保留，但旧CPU卸载/GAS6保存点未被转换为新的GPU/GAS1配方，
不能直接据此恢复；本次清理没有启动训练或修改任何保存点。

使用物理 GPU 6、7，tmux 会话 `openwam-nero-2gpu`，WandB online。
原六卡运行和保存点保持不变；恢复到独立目录：
`outputs/nero_multitask_2gpu/resume_step4000`。

## 配方

| 配置 | 值 |
|---|---|
| 每卡 microbatch | 8 |
| 梯度累积 | 6 |
| 全局 batch / optimizer update | 2 × 8 × 6 = 96 |
| 精度 / 分片 | BF16 / ZeRO-2 |
| 优化器 | 原 torch AdamW，FP32 主权重与状态卸载到 CPU |
| 初始学习率 / 调度 | 原 1e-4 / warmup + cosine；恢复时 LR 约 9.948868e-5 |
| 梯度节点复用 | 开启，逐 hook 慢速审计关闭 |
| VAE / T5 | 仍在线编码，未启用缓存 |
| DataLoader | 每 rank 4 workers |
| CPU 计算线程 | 每 rank OMP_NUM_THREADS=8 |
| 总训练计划 | 5 epochs，约 42780 次 optimizer update，包含原 4000 次 |
| 保存间隔 | 12000 microsteps = 2000 次 optimizer update |

恢复时 GPU7 有其他进程，约占 31 GiB。原 batch16/GAS3 方案被空闲显存检查拦下，
改为 batch8/GAS6 保留全局 batch。启动要求每卡至少 44 GiB 空闲，训练分配器额外保留 3 GiB。
不会终止其他作业；其他进程继续增加显存仍可能导致训练 OOM。

CPU 卸载是本次两卡资源条件下的选择，会引入 CPU 更新和主机/设备传输开销。
这次两卡吞吐不能作为六卡纯 GPU 优化后的吞吐结论。

## 完整状态迁移

源：`outputs/nero_multitask_6gpu/2026-09-17_18-22-36/accel_state_step_4000`。
DeepSpeed 0.18.5 的 rigid ZeRO-2 不支持直接从 6 ranks 加载到 2 ranks，
因此用 `scripts/convert_nero_zero2_6to2.py` 在独立目录重新分片。

- 限制为原有单参数组、无 padding、无 round-robin 的 ZeRO-2 AdamW 保存点。
- 六个分片合并成两个，每个新分片 3010577544 个参数元素。
- FP32 主权重、exp_avg、exp_avg_sq 写出后与原分片逐元素比较，全部一致。
- 保留 optimizer step=4000 和已处理 global_samples=384000。
- 新 trainer global_step / Accelerate accumulation step = 24000（4000 × 6）。
- CUDA RNG 状态数量改为两卡；rank/microbatch 随机数流变化，不能逐位复现原六卡随机轨迹。
- SequentialLR 按 world-size 比例换算，接下来 1000 次更新的实际 LR 对照误差小于 1e-16。
- 原归一化统计和模型组件配置保留，不重新计算数据统计。

本次状态转换最初使用 batch16/GAS3，随后将 trainer/Accelerate microstep 调整为 batch8/GAS6。
转换后的 DeepSpeed 文件内部 ds_config 仍记录最初 16/3；该字段是保存时的元数据，
实际引擎使用当前配置的 8/6（全局 batch 都是96）。迁移说明在 `migration.json`。
转换脚本现已支持 `--micro-batch 8`，重跑新目录时会同步生成正确的 ds_config 元数据。

最新持久化进度是第4000次更新，暂停前未保存部分需要重跑。
本次恢复点命名为 `accel_state_step_24000`，下一次常规保存是 microstep36000 / optimizer step6000。

## 运行与监控

历史tmux会话为 `openwam-nero-2gpu`，已停止。
旧启动器 `scripts/train_nero_2gpu.sh`、配置 `configs/train_nero_2gpu.yaml`
及转换器 `scripts/convert_nero_zero2_6to2.py` 已移除。
训练日志：`outputs/nero_multitask_2gpu_launch/train.log`。
WandB：https://wandb.ai/xxxwwwzzz77-shenzhen/openwam-nero/runs/gyccs85e。

`scripts/train_nero_monitored.py` 通过现有 log_step 记录每 rank 的 microstep 时间、
optimizer step、loss、LR、显存和训练主进程 CPU 使用率，不添加 CUDA 同步。
记录位于运行目录 `telemetry_rank{0,1}.jsonl`。

`scripts/monitor_nero_resources.py` 每2秒记录两卡整卡指标及本次训练进程树，
写入 `outputs/nero_multitask_2gpu_launch/resources.jsonl`。GPU 利用率包含其他作业；
进程显存可以区分归属。CPU 百分比按一个逻辑核=100%计，允许超过100%。
吞吐对比应使用完整6个 microstep 的 optimizer update 耗时，或样本/秒。

## 恢复后的稳定窗口实测

已观察到 optimizer step4006 / microstep24036。排除首个114.44秒的更新周期后，
连续5个完整更新周期覆盖约367秒，耗时为76.42、71.87、73.72、67.80、77.26秒。
平均73.42秒/更新，变异系数4.63%，吞吐约1.31窗口/秒。
普通microstep平均8.69秒；包含优化器更新的第6个microstep平均29.93秒。
这里没有对CPU更新单独做同步计时，不能把两者差额全部视为精确的AdamW耗时。

训练进程各占30.40 GiB显存；PyTorch峰值allocated分别27.36/28.73 GiB，reserved均28.78 GiB。
本次训练进程树CPU平均占用约3.32个逻辑核（整机128个逻辑核）。
GPU6/7整卡利用率平均86.31%/92.97%，但其他进程分别约占2.21/30.63 GiB，
因此整卡利用率不代表本训练独占计算效率，也不能证明CPU/GPU无等待。

观察窗口内无OOM、无非有限loss；这只是稳定性/吞吐观察，不是策略成功率评估。
状态及原始汇总：`outputs/nero_multitask_2gpu_launch/stable_summary.json`。
CPU融合AdamW候选微测更快，但出现FP32舍入差异，本次正式训练未切换该候选。
