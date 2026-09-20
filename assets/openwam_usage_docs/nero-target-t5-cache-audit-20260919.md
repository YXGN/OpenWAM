# 目标机 T5 缓存与八卡训练审计

目标项目：`robot-server:/data2/xieweize/project/OpenWAM`。

## 改动范围

- `training.cache_frozen_text_embeddings` 默认关闭，仅目标八卡配方开启。
- Wan 训练预处理可复用冻结 T5 的 `context` 和 `seq_lens`。首次遇到未缓存文本时仍编码原始完整 batch；随后按完整文本取缓存，保持重复文本、样本顺序、padding 和序列长度。
- 缓存只存在于各 rank 内存，三条任务文本共 12,582,936 bytes（约 12 MiB）。返回独立张量，避免下游原地操作污染缓存。
- 编码器必须被冻结且所有子模块处于 eval 模式，否则报错。设备、精度或 autocast 设置变化时清空缓存。
- 不缓存 VAE、proprio、扩散噪声或 timestep；不删除冻结 T5 权重。保存点格式不变，重启后自动重新填充缓存。
- 不改变 batch、累积、学习率、损失函数、梯度检查点及已有 DeepSpeed 梯度节点优化。

## 数值验证

证据位于 `outputs/target_t5_cache_validation/`，实际使用目标 base 权重和三个任务的真实训练窗口。

1. batch64、重排 batch64、batch3：缓存和在线 T5 输出全部逐元素相等，最大绝对差 0；序列长度相等，CPU/CUDA 随机数状态不变。
2. batch64 热缓存约 0.00431 秒，在线编码约 2.36130 秒；冷缓存第一次仍需正常编码。
3. 全模型的 prepared tensor、首帧条件、视频/action/总 loss 均逐元素相等。
4. 首次普通反向对照的梯度相对 L2 差异为 0.00526135，不能直接宣称逐位一致。随后在线路径重复运行的对照也出现 0.00736303 的差异，确认既有反向计算存在数值非确定性。
5. 仅在诊断中开启 `torch.use_deterministic_algorithms(True)` 后，在线与缓存的全部 1651 个有梯度张量逐元素相等，最大绝对差和相对 L2 差均为 0。正式训练不启用该诊断选项。

这支持缓存保持当前训练语义，不等于保证普通非确定性 GPU 训练的最终权重逐位相同，也不是任务成功率评估。

## 配方与链路

- 从 `/data2/public/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors` 新微调，步数从 0 开始，无旧保存点或数据偏移。
- 八卡 × 每卡 64 × 累积 1 = 全局 512。821344 个窗口，每遍 1605 步，共 5 遍、8025 次更新；每遍尾部补齐 416 个窗口。
- GPU AdamW、BF16、ZeRO-2；LR 1e-4，cosine，401 步 warmup，最低 LR 1e-6。调度器补偿 Accelerate 每次更新的多 rank 内部 tick。
- 33 原始帧、32 个未来动作、stride4 后 9 帧视频，三相机布局为 head 上方、left 左下、right 右下，384×320。
- 双臂动作：相对锚点 xyz（body 轴）及 `R_anchor.T @ R_future` 的 rot6D，夹爪绝对宽度；proprio 为锚点绝对状态。动作与状态分别归一化，左臂槽位 0–9、右臂 34–43，其余由 mask 排除。
- 正式读取器校验数据指纹、统计契约、相机和槽位映射；随机颜色增强关闭，不做任务均衡或验证集划分。
- 完整状态每 2000 步保存到 `/data3/xieweize/openwam_runs/nero_multitask_8gpu_b64/<运行时间>`，先写新状态再清旧，保留最近一套。两套并存约 250.8 GiB。
- 第 4000 步另复制约 23.1 GiB 的独立权重及加载资产到 `/data2/xieweize/project/OpenWAM/outputs/nero_multitask_weights/<运行时间>/step_4000`，不含优化器且不参与滚动清理。
- 正常训练结束沿用原逻辑清理完整恢复状态、保留最终权重；第 4000 步独立副本不受影响。
- 启动脚本 `scripts/train_nero_target_8gpu.sh` 固定八卡，W&B online，账号 `xxxwwwzzz77`，使用该账号 API 返回的默认空间 `xxxwwwzzz77-shenzhen`，项目 `openwam-nero`。

## 跨 epoch 恢复的已知限制

原 RandomSampler 不能在第二遍及以后中途恢复时完整重建原样本顺序。从 base 连续训练不受这个恢复问题影响，但中断恢复的轨迹不保证一致。
此前候选修复涉及 SeedableRandomSampler 和 skip-first 后设置 epoch，仍单独等待用户审核；T5 缓存不会自动修复该问题。

## 回归与八卡实测

CPU 回归累计 109 passed / 2 skipped；其中缓存、Nero 读取、归一化、保存、seed 和调度器 55 项也在目标环境通过。
八卡短测每个 rank 完成 24 次真实更新，退出码 0；第 5–24 步取每步最慢 rank：

- 平均 11.4367 秒/步，范围 11.1705–11.8576，CV 1.72%；44.768 窗口/秒。
- 相比未缓存的 13.8117 秒/步，步时减少约 17.2%，吞吐提高约 20.8%。
- rank0 分段：T5 热缓存 0.00222 秒，像素处理 0.8373 秒，VAE 2.4809 秒，前向/loss 1.8645 秒，反向 6.0313 秒，DeepSpeed 更新 0.1707 秒。CUDA event 区间和整步墙钟定义不同，分段不能要求精确求和。
- 每卡训练进程峰值 74.49 GiB，PyTorch allocated 64.12 / reserved 72.87 GiB；各卡利用率均值 91.3–92.9%。
- 全部 rank 均只有一次 T5 编码，缓存三条文本，显存分配器重试 0。短测不保存模型、不上传 W&B；使用完整 8025 步 LR 日程，只限制退出步数。
- 纯计算预计 25.49 小时，计入初始化、保存和波动，完整训练约 26–28 小时，仍是短测外推。

正式训练 tmux 会话：`openwam-nero-8gpu`。日志：`outputs/target_t5_cache_validation/training.log`。
正式运行路径：`/data3/xieweize/openwam_runs/nero_multitask_8gpu_b64/2026-09-19_00-15-03`。
W&B：https://wandb.ai/xxxwwwzzz77-shenzhen/openwam-nero/runs/bclr6ixw 。登录账号仍为 `xxxwwwzzz77`。

正式配置落盘后再次检查通过：八卡 B64/GAS1、5 epochs、无短测步数上限、fresh base、T5 缓存、2000 步保存及 4000 步归档、相机布局和动作槽位、归一化文件及 tokenizer 资产全部符合预期。

初期监控快照（八个 rank 均完成第 22 步，训练继续运行）：

- 第 5–22 步最慢 rank 均值 11.4084 秒，范围 11.1865–11.8271，CV 1.67%；44.88 窗口/秒。
- 训练进程显存峰值 74.49 GiB/卡，CPU 进程树均值约 16.94 个逻辑核，各卡 GPU 利用率均值 91.8–97.2%。
- 第 22 步 total loss 0.205596，video 0.131098，action 0.074498，grad norm 1.988963，LR 5.51575e-6；全部有限。训练仍处于 warmup，不能用这些短期 loss 代替任务成功率评估。
- 原采样器保留，未应用未获审核的跨 epoch 恢复补丁。

查看 tmux：

```bash
ssh -t robot-server 'tmux attach -t openwam-nero-8gpu'
```

启动脚本将输出写入日志；查看滚动日志可在目标机执行：

```bash
tail -f /data2/xieweize/project/OpenWAM/outputs/target_t5_cache_validation/training.log
```

按 `Ctrl-b` 再按 `d` 离开 tmux，不停止训练。资源采样覆盖正式启动后前 30 分钟；各 rank 的逐步 telemetry 会继续写入运行目录。
