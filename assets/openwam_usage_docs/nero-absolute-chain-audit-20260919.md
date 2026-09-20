# 绝对 EEF 四卡训练链路审计（2026-09-19）

## 当前结论

数据、归一化、训练配置及部署语义检查通过；未发现绝对/相对 EEF 混用。
首次在 GPU 1、2、5、6 的 B64/GAS1 短测完成第 1 次更新后 OOM；GPU 1 在启动前检查后新增其他用户约 8.14 GiB 占用。
该次已停止。随后用户审核换到 GPU 3、4、5、6，相同 B64/GAS1 **完成全部 24 步，四个 rank 均正常退出**。
稳定段平均 19.552 秒/步，CV 0.236%；进程显存峰值 75.10 GiB/卡，数值有限，但存在显著 VAE 卷积性能回退。
用户要求进一步定位耗时后，暂缓正式启动；已用真实四卡第 1/3 步 CUDA trace 找到慢卷积内核。
详见同目录 `nero-absolute-vae-kernel-profile-20260919.md`。所有诊断进程已结束，未改变生产 batch、累积、模型或 trainer。

## 语义和数据

- 双臂 native 20D：每臂 body 坐标系下 gripper_flange 的绝对 XYZ、绝对旋转矩阵前两列 rot6D、绝对夹爪宽度（米）。这不是 world frame，也不是逐步 delta。
- 窗口 `t..t+32`：proprio 取 t，监督动作取 `t+1..t+32`，首帧条件取 t。源 action 是当前关节状态，没有被误用为 EEF 标签。
- 左臂使用统一 80D 的 0–9，右臂 34–43；其它维度不计入动作 loss。动作 mask 为逐时间、逐维 mask，proprio mask 独立处理。
- 三路相机 `[head_fpv,left_hand,right_hand]` 对应头部上方、左腕左下、右腕右下；33 帧按 stride4 取 9 帧，384×320。首帧条件和视频共用同一读取结果。
- 408 episodes、834400 帧、821344 个完整窗口；全部用于训练，不做任务均衡或验证集划分。
- 独立重扫全部源 EEF，确认所有未来标签与当前状态都在各自 min-max 统计范围内，XYZ 极值与统计完全匹配。旋转不压缩，夹爪固定物理端点 0..0.1 米。
- 三任务 24 个真实首尾窗口再次通过对照：动作反归一化最大误差 5.96e-8；状态、任务文本、mask 与帧偏移正确；独立解码的首尾三相机拼图逐像素一致。
- 原相对统计哈希保持不变。绝对 reader 与部署加载器均拒绝相对统计；推理反归一化后不加回 anchor。

## 训练、调度和保存

- 从本地 foundation `checkpoint_step_154000.safetensors` 新微调。四个 rank 均实际加载权重成功，无旧 step 4000 恢复或数据偏移。
- 运行时 DeepSpeed 确认：4 ranks、micro batch 64、GAS1、global batch 256、bf16、ZeRO-2、GPU optimizer、梯度裁剪 1.0。
- 运行时确认 T5 与 VAE 无可训练参数且处于 eval；约 6,021,155,088 参数训练，来自 video DiT、action DiT 和 proprio encoder。冻结 VAE 仍在线计算；只有 T5 缓存。
- 模型和训练策略沿用现有官方配方及已审核性能优化。没有因卡数增加 offload、分组或其它策略分支。
- 5 epochs：每遍 3209 步，共 16045 步；每遍尾部补齐 160 个窗口。LR 1e-4，cosine，warmup802，最低 1e-6。短测虽在 24 步退出，但仍使用完整 16045 步 LR 日程。
- 短测保存到运行目录的 config、绝对统计和 tokenizer 已检查；部署 normalizer 可直接加载，输出 contract 为 absolute_body_gripper_flange。旧 backbone model_path 占位路径不被用于加载：重建依赖保存的 component specs 和实际 safetensors/tokenizer。
- 正式配方每 2000 步写完整状态，再清旧，保留最近 1 套；4000 步单独归档权重与加载资产，不含优化器。运行和归档均为独立的本地绝对 EEF 路径。
- 约 585 GiB 可用空间足以容纳两套完整状态写入重叠及独立权重归档。正常训练结束沿用现有行为，仅保留最终权重并清理完整恢复状态。
- 现有跨 epoch 中途恢复无法严格重建原 RandomSampler 顺序的限制仍存在；不影响此次从 base 连续训练，未擅自应用候选修复。

## 验证证据

CPU 回归：146 passed、2 skipped；跳过的是需显式开启的通用 GPU 训练/部署集成测试，本次另有真实四卡短测。
覆盖绝对/相对 reader、归一化、slot mask、损失、冻结、T5 缓存、finetune 合并、保存、调度器和 dataloader seed。

真实四卡第 1 步：所有 rank 的 optimizer_step 均为 1；total loss 0.14420247，video 0.13695079，action 0.00725167，grad norm 0.35112053，均有限。
这只能说明首步前向、反向与更新可以执行，不代表稳定性能或任务成功率。

环境：独立 `.venv`，PyTorch 2.7.1+cu126、cuDNN 90501。
W&B 已核实登录账号 `xxxwwwzzz77`、默认空间 `xxxwwwzzz77-shenzhen`；短测关闭上传，正式启动器配置 online 独立 run。

证据目录：`outputs/nero_absolute_validation/20260919_4gpu_b64/`：

- `smoke.log`、`smoke.exit`、`smoke_summary.json`：运行日志、退出状态与未通过原因。
- `runtime_rank*.json`：各 rank 的实际 DeepSpeed 配置、冻结和参数数目。
- `stages_rank*.jsonl`、`runs/*/telemetry_rank*.jsonl`：首步分段和数值。
- `full_range_audit.json`：全量 EEF 范围审计。
- `launch_training.sh`：指定四卡、B64/GAS1 的正式启动入口；必须先有通过的短测报告才允许启动。

真实 reader 抽查：`outputs/nero_absolute_preparation/reader_audit.json`。
本轮仅写审计/诊断产物，没有修改生产模型、reader、trainer 或正式训练配置。
换卡成功短测证据在 `outputs/nero_absolute_validation/20260919_gpu3456_b64/`，内核 trace 在 `outputs/nero_absolute_validation/20260919_gpu3456_profile/`。
