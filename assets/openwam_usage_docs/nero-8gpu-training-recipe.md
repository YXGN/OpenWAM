# Nero：8 × A100 80GB 训练配置与审计

日期：2026-09-17。目标是在保持官方微调算法与 global batch=128 的前提下，提高训练吞吐，并具备周期中断恢复能力。本配置是当前证据支持的推荐起点；8 卡 GPU 吞吐尚未实测，不能称为已测得的全局最优。未启动正式训练。

## 推荐配置

入口为 `configs/train_nero_8gpu.yaml`，继承官方 `train.yaml`，选择独立 `nero_multitask` reader。Foundation 默认指向项目的 `models/OpenWAM-Alpha-Pretrain-Foundation-Model`，可由 `OPENWAM_FOUNDATION_DIR` 或 `training.finetune_ckpt_path` 覆盖；当前本地入口已核验官方权重哈希。

| 设置 | 值与原因 |
|---|---|
| 模型 | 官方 dual_system / joint_self_attn，Wan2.2 TI2V 5B + ActionDiT |
| 更新范围 | Video DiT + ActionDiT + proprio encoder；冻结文本编码器和 VAE |
| 每卡 batch / 累积 | 16 / 1；8 卡 global batch=128，保持官方配方的优化尺度 |
| 训练预算 | 5 epochs、max_steps=null、debug=false |
| 优化器 | AdamW，lr=1e-4，betas=(.9,.95)，weight decay=.01，clip=1 |
| 调度 | cosine + 5% warmup，最低 LR=1e-6；action/video 不单独改 LR |
| 精度与分片 | bf16、ZeRO-2；无 CPU optimizer offload |
| 激活 | 开启 gradient checkpointing，关闭 checkpoint activation offload |
| Loader | 4 workers/rank，共 32 个；从该值与 8 workers/rank 比较 |
| CPU 线程 | 启动脚本默认 OMP=2、OPENBLAS=1；允许环境覆盖 |
| 随机性 | seed=42，保留官方确定性设置；不以 seed=null 换取未经验证的提速 |
| 保存 | 每 2000 micro-steps，完整状态=true，最近 2 个保存点 |
| 监控 | 按用户要求，WandB 默认 online；训练启动后同步至 openwam-nero 项目 |

与选择同一 Nero reader 的基础训练配置比较，实际训练字段仅改变：output_path、dataset_num_workers（8→4）、save_full_states_for_resume（false→true）、keep_last_k_ckpts（1→2）。没有修改模型、reader、归一化、优化器或训练算法实现。没有启用本地 MoonCake bounded optimizer groups。

## 步数、数据覆盖及学习率

408 episodes / 834400 原始帧，完整训练窗口为 821344。一个窗口包含当前绝对 proprio、32 个未来动作、9 帧三相机视频；所有 episode 用于训练，不做任务均衡和验证集划分。

8 个真实 CPU/Gloo rank 使用当前环境的 Accelerate 执行了与正式 loader 相同的 shuffle、generator 同步、batch=16 分片：

- 每 rank 每 epoch 为 **6417 steps / 102672 windows**。
- 全局每 epoch 为 821376 次取样，覆盖全部 821344 个不同窗口，仅尾部补齐重复 **32** 次。
- 下一 epoch 的 shuffle 确实变化。不存在每张卡各自重复跑一遍完整数据的配置错误。
- 5 epochs 为 **32085 次更新**，总计 4106880 次取样（含 160 次尾部补齐）。相邻窗口重叠，不能把它解释为 82 万段独立示范。

同时使用真实 `AcceleratedScheduler` 在 8 rank 环境中走完 32085 次 scheduler update：

| 完成更新数 | 下一次更新使用的 LR |
|---:|---:|
| 0 | 7.793e-9 |
| 1 | 7.013e-8 |
| 1604 | 1e-4 |
| 6417 | 9.403e-5 |
| 19251 | 3.835e-5 |
| 32085 | 1e-6 |

当前 trainer 把调度区间乘以 world size，当前 Accelerate 对未 split 的 batch 每次调用推进同样的次数，两者抵消。SequentialLR 在现有 DeepSpeed 路径中走 AcceleratedScheduler，而不是由 DeepSpeed 自己管理的 scheduler。本次没有发现 warmup/cosine 被错误加速 8 倍的问题。测试使用 CPU 以验证调度与采样语义，不包含 NCCL、模型计算或 ZeRO 优化器状态。

`global_step/max_steps/save_steps` 在本仓库按 **micro-step** 计数。累积=1 时它等于 optimizer update；若改为 batch=8、累积=2，global batch 仍为128，但5 epochs是64170个micro-steps、约32085次更新，不能沿用32085的max_steps作为更新预算。

## 显存与资源判断

本机有 8 张 A100 80GB，GPU 之间均为 NV12，64 个物理 CPU 核/128 个逻辑 CPU，主机内存2 TiB。审计时只有5张卡完全空闲，无法验证完整8卡模型训练，也没有停止已有进程。

此前真实单卡诊断保持 bf16 / gradient checkpointing，开启 CPU optimizer offload：batch=1/4/8/16 的实际张量峰值分别约 25.0/25.6/27.4/31.4 GiB；batch=16 的缓存分配器保留峰值约34.4 GiB。诊断只说明这些形状可运行，不能替代多卡无offload的峰值。

可训练参数约6.021B，冻结参数约6.386B；所有bf16权重本身约23.1 GiB/卡。[DeepSpeed 内存估算](https://deepspeed.readthedocs.io/en/stable/memory.html)应用到可训练参数并单独加入冻结权重后，8卡ZeRO-2无offload的模型状态估算约45.5 GiB/卡。结合单卡激活差额，张量峰值预估约53.9 GiB，再预留通信、缓存和临时分配空间，规划为 **55–65 GiB/卡**。这是估算，非8卡测量结果。

选择 batch=16 而非直接改24/32的原因：16有单卡形状验证，并保持官方8卡global batch=128。增大batch会同时减少每epoch更新数，改变优化过程。没有多卡稳定吞吐和收敛证据时，不把“显存用得更满”等同于“训练更优”。也不盲目随batch线性提高LR。

## 启动、测速和选择规则

使用项目内独立的 `.venv` 环境；不再依赖借用的 Python。环境安装和版本说明见 [独立环境说明](openwam-environment.md)。Foundation 在 `models/OpenWAM-Alpha-Pretrain-Foundation-Model`，tokenizer/config 已复制到项目中，约24.8 GB的官方 safetensors 通过符号链接复用，需确保链接目标持续可读。

```bash
cd /data2/xieweize/project/OpenWAM
source scripts/activate_openwam.sh

# 仅打印最终配置，不分配 GPU、不训练。
bash scripts/train_nero_8gpu.sh --print-config

# 资源齐备后测量默认候选：8卡，每卡16，workers=4。
bash scripts/train_nero_8gpu.sh --benchmark

# 比较数据读取并行度；仍使用同一batch和模型配置。
bash scripts/train_nero_8gpu.sh --benchmark training.dataset_num_workers=8

# 正式5-epoch训练；本次未执行。
bash scripts/train_nero_8gpu.sh
```

启动器固定单机8进程，使用指定Python的`-m torch.distributed.run`，避免旧torchrun脚本的shebang指错环境。预检要求恰好8张可见80GB级GPU且每卡至少70 GiB空闲；不终止已有进程，也不自动改用单卡/其他配方。

`--benchmark` 是独立诊断：100次更新、恒定LR、关闭checkpoint保存，前10步作为预热。为了控制输出和调度语义，这些诊断覆盖固定在启动器内；正式模式仍使用5epochs+cosine。输出在 `outputs/nero_multitask_8gpu_benchmark/<timestamp>/`：

- `benchmark_steps.csv`：每步loss/grad norm，最慢rank步时，以及所有rank中最大的allocated/reserved显存。
- `benchmark_summary.json`：后90步的平均步时与全局windows/s，峰值显存覆盖整个测试。
- 计时包含数据加载与每步日志；不是仅GPU kernel耗时。CUDA显存统计不包含所有驱动/NCCL分配，测试时同时观察nvidia-smi。

先比较workers=4/8。若差异在短测波动范围内，保持4以减少CPU竞争。若batch16稳定、每卡仍有至少约8–10 GiB余量，可再测batch24/32；只有端到端windows/s稳定提升才考虑采用，并重新审视global batch与训练预算。对接近的结果重复同长度测试，不凭一次短测的小差距宣称最优。正式耗时用 `4106720 / 实测全局windows_per_second` 估算，再另加启动、保存和尾部补齐开销；不使用此前单卡CPU-offload的21秒/步外推8卡速度。

可作为显存回退的方案是每卡batch8、累积2，维持global batch128；它会增加micro-step数及部分开销，需要单独测速。若资源充足，不优先CPU optimizer offload或ZeRO-3，以免无必要地增加主机传输或参数通信。

## 保存、恢复及未解决项

完整状态开启是为降低长任务意外中断的损失。保存点同时包含独立部署权重和分片optimizer等状态，可能为百GiB量级；保留2个点时写入第3个再裁剪，磁盘必须容纳暂存峰值。模型权重每份约23.1GiB；完整保存的精确大小和耗时还需8卡验证。本机审计时可用磁盘约974GiB，当前容量足以开始这一策略，但不建议把keep_last_k无依据地调得很大。

**当前trainer正常完成仍会删除所有完整续训状态，只保留最近2份权重。** 因此完整状态用于意外中断后的周期点恢复，不保证5epochs完成后还能沿用最终optimizer继续训练。本次没有修改这一官方行为。

恢复中断运行时，必须使用包含可用`accel_state_step_*`的运行目录，并清空finetune入口：

```bash
bash scripts/train_nero_8gpu.sh \
  training.finetune_ckpt_path=null \
  training.resume_ckpt_path=/absolute/path/to/interrupted/run
```

尚待资源齐备后的验收：8卡NCCL/ZeRO-2真实训练、稳定吞吐、显存峰值、完整状态保存/加载。CPU8-rank检查与原单卡模型smoke不能替代这些项目。长期收敛/实机成功率未验证；月饼prompt的`[box destination]`含义仍未确认。

进一步提速可考虑缓存三个固定prompt的冻结文本特征，但需保证编码器eval状态、dtype、tokenizer、文本预处理与batch展开一致并做数值对齐；本次未实现。视频VAE含时序上下文，不应直接把整段latent随意切片当作窗口独立编码。

## 本次修改与验证产物

```text
configs/train_nero_8gpu.yaml                  # 正式推荐配置，继承官方默认
scripts/train_nero_8gpu.sh                    # 8卡预检、配置打印、测速/训练入口
scripts/benchmark_nero_8gpu.py                # 跨rank显存和端到端吞吐记录
assets/openwam_usage_docs/nero-8gpu-training-recipe.md
outputs/nero_8gpu_recipe_audit/
  composed.yaml / launcher_resolved.yaml     # 实际Hydra解析结果
  config_diff.json                          # 与基础配方的字段差异
  distributed_cpu_audit.py / .json / .log    # 8-rank采样和完整LR曲线验证
  benchmark_instrumentation_test.json        # 计时/峰值/预热/NaN检查的CPU mock
  no_gpu_guard.log                          # 无GPU时启动器拒绝运行
```

已通过Hydra合成与模型/数据配置一致性检查、完整8-rank CPU审计、benchmark记录器的CPU mock、无GPU启动拒绝、bash语法和Python lint/format检查。没有修改共享Python环境、没有启动正式8卡训练。
