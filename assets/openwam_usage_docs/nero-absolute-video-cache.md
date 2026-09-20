# 绝对 EEF 的完整窗口 VAE 缓存

保持原训练配方：四卡、每卡 B64、梯度累积 1、5 epochs / 16045 步、原 LR、每 2000 步保存完整状态并保留最近 1 套、第 4000 步额外归档不带优化器的权重。
T5 沿用已有的冻结输出缓存。VAE 缓存只替换 RGB 解码、像素预处理和在线 VAE 编码；动作、proprio、mask、绝对 EEF 统计、采样顺序、噪声和 timestep 保留原路径。

## 数值边界

保存和读取 bf16 latent 无损。真实 Foundation 模型 B64 审计中，同样的快速 VAE 编码路径下，在线与正式缓存入口的所有准备张量及条件标志逐位相等。
这不等于与旧的显存受限在线路径逐位相等：此前 6144 窗口对照中，未来 latent 相对 L2 最坏约 0.41%；固定权重、同一噪声、每任务一例的三样本审计中梯度相对 L2 约 7.96%，总 loss 变化约 0.037%。
这不改变数学目标、数据语义或模型参数，但会改变数值优化轨迹；没有任务成功率评测来保证最终效果完全不变。完整证据见 [隔离对照报告](nero-absolute-latent-cache-audit-20260919.md)。

## 实现

```text
openwam/dataloader/utils/nero_video_cache.py  # 完成清单、数据/预处理/环境指纹、VAE 权重身份、分片读取
openwam/dataloader/nero_multitask_absolute.py # 可选缓存，仅跳过相机解码，仍按原窗口生成标签/状态
openwam/model/architectures/base.py          # 将缓存 latent 和固定几何信息交给 backbone，拒绝混合批次
openwam/model/video_backbone/wan_backbone.py # 可选原生 Wan2.2 TI2V latent 入口；原在线路径保留
openwam/train/openwam_trainer.py             # 冻结后验证实际 VAE 权重与缓存匹配
scripts/data/preencode_nero_video.py          # torchrun 分片预编码、断点继续、校验并发布完整清单
configs/train_nero_absolute_cached_4gpu.yaml  # 继承原 B64/GAS1 配方，仅启用缓存并区分 W&B 名称
scripts/train_nero_absolute_cached_4gpu.sh    # 明确指定四卡，保留每卡 77 GiB 显存启动检查
tests/dataloader/test_nero_video_cache.py    # 覆盖缓存读写、缺失/过期检查、标签保持与首帧条件
```

- 每个窗口 33 个原始状态，视频取 9 帧；相机顺序仍是 head_fpv、left_hand、right_hand，画布 384×320。
- 必须每个窗口独立编码，不能从整段 episode 编码结果中切片；VAE 的因果历史在每次 encode 开始重置。
- 每个窗口 latent 为 bf16 `[48,3,24,20]`；编码 batch 始终 64，最后不足一批时仅填充编码输入，再丢弃补齐输出，不改变训练样本数。
- 每片默认 512 个窗口，使用 safetensors 保存。源数据、相机布局、窗口、代码、PyTorch/cuDNN/Pillow/PyAV 版本和实际 VAE 权重共同约束缓存身份。
- 每片写临时文件并原子发布，SHA256 与计数校验通过才写完成记录。全量覆盖和全部分片校验通过才发布 `manifest.json`。
- reader 拒绝部分缓存、源数据/预处理不一致、缺片、尺寸/精度错误；模型加载后再验证冻结 VAE 权重。不会在缺失时偷偷在线编码或替换样本。
- 重启构建会校验并跳过已完成分片。同一缓存目录用文件锁避免两个构建器并发写入。
- 预编码如发生显存分配重试会停止，避免因内存压力静默生成另一种卷积回退路径的结果。不会自动改变 batch、卡数或训练策略。
- 全量 latent 本体约 105.74 GiB，清单与分片头另占少量空间。构建时预留 300 GiB 供后续训练保存。

## 本次执行

tmux：`openwam-nero-absolute-cache`，GPU：**1、3、5、6**。

```bash
tmux attach -t openwam-nero-absolute-cache
cat outputs/nero_absolute_validation/20260919_cache_production/pipeline.stage
tail -f outputs/nero_absolute_validation/20260919_cache_production/preencode_full.log
```

流程：全量编码 → 完整 cache reader 校验 → 24 步四卡正式缓存路径短测 → W&B 账号/磁盘检查 → fresh base 正式训练。
任何一步失败都会停止，不会改用其他 GPU、缩小 batch 或绕过保护；tmux 保留退出窗格，退出码记录在 `pipeline.exit`。
短测保持完整 16045 步 LR 日程，仅设置 24 步退出上限，关闭 W&B 和模型保存。正式训练启用当前账号 `xxxwwwzzz77` 的 W&B online。

编码目录：`outputs/nero_video_cache/`，完成前只有 `build.json`、分片与各 rank 进度，不能用于正式训练。
正式训练保存路径沿用 `outputs/nero_absolute_4gpu/<时间>/`，单独归档沿用 `outputs/nero_absolute_weights/<时间>/step_4000/`。
训练日志：`outputs/nero_absolute_validation/20260919_cache_production/training.log`。

用户随后授权改用 1、3、5、6：已停止原 0、1、5、6 预编码进程，复用已完成分片在新四卡断点续算。预编码、后续短测和正式训练统一使用新卡号；仍执行每卡 77 GiB 启动检查。当前 tmux 运行的是预编码，不应视作正式训练已经开始。换卡记录见 `outputs/nero_absolute_validation/20260919_cache_production/gpu1356_switch.json`。

## 已完成检查

- 首批 2048 个窗口在原 0、1、5、6 四卡上编码/校验成功，allocated 峰值约 33.18 GiB/卡。
- 真实 Foundation B64 原始输入与新缓存输入逐元素相等，VAE 权重身份检查通过；仅输入审计，无权重更新。
- CPU 回归：28 passed、2 skipped；跳过通用 GPU train/deploy 集成测试，本次另做上述真实 GPU 输入验证。
- 隔离原型的四卡 B64/GAS1 24 步验证：19.62 → 8.11 秒/步，进程显存峰值最高约 71.0 GiB/卡。正式全量 cache 路径仍需流水线中的独立短测，不能把原型结果冒充其完成结果。
- Python lint、编译检查、shell 语法和 diff 空白检查通过。

重启全量构建：

```bash
source scripts/activate_openwam.sh
CUDA_VISIBLE_DEVICES=1,3,5,6 python -m torch.distributed.run --standalone --nproc_per_node=4 \
  scripts/data/preencode_nero_video.py --output-dir outputs/nero_video_cache
```

编码完成并满足检查后，独立启动正式配方：

```bash
CUDA_VISIBLE_DEVICES=1,3,5,6 WANDB_MODE=online \
  bash scripts/train_nero_absolute_cached_4gpu.sh project.wandb.entity=xxxwwwzzz77-shenzhen
```

上述命令与已在 tmux 中运行的流水线只能择一，避免重复占卡训练。
