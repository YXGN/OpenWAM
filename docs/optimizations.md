# 推理与训练优化

## 来源

用户的 [inference-performance 分支](https://github.com/YXGN/OpenWAM/tree/codex/inference-performance) 固定到 `f537c57e9e80e53d05a73aca9932d8b969681acc`，已由本地 `e7513d69f9f94a7017ddf7f552849f52a8a7b2e0` 适配到本次导入基线。现有请求内 projected K/V、静态 action 索引、MoT CUDA Graph 和 eager 回退继续保留，没有用旧分支覆盖后来加入的 Nero 实现。

训练改进来自 [官方 PR #58](https://github.com/OpenWAM-Official/OpenWAM/pull/58)，净变更范围 `3950b6d9… → 66c8d89e…`，官方合入提交 `0c9ed80d…`。采用最终修正版，来源于 [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent) 的加速工作，不需要安装该 agent 项目。完整 SHA 与适配记录见 optimization_sources.json。

## 已接入能力

- CPU 图像保留 uint8、pinned/nonblocking 传输，减少输入搬运。
- 文本编码缓存、miss 去重、mask/长度优化、RoPE 缓存。
- 可选 cuDNN SDPA、fused RoPE、拼接与 block 编译、Wan VAE 全片段与 channels-last。
- 按前 N 层重计算、冻结文本编码器 CPU offload，及 GC/NUMA/DeepSpeed 开销选项。
- DeepSpeed 特定版本绕过 engine.forward 时的 GAS loss 缩放修正；该正确性修复不依赖 fast preset。

本地融合保留预编码视频跳过 VAE、旧文本缓存，以及设备/dtype/autocast/encoder 更换后的缓存失效。fused block 在无梯度推理中不接管已有 projected K/V 路径，避免训练优化绕开推理缓存。原生 Nero 部署配置、协议、CUDA Graph runner 保持原值。

## 使用

默认 `training.speed_preset=null`。新 recipe 可设 `"speed_preset": "fast"`；直接使用原生训练时，在任务 YAML 中设置：

```yaml
training:
  speed_preset: presets/fast.env
```

preset 是字面 KEY=VALUE 文件，经 Python 解析，不执行任意 shell。文件值覆盖同名环境变量。`null` 不会清除手动设置的 OPENWAM_*；要评估基线，应使用干净进程环境，不在推理服务环境中 source 训练 preset。

重计算与 offload 仍由 YAML 单独控制：`training.use_gradient_checkpointing`、`gradient_checkpointing_layers`（null 全部，0 关闭对应层）和 `text_encoder_offload`。T5 offload 需开启文本缓存；减少重计算会提高显存占用。fast 不代表自动适合 4090/A100，保持默认关闭，按实际硬件验收。

## 验证范围

移植后的隔离副本 CPU 回归 **2453 passed，19 skipped，40 deselected，8 subtests passed**；跳过的 Docker、CUDA 和未配置外部客户端测试不算验证通过。正式整理目录加入训练入口测试后，2488 项通过，另 5 项回环 HTTP 测试在解除沙箱监听限制后通过；跳过/未选数量相同。通过测试覆盖缓存兼容、GAS、配置及 CPU 合同，没有加载真实权重或运行训练。

官方 2.54–2.77 倍数据来自四张 H200 的特定七任务配置，不能推断为本机收益。真实 GPU 的数值对比、CUDA Graph replay、峰值显存、吞吐及训练质量仍需对应硬件验证。
