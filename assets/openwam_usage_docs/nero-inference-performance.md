# Nero 推理优化引入与验证

来源：YXGN/OpenWAM `codex/inference-performance`，提交
`f537c57e9e80e53d05a73aca9932d8b969681acc`。在现有 Nero 链路上引入，
保留相对 EEF 的 anchor 还原、绝对 EEF 输出、相机布局、归一化和客户端合同。

## 实施方案

1. 复用源分支的静态 inactive-action 索引、请求内 context/KV 复用和 MoT CUDA Graph。
2. 通用引擎缺少总开关时保持 eager；Nero 的共享 `inference_config` 显式开启三项优化，
   服务端和开环评估共用该配置。保持 10 步、CFG=1、BF16、无近似 DiT cache、不解码视频。
3. 使用真实录制观测、固定 seed、相同权重，依次对照 eager、静态索引、context/KV、完整图路径。
   覆盖三任务、不同 anchor、相同图像下 prompt 和 state 切换、A→B→A，以及相对/绝对权重。
4. 同时比对归一化输出、原生动作和对外绝对 EEF；检查实际图重放与缓存命中。
   分开记录冷启动和稳定延迟、显存；功能回归通过后维护本地提交。

不重新训练、不修改权重，不自动重启正在连接机器人的服务。保留训练工作区的独立改动。

## 开关和适用边界

```yaml
optimization:
  inference_fast_path: {enabled: true}
  request_context_cache: {enabled: true}
  cuda_graph: {enabled: true}
  compile: {enabled: false}
  dit_cache: {enabled: false}
  decode_video: false
  prompt_embed_cache: {enabled: true, maxsize: 32}
```

Nero 原启动命令继续适用；追加 `--disable-inference-fast-path` 可做 eager 对照。
启动日志中的 `fast_path=True` 只代表配置开启；推理日志的 `backend=cuda_graph` 表示
该次最后一个去噪步实际重放。完整诊断可读 `architecture._mot_cuda_graph_runner.status()`。

请求内缓存每次 generate 重新创建，只复用本请求的文本/状态 context 和逐层 cross-attention K/V。
T5 prompt 缓存仍按原文本管理。变化的视觉/动作 self-attention K/V 不会跨去噪步复用。

CUDA Graph 只捕获每个去噪步的联合 Transformer 层循环；图片处理、VAE、T5、状态准备、
scheduler 和动作还原仍在图外。首次捕获含三次预热；shape/stride/dtype/mask 模式不匹配走
eager，捕获错误直接报错。切换权重和 device/dtype 会清图；不支持与 torch.compile 同开。

## 验证入口

`scripts/audit_nero_inference_performance.py` 使用现有服务加载器、观测读取和预测记录器，
不连接机器人。结果保存完整数值对照、计时、配置身份、图状态和输出数组。
复现命令（从项目根目录运行，选择空闲 GPU，output 必须是新目录）：

```bash
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/audit_nero_inference_performance.py \
  --ckpt-dir models/nero_relative_8gpu_b64_20260919/step_4000 \
  --ckpt-name checkpoint_step_4000.safetensors \
  --dataset /path/to/nero_multitask_lerobot_v2 \
  --output outputs/nero_inference_performance/new_run
```

每个模式重建优化缓存，先记录首请求，再预热全部任务，最后对同一输入计时 10 次。
耗时包含引擎内 VAE/T5、10 步去噪、归一化/还原和 CPU 动作返回，不含 HTTP、图片解码
与网络。首请求不含权重加载；各模式共用一个进程，因此不应将首请求列理解为独立进程
冷启动的严格消融。稳态计时才用于性能比较。

固定 seed=42；每份权重验证三任务的 9 个真实 anchor，再加相同观测只切换 prompt、
相同图像只切换 state、回到首个 A 的探针，共 12 组。每组同时保存对外绝对 EEF20、
反归一化后的原生 EEF20、归一化的有效 20 槽位，要求逐元素完全相同，不放宽浮点阈值。

## 2026-09-20 相对 step4000 实测

A100 80GB，PyTorch 2.7.1+cu126，BF16，OMP=8，GPU 2：

| 模式（逐项叠加） | 稳态中位数 | P95 | 峰值 allocated | 峰值 reserved |
|---|---:|---:|---:|---:|
| 原 eager | 789.7 ms | 806.4 ms | 23.38 GiB | 23.68 GiB |
| 静态索引 | 770.3 ms | 786.1 ms | 23.38 GiB | 23.68 GiB |
| 加 context/KV | 709.4 ms | 716.9 ms | 23.59 GiB | 23.74 GiB |
| 加 CUDA Graph | 455.7 ms | 456.6 ms | 23.98 GiB | 24.31 GiB |

完整路径耗时下降 42.3%，吞吐约 1.73 倍；allocated 峰值增加约 0.60 GiB。
图捕获 1 次、重放 350 次；每请求 context 准备 1 次、命中 9 次。三层输出全部精确相等，
prompt/state 改变均会改变输出，回到 A 和连续重复 A 均精确复现。

证据：`outputs/nero_inference_performance/relative4000/{result.json,predictions.npz}`。
数值等价只覆盖这些权重、观测和配置，不替代真实机器人任务成功率评估。
同步 receding-horizon 仍在动作耗尽后发起请求；低于动作块跨度并不等于推理与执行已重叠。

## 2026-09-20 绝对 step4000 实测

相同软件与推理参数，GPU 4 独占测试：

| 模式（逐项叠加） | 稳态中位数 | P95 | 峰值 allocated | 峰值 reserved |
|---|---:|---:|---:|---:|
| 原 eager | 768.0 ms | 771.0 ms | 23.38 GiB | 23.68 GiB |
| 静态索引 | 760.9 ms | 762.2 ms | 23.38 GiB | 23.68 GiB |
| 加 context/KV | 698.0 ms | 707.7 ms | 23.59 GiB | 23.74 GiB |
| 加 CUDA Graph | 453.1 ms | 454.5 ms | 23.98 GiB | 24.31 GiB |

完整路径耗时下降 41.0%，吞吐约 1.69 倍。图捕获 1 次、重放 350 次；12 组输入的
三层输出均精确相等，prompt/state 新鲜度与重复 A 检查通过。

证据：`outputs/nero_inference_performance/absolute4000_gpu4/{result.json,predictions.npz}`。
此前 GPU 2 的 `absolute4000/` 也通过数值检查，但期间同卡新增其他服务，
其耗时不作为正式性能依据。测试期间其他 GPU 上仍有训练负载；上述数字不是空机器保证值。

## 代码验证范围

- CUDA Graph、context/KV、静态索引和 Nero 基础回归：GPU 实跑 109 项通过、8 个子测试通过。
- 扩展 MoT、proprio、归一化、Wan 和其他架构消费者回归：200 项通过、8 个子测试通过，
  1 项需要额外 CosmosPredict25 安装及 CUDA 的测试跳过。
- 最后补充显式配置、实际 backend 日志与 HTTP 回归：38 项通过。
- Ruff 与 diff 空白检查通过；两仓 Quickstart 字节一致、7 段 bash 语法检查通过，
  客户端文档布局与拼写检查通过。

以上测试集合存在重叠，不相加作为独立测试数。真实硬件控制与任务成功率不在此次审计范围。

## 最终权重的客户端联合验证

复用 `scripts/audit_nero_prompt_switch.py`，相对 step8025 权重、GPU 4，走真实
InferencePanel → Controller → 同一 ObservationBuilder → HTTP → 官方引擎/T5。
优化已由 Nero 共享配置启用。固定观测下 A（锤子）→B（插头）→A 通过，
服务端与模型收到的原文一致；T5 实际编码 A、B 各一次，返回 A 使用原有 T5 缓存。
重复 A 的动作精确一致；运行中改任务被拒绝，停止清理能使旧 A 的在途结果失效，
录制页任务标签仍不改变模型 prompt。

证据：`outputs/nero_inference_performance/client_prompt_8025/{result.json,predictions.npz}`。
使用录制的 Soma 观测，机器人运动回调为桩；没有启动硬件，也没有声称真机闭环完成。
