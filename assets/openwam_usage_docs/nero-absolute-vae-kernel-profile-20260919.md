# 四卡 B64 慢步时：VAE 卷积内核回退

2026-09-19，本地 GPU 3、4、5、6，绝对 EEF，4×64/GAS1；生产配置和计算代码未修改。
正式训练尚未启动。用户要求定位耗时后，完成稳定短测与独立内核诊断；全部诊断进程已退出。

## 实际耗时

24 步真实训练更新正常结束，统计第 5–24 步的最慢 rank：平均 **19.5521 秒/步**，范围 19.5186–19.6893，CV 0.236%。

| 部分 | rank0 平均 CUDA event 时间 |
|---|---:|
| 冻结 T5 缓存 | 0.00152 秒 |
| 像素预处理/H2D | 0.31647 秒 |
| VAE 编码 | **11.29287 秒** |
| 前向与 loss | 1.85569 秒 |
| 反向 | 5.84385 秒 |
| DeepSpeed 更新 | 0.20648 秒 |

CUDA event 分段与整步墙钟定义不同，不能要求精确求和。VAE 约占整步 58%，是此次新增慢步时的主要来源。
四卡利用率均值 98.5%–98.9%；训练进程树 CPU 均值约 5.64 个逻辑核。当前主瓶颈表现为 GPU 卷积执行，不是长期等待 CPU 或数据。
T5 每个 rank 只编码一次，缓存三条任务文本。反向与之前目标机八卡约 6.03 秒接近，CPU 梯度节点重复处理优化没有表现出回退。

## 从真实训练 trace 得到的证据

另一次同配置 4 步诊断，仅在 rank0 第 1 和第 3 步的 VAE 调用内打开 profiler，不改变 tensor 或模型计算。

| 项目 | 第 1 步 | 第 3 步 |
|---|---:|---:|
| VAE 输入 | 64×3×9×384×320，bf16 | 完全相同形状和精度 |
| VAE 前 allocated | 29.8178 GiB | 41.0643 GiB |
| VAE CUDA kernel 时间合计 | 2.45196 秒 | 11.28035 秒 |
| indexed Tensor Core 卷积内核 | 18 次，共 0.73312 秒 | 10 次，共 0.11901 秒 |
| `implicit_convolveNd_sgemm` | 0 次 | **8 次，共 9.53229 秒** |

其余 63 次 `sm80_xmma_fprop_implicit_gemm...` 约 0.63 秒，基本不变；数据拷贝、布局转换和逐元素运算也未出现对应量级的增长。
也就是说，主要差异集中在 8 次 3D 卷积从原快速内核切换成 `implicit_convolveNd_sgemm`。
GPU 利用率接近 100% 是因为这些慢卷积仍然在执行，不能据此认为计算效率正常。

## 为什么首步快、后面一直慢

首步 Adam 状态尚未全部分配；首次更新后，状态常驻 GPU。训练步末 allocated 从八卡历史记录的约 31.6 GiB 提高到四卡约 40.0 GiB；四卡 ZeRO-2 的每卡优化器分片更大。
进入 VAE 前还包括当前批次像素等张量，实测第 3 步为 41.06 GiB。
现有监测入口同时沿用 95% 的 allocator 上限，约 75.46 GiB；没有为本次诊断调高限制。

24 步记录显示：显存分配重试从首步 0 增至第 2 步 24，此后始终为 24；VAE 从首步 2.58 秒转为第 3 步起约 11.29 秒。
因此早期“持续回收缓存拖慢每一步”的推测不准确：后续重试不再增长，但慢内核持续存在。

结合运行记录、内核 trace 与 PyTorch 2.7.1 实现，证据指向 **workspace 显存压力触发卷积 plan 回退，并缓存了能够运行但很慢的 plan**。
PyTorch 的 `run_conv_plan` 从 caching allocator 申请 workspace；缓存 plan 遇 OOM 后会重选，成功 plan 会写入缓存，后续直接使用。
来源：[PyTorch v2.7.1 Conv_v8.cpp](https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/cudnn/Conv_v8.cpp)。
本次没有记录每一个 cuDNN plan 的 workspace 大小，因此不声称已经测出精确的 workspace 临界值。

本机实际 PyTorch 2.7.1+cu126、cuDNN 9.5.1；trace 仍是 cuDNN 卷积内核。本次不是此前别人截图所述的 torch2.9/cuDNN9.8–9.14 组合。

## 对照实验与边界

仅加载冻结 VAE，在同 B64 输入和约 29 GiB 显存占位下，稳定编码为 2.44 秒，重复输出逐元素相同。
人为增加大块显存占位到约 40 GiB 时，该独立实验在 padding 分配中 OOM。大块占位不能等同真实模型的 allocator 布局，所以根因判断采用上述真实四卡 trace，而不是将人工占位实验视为成功复现。

## 后续处理候选，尚未实施

1. 若保持训练 B64/GAS1，可验证仅将冻结 VAE 的 64 个独立视频窗口按固定 32 个一组编码，再按原顺序拼回 latent。时序窗口和训练 batch 保持原样，不按卡数动态切换。需要先做真实 latent 数值对照和完整训练短测，不能提前承诺逐位一致或提速幅度。
2. 也可直接验证训练 B32；这会将全局 batch 从 256 改为 128，5 遍所需更新步数和学习率日程随之变化，必须单独审核。

当前未调整 batch、梯度累积、VAE 实现、cuDNN 设置或显存上限。按 19.55 秒估算，原 16045 步纯计算约 87.14 小时；在性能处理方案审核前未启动正式训练。

## 证据位置

- `outputs/nero_absolute_validation/20260919_gpu3456_b64/smoke_summary.json`：完整 24 步汇总。
- `outputs/nero_absolute_validation/20260919_gpu3456_b64/stages_rank*.jsonl`：每步耗时、T5 调用及 allocator 重试。
- `outputs/nero_absolute_validation/20260919_gpu3456_profile/kernel_comparison.json`：首步与第 3 步 CUDA 内核对比。
- 同目录 `vae_step{1,3}_trace.json`、`vae_step{1,3}_memory.json`、`vae_step{1,3}_operators.txt`：原始 trace、实际形状/显存与算子表。
- `outputs/nero_absolute_validation/20260919_gpu3456_b64/vae_memory_profile/`：独立显存对照中的低占用 trace；高占用 OOM 保存在外层日志中。
