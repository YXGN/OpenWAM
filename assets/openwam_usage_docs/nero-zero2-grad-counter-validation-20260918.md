# Nero ZeRO-2 梯度统计优化与验证

2026-09-18。正式训练保持暂停；没有实现或启用 VAE 数据集缓存，没有修改已安装的依赖。

## 优化内容

DeepSpeed 0.18.5 原实现每次参数梯度 hook 都重新寻找全部参数的梯度累积节点。
完整模型有 1651 个可训练参数张量，每次反向约触发 2725801 次参数访问。
`_get_grad_fn_or_grad_acc` 的叶子参数路径包含 `view_as`，大量 CPU/autograd 操作延迟 GPU 工作提交。

新增 `openwam/train/utils/deepspeed_grad_counter.py`，只在当前 backward graph task 中复用叶子参数的梯度累积节点。
每次 hook 仍调用 `_will_engine_execute_node` 检查实际参与状态；没有把参与参数数量固定为 1651。
graph task 或 requires_grad 状态改变会重建节点集合。原来的 process_gradients、归约和 epilogue 顺序保留。

安装入口校验 DeepSpeed 版本及原函数源码 SHA256，只支持已验证的 0.18.5 ZeRO-2。
修改仅存在于启用开关的 Python 进程，不编辑 site-packages，不改变优化器/保存点格式。

| 文件 | 用途 |
|---|---|
| `openwam/train/utils/deepspeed_grad_counter.py` | 节点复用、逐 hook 审计及版本保护 |
| `scripts/train.py` | 在 Accelerator/DeepSpeed 初始化之前按开关安装 |
| `configs/train.yaml` | 两个开关，均默认关闭 |
| `tests/test_deepspeed_grad_counter.py` | 动态参与、重复参数、重入/非重入检查点、累积和保护测试 |
| `scripts/validate_zero2_grad_counter.py` | 真实 ZeRO-2 的梯度、AdamW 状态、更新和恢复对照 |
| `scripts/validate_nero_grad_counter.py` | 完整模型的统计耗时及全部梯度对照 |

## 原计算内核下的全模型结果

GPU0 / A100，BF16，batch 16，4000 步权重；固定 batch 覆盖三个任务。
原版与优化版各测两次，单独预热。不做参数更新。
输入仅在此隔离对照中复用，以保证 A/B 完全一致；没有写入数据集缓存。

| 阶段 | 原统计 | 优化统计 |
|---|---:|---:|
| 前向与 loss | 0.675 s | 0.857 s |
| 反向 | 17.275 s | 2.479 s |
| 其中统计函数 CPU 累计 | 16.388 s | 1.126 s |
| 前向与反向合计 | 17.950 s | 3.336 s |

反向耗时下降约 85.6%。CPU 累计包含在反向时间内，不能重复相加。
主机及 GPU 有其他作业，两次样本不足以代表稳定吞吐；当前原版比此前的 12.203 s 更慢。
此前原版反向 12.203 s、统计 11.581 s 的独立实验也支持同一瓶颈，详见
`nero-step-time-root-cause-20260918.md`。这些是完整模型的单卡梯度计算对照，
不是六卡完整训练整步时间，不能把 2.479 s 称为修复后的正式 step time。

所有模式 loss 精确一致：0.0993628054857254。
审计模式逐次比较原计数与新计数，1651 次全部一致。

原计算内核下梯度不保证位级确定性：原实现重复运行的全梯度相对 L2 差异为 0.255%；
优化版相对同一参考分别为 0.285%、0.464%。最大绝对差分别为 1.526e-5、3.052e-5。
按测试前设定的“原版单次重复差异 × 1.25”范围，第二次优化结果未通过。
不能只凭 loss 一致就忽略这个结果，也不能用这一次原版重复估计完整波动分布。

原始产物：`outputs/nero_grad_counter_validation_20260918/{steps.json,summary.json,run.log}`。

## 确定性全模型对照

为进一步定位上述差异，保持同一权重、输入、batch 和 BF16，另开独立进程启用
`torch.use_deterministic_algorithms(True)`。当前环境的 attention 后端为 PyTorch SDPA；
该设置仅用于诊断，没有修改正式训练的 attention 配置。

原版重复、优化版两次、逐 hook 审计版相对原版参考，全部 1651 个参数张量的梯度均逐元素一致：

| 比较 | loss | 梯度最大绝对差 | 梯度相对 L2 差 | 不同的参数张量数 |
|---|---:|---:|---:|---:|
| 原版第二次 vs 第一次 | 完全一致 | 0 | 0 | 0 |
| 优化第一次 vs 原版 | 完全一致 | 0 | 0 | 0 |
| 优化第二次 vs 原版 | 完全一致 | 0 | 0 | 0 |
| 审计版 vs 原版 | 完全一致 | 0 | 0 | 0 |

审计版的 1651 次动态参与计数也全部一致。这支持之前差异来自非确定性计算，
且节点复用没有改变该全模型 batch 的梯度数学语义。
确定性对照中反向均值原版 21.096 s、优化版 2.600 s；主机竞争使绝对时间有波动，
不能把确定性与非确定性两组的绝对耗时之差解释为确定性开关的代价。

产物：`outputs/nero_grad_counter_deterministic_20260918/{steps.json,summary.json,run.log}`。
本实验包含完整梯度，但没有完整模型 AdamW 更新；优化器与恢复验证见下一节。

## ZeRO-2 状态与恢复验证

真实 DeepSpeed 小模型，5 次 AdamW 更新，包含未参与参数及非重入梯度检查点。
覆盖 FP32 / 累积 1、BF16 / 累积 1、BF16 / 累积 2；分别跑原版、原版重复、优化版和逐 hook 审计版。
比较 loss、归约后梯度、BF16/FP32 权重、FP32 主权重、AdamW 一阶/二阶动量。
BF16 累积 1 另做原版第 4 步保存、优化版恢复后第 5 步与连续运行对照。

- GPU0 单进程：7 个结果组、458 项 equal 检查均为 true。
- CPU/Gloo 两进程：各 rank 使用不同输入；每个 rank 的 7 个结果组、458 项 equal 检查均为 true。
- 定向回归：49 项测试通过。

产物：`outputs/zero2_grad_counter_validation_20260918/rank0/results.json`；
`outputs/zero2_grad_counter_validation_cpu_distinct_20260918/rank{0,1}/results.json`。
CPU 分布式验证覆盖分片/归约/恢复流程，但不能替代六卡 NCCL 验收。

## 使用与边界

`training.zero2_cached_grad_nodes=false` 为默认配置。显式设为 true 才启用优化。
`training.zero2_grad_counter_audit=true` 会在每个 hook 同时运行原统计作比较，适合诊断，不能用于加速测量。
回退只需关闭节点复用开关，无需转换保存点。

未来恢复/验证训练时，在原命令的所有恢复参数之外加
`training.zero2_cached_grad_nodes=true`；无需修改 batch、学习率、累积步数、精度或冻结策略。
不要为了使用该优化把正式 attention 强制切换到确定性模式。

复现确定性全模型对照（使用新的输出目录）：

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled OMP_NUM_THREADS=4 PYTHONPATH=. \
  .venv/bin/python scripts/validate_nero_grad_counter.py --deterministic \
  --checkpoint outputs/nero_multitask_6gpu/2026-09-17_18-22-36/checkpoint_step_4000.safetensors \
  --output outputs/nero_grad_counter_deterministic_repeat
```

当前没有自动恢复正式训练。六卡完整模型的端到端步时、NCCL 通信和保存恢复仍需在相应资源可用时验收。
有限数值对照不能证明长期策略成功率必然完全相同；此优化的目标是保留训练数学定义和更新语义。
