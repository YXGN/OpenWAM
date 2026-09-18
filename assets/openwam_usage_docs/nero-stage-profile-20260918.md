# Nero 训练阶段耗时诊断：2026-09-18

最新进展：后续GPU0对照已复现主要CPU开销：DeepSpeed参数hook反复遍历完整参数列表，
统计函数约11.58秒/步。见 `nero-step-time-root-cause-20260918.md`。
下文保留第一阶段的测量范围与当时结论，不能把此前未归属差额再直接当作通信时间。

第一阶段测量时保持正式训练 `openwam-nero-6gpu` 继续运行。正式普通步时约14.93–15秒；
GPU0独立对照的本地模型流水线约3.317秒。两者范围不同，不能把差额全部归因于NCCL。

后续状态：用户要求暂停后，已向原训练 launcher 发送 SIGINT，确认六个 rank 退出。
最后观察到约4637步，完整可恢复保存点为4000步，未保存约637步需重跑。
原 tmux pane 保留，launch/status 标记为 paused_by_user。未自动恢复正式训练。
已新增 `scripts/profile_nero_distributed.py`，用隔离输出目录执行12步六卡原训练链路计时，
配置解析和静态检查通过；GPU2–7随即被其他 Isaac-GR00T 作业占用，尚未实际运行六卡诊断。
VAE 缓存样本验证已完成，见 `nero-vae-cache-audit-20260918.md`。

## 测量条件

- 独立GPU0，A100 80GB；已有其他进程占用约25GiB，采样时其GPU利用率为0。
- 读取正式第4000步checkpoint，未修改或保存权重。
- 与正式每rank相同的batch=16、bf16、384×320视频、9帧、32动作步、4 loader workers、梯度检查点。
- 官方 reader、架构和compute_loss代码直接调用；6个真实batch计时，之前预热2步。
- 保留当前随机种子/确定性设置。独立模型无DeepSpeed包裹；反向无梯度跨卡通信。
- CUDA同步边界测wall time。阶段包含本地CPU提交与设备执行；嵌套阶段不能重复相加。
- 另测2个减少细粒度同步的batch，总计约3.32–3.43秒，未显示数量级的测量开销。
- 另录1步PyTorch CPU/CUDA trace，trace步不计入上面的均值；其采样本身会增加开销。

## 本地流水线平均耗时

| 阶段 | 秒/step | 占独立流水线比例 |
|---|---:|---:|
| DataLoader 预取后的取数等待 | 0.000243 | 0.0% |
| 文本 tokenizer + T5 编码 | 0.615460 | 18.6% |
| 逐帧像素转换和传入 GPU | 0.079437 | 2.4% |
| 视频 VAE 编码 | 0.618141 | 18.6% |
| 视频/动作联合前向 | 0.494331 | 14.9% |
| 视频与动作 loss | 0.003775 | 0.1% |
| 本地反向（含梯度检查点重算） | 1.492655 | 45.0% |
| 梯度清理 | 0.009015 | 0.3% |
| 其余准备/加噪/计时边界 | 0.004031 | 0.1% |
| 合计 | 3.317087 | 100% |

DataLoader的数字是预取后的等待，不是全部视频解码CPU工作量。
它也不能证明正式6rank、24workers的每一个batch都没有数据等待。
2个预热batch中的首次取数约1.97秒，不计入稳态均值。
本地张量峰值约37.22GiB，缓存保留峰值约38.36GiB；不代表正式ZeRO-2的显存峰值。

## 优化器和通信的独立检查

按可训练参数6021155088 / 6构造1003525848个FP32元素的AdamW shard，
只测本地更新算子，预热后7次平均0.052750秒。
这不含DeepSpeed梯度归约、范数/裁剪、类型转换和参数all-gather，不能当作正式optimizer整段耗时。

另外使用同一组物理GPU2–7创建独立NCCL通信组。每种操作预热2次、测5次；
最大buffer为256MiB，测试有90秒超时，已正常退出。
- 16MiB all-reduce：最慢rank平均约2.71ms。
- 256MiB all-reduce：约21.26ms（单次约7–26ms）。
- 256MiB all-gather：约13.24ms（该buffer为聚合后的总大小）。
- 日志显示 `P2P/CUMEM/read`，主机拓扑为NV12。

通信测试与正式训练短暂并行，其数字包含并发影响；它确认小规模通信可用，
不能按字节数直接外推正式所有梯度bucket、rank到达时间或重叠后的耗时，
也不能替代读取正式训练内部的通信路径和计时。

## 结论与边界

1. 独立模型计算本身约3.3秒，当前正式约15秒值得进一步检查，不能笼统归因于模型大。
2. 本次逐帧搬运约0.08秒，文本编码约0.62秒。缓存文本可能有收益，但单独不能解释约11.6秒差额。
3. 优先定位正式DeepSpeed反向/梯度归约、step内梯度处理和all-gather，以及rank等待和主机提交开销。
4. 尚未精确归属的差额不是测得的“通信耗时”。独立测量未包含真实优化器和通信的相互作用。
5. 正式训练与诊断进程均可用128个CPU；已检查cgroup无CPU配额限速，未发现额外CUDA阻塞环境变量。
6. 运行中进程的py-spy附加被OS拒绝，ptrace_scope=1；sudo非交互权限不可用。
   第一阶段没有更改系统安全设置或中断正式训练；后续按用户要求暂停，见本文开头状态。

## 复现和产物

```bash
source scripts/activate_openwam.sh
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  python scripts/profile_nero_stages.py \
  --checkpoint outputs/nero_multitask_6gpu/2026-09-17_18-22-36/checkpoint_step_4000.safetensors \
  --output outputs/nero_stage_profile_20260918 --trace
```

脚本要求单张可见GPU且至少50GiB空闲，并限制自身分配器最多使用该卡60%显存。
重新运行时请用新的output目录以保留已有测量。

- `scripts/profile_nero_stages.py`：独立阶段计时与synthetic AdamW shard测试。
- `outputs/nero_stage_profile_20260918/steps.json`：所有预热/实测batch。
- `summary.json`：均值、范围、对照及optimizer微基准。
- `trace.json`：Chrome/Perfetto格式CPU/CUDA时间线。
- `operators.txt` / `operators_cpu.txt`：算子汇总，嵌套的CUDA annotation存在重叠，不应求和。
- `collectives.py` / `collectives.json` / `collectives.log`：独立六卡通信检查。

若要给出当前六卡全部阶段的准确占比，需要在下一次启动/恢复时加入训练内计时，
覆盖数据等待、prepare_inputs、forward、DeepSpeedEngine.backward、DeepSpeedEngine.step、
梯度范数、底层AdamW、all-gather、指标归约和日志；各rank分别记录，再按step对齐。
在稳定checkpoint之后切换可避免回退未保存进度。后续用户要求立即暂停，已停止并保留4000步保存点。
