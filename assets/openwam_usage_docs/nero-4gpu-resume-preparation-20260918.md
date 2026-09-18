# 四卡恢复：等待任务已暂停，改用0/1双卡验证

后续验证更新：用户指定GPU0、1、2、3四卡，已完成每卡batch32的24步短测。
排除前4步后平均6.91秒/步，训练进程峰值65.37 GiB/卡，无OOM，loss有限。
详见 `nero-gpu0123-validation-20260918.md`。这验证了四卡batch32计算与更新，
尚未验证完整状态恢复；正式训练和原自动等待继续暂停。

用户后续要求先换物理GPU0、1验证。四卡等待进程已收到SIGTERM并退出，
`outputs/nero_multitask_4gpu_launch/status` 为 `paused_by_user`；不会自动启动四卡训练。
四卡4000步转换已完成，全部FP32分片逐元素核验通过，文件保留供后续恢复。
以下自动等待说明记录的是暂停前已部署的流程。

用户指定物理GPU2、3、4、7，并要求保留原配方、增大batch。
目前按每卡32准备，梯度累积1，全局batch128；32/64按每卡口径解释，先验证32。
保留BF16、GPU AdamW、原模型/数据/冻结策略、1e-4峰值LR、5 epochs和每2000次更新保存。
启用已验证的梯度节点复用，不使用CPU优化器卸载或VAE缓存。

## 当前阻塞

初次检查四卡空闲，准备期间出现其他用户的训练进程。
再次检查GPU2/3约42.4 GiB空闲，GPU4/7约50.4 GiB空闲。
每卡batch32的独立四步测试在显存预检阶段被拦下，尚未创建训练rank或更新参数。
用户随后明确要求等待四张卡空闲后再用。已启动 tmux `openwam-nero-4gpu`，
运行 `scripts/wait_nero_4gpu.py`；不操作其他进程，等待不分配GPU显存。
原4000步完整保存点保持不变；CPU转换的新四卡完整状态写入独立目录。

## 自动等待流程

1. CPU转换六卡保存点至四卡，并逐元素核验FP32主权重及AdamW两个动量。
2. 每30秒检查物理GPU2、3、4、7：均无计算进程、空闲显存至少75 GiB、利用率不高于5%。连续3次通过才启动。
3. 独立运行四步batch32测试，四个rank均产生完整且有限的loss记录后才继续。
4. 再次等待连续三次空闲检查，通过后从完整4000步状态恢复；W&B online。
5. 记录四rank逐步耗时及显存，并采样首小时CPU/GPU资源；跳过最初10步，以最近20–50步耗时变异系数不高于20%作为稳定窗口。

启动前还会复查是否存在计算进程和显存余量。共享机器没有跨用户GPU预约锁，
预检后其他作业仍可能启动；启动/测试异常时等待程序停止并记录`needs_attention`，不会反复重试OOM。

状态及日志在 `outputs/nero_multitask_4gpu_launch/`：

- `status`、`queue_state.json`、`queue.log`：当前阶段与最近GPU检查。
- `conversion.log`、`batch32_validation.log`、`train.log`：转换、测试与正式训练日志。
- `resources.jsonl`、`stable_summary.json`：资源采样及稳定耗时摘要（训练启动后生成）。

查看：`tail -f outputs/nero_multitask_4gpu_launch/queue.log`。
取消等待或暂停该任务：`kill -TERM "$(cat outputs/nero_multitask_4gpu_launch/queue.pid)"`。
该信号只转发到本等待程序启动的子进程组。

## 已准备

- `configs/train_nero_4gpu.yaml`：每卡32的候选配置，未通过GPU显存实测。
- `scripts/train_nero_4gpu.sh`：指定四卡启动器，要求每卡至少75 GiB空闲。
- `scripts/convert_nero_zero2_6to4.py`：无padding、单参数组的精确FP32主权重/AdamW状态分片转换；写出后逐元素核验。
- `openwam/train/utils/zero2_reshard.py`：跨原分片边界的切片重组，以及按样本进度换算调度器。
- `scripts/monitor_nero_resources.py --gpus 2,3,4,7`：资源采样支持指定GPU。

## 更改batch时的恢复语义

原4000次更新已处理384000个窗口。全局batch由96变128后，这对应新数据流中的3000个batch，
而优化器计数仍为4000。使用`resume_data_step_offset=1000`保持两者关系，避免错误跳过128000个窗口。
训练循环和进度条支持该偏移，默认0时保留原有行为。

调度器保留恢复瞬间LR，并按已处理样本数继续原warmup/cosine曲线。
不是重置warmup，也不自动线性放大学习率。5 epochs计划对应约29085次剩余更新，
最终累计optimizer step约33085（包含原4000次）；与原42780次的差异来自batch增大。

43项定向测试通过，覆盖跨分片边界的逐元素等价、学习率按样本进度对齐、
数据位置偏移及原checkpoint/梯度统计回归。
这些CPU测试不能代替尚未进行的四卡加载、显存和稳态吞吐验证。

以下是等价手动命令，仅供记录；等待任务运行期间不要重复执行：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. .venv/bin/python scripts/convert_nero_zero2_6to4.py \
  --source-run outputs/nero_multitask_6gpu/2026-09-17_18-22-36 \
  --target-run outputs/nero_multitask_4gpu/resume_step4000 --micro-batch 32
CUDA_VISIBLE_DEVICES=2,3,4,7 bash scripts/train_nero_4gpu.sh
```

转换需要新的目标目录；不可将原保存点作为输出。实际正式启动应放在tmux并持续观察稳态更新耗时。
