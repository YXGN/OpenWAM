# 物理GPU0/1双卡验证（2026-09-18）

后续清理：用户已要求撤回双卡CPU卸载与额外梯度累积的兼容设计。
`validate_nero_gpu01.sh` 现只接受batch参数，采用GPU AdamW、累积1；
原 `8 cpu` 命令已失效且会在启动前报错。以下CPU卸载测量仅作为历史记录保留。

用户要求从四卡等待改为先用GPU0、1验证。已停止 `openwam-nero-4gpu`
等待任务，其状态为 `paused_by_user`。四卡4000步保存点转换文件仍保留。
两轮测试使用独立输出目录，加载4000步模型权重，不恢复/修改正式优化器状态，
不保存训练权重，W&B disabled。测试后全部退出，GPU0/1已释放。

## 测试时是否有其他进程

启动前GPU0仅15 MiB、GPU1仅5 MiB显存占用，均无NVML计算进程。
两轮采样均未发现这两张卡上其他计算进程。其他卡仍有作业。
低GPU利用率本身不能保证无影响：进程仍可能保留显存，稍后恢复计算，
以及共享CPU、内存带宽、存储和通信资源。两次不同时间的运行不是严格因果对照。

## 每卡32、GPU AdamW

命令：`bash scripts/validate_nero_gpu01.sh 32`。
BF16、ZeRO-2、梯度节点复用开启，GAS1，不使用VAE缓存。

首个优化器更新在AdamW初始化 `exp_avg_sq` 时OOM，未完成任何更新。
单个FP32分片需11.22 GiB；失败时PyTorch allocated约56.83 GiB，
reserved但未用约8.55 GiB，物理空闲约12.43 GiB。
现有监控入口保留显存余量，分配器上限约75.46 GiB（总可见79.44 GiB）。
因此这是当前带余量保护、当前分配模式下的失败，不能据此证明取消保护也一定失败。
未取消保护或更换优化器实现。

双卡的每卡FP32主权重及两个AdamW动量约33.65 GiB，四卡约16.82 GiB。
双卡失败不能直接推断四卡batch32失败；四卡仍须实际验证。
日志：`outputs/nero_gpu01_validation/batch32/train.log`。

## 原双卡CPU卸载配方对照

命令：`bash scripts/validate_nero_gpu01.sh 8 cpu`。
每卡8，累积6，全局batch96，原torch AdamW CPU卸载，OMP8，每rank4 workers。
完成24微步/4次更新，loss全部有限，退出码0。正式四卡配置未更改。

首个完整更新（按两rank逐微步最大值合计）65.28秒，作为预热排除。
后续三次分别37.38、36.15、36.39秒，平均36.64秒，变异系数1.45%。

| 指标 | 之前GPU6/7 | 本次GPU0/1 |
|---|---:|---:|
| 完整更新，96窗口 | 73.42秒 | 36.64秒 |
| 普通微步 | 8.69秒 | 3.97秒 |
| 包含优化器更新的微步 | 29.93秒 | 16.81秒 |
| 吞吐 | 1.31窗口/秒 | 2.62窗口/秒 |

本次稳态采样：GPU0/1平均利用率45.5%/65.4%，各进程显存峰值30.40 GiB，
PyTorch allocated峰值27.36/28.73 GiB，reserved峰值均28.78 GiB。
训练进程树平均占4.30个逻辑核（整机128个逻辑核）。
利用率包含计算、通信等GPU活动，不能等同于有效算力利用率；
CPU卸载和同步仍带来等待，没有消除所有CPU/GPU相互等待。

观察到约2倍吞吐改善，但新测试是从权重初始化AdamW，数据位置、物理GPU及整机负载不同，
样本量只有三个预热后更新。不能将改善全部归因于同卡竞争，也不能当作四卡GPU AdamW性能。
此次短测未启用/关闭梯度节点优化做A/B，因此不是该优化收益的独立测量。

原始统计：`outputs/nero_gpu01_validation/batch8_cpu/summary.json`。
逐rank记录：`outputs/nero_gpu01_validation/batch8_cpu/runs/*/telemetry_rank{0,1}.jsonl`。
资源采样：`outputs/nero_gpu01_validation/batch8_cpu/resources.jsonl`。
