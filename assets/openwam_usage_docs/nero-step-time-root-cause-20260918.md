# Nero 每步约15秒：GPU0定位结果

2026-09-18。正式训练保持暂停，本次只运行隔离诊断，没有修改安装的DeepSpeed、原训练代码或保存点。

后续已增加默认关闭的节点复用实现，及独立数值/优化器恢复验证；见
[ZeRO-2 梯度统计优化与验证](nero-zero2-grad-counter-validation-20260918.md)。以下保留首次根因定位的数据与范围。

## 主要发现

主要慢点已在单卡完整模型上复现：DeepSpeed 0.18.5 的参数梯度hook每次调用
`count_used_parameters_in_backward(all_params_requiring_grad)`，重复扫描全部可训练参数张量。

当前模型有1651个可训练参数张量、6021155088个参数元素。完整反向触发1651次hook，
每次扫描1651个张量，共2725801次参数访问。每个正式rank都有完整模型，
ZeRO-2切分优化器状态不会把该遍历数量除以6。

安装包代码位置：

- `.venv/lib/python3.12/site-packages/deepspeed/runtime/zero/stage_1_and_2.py:1028`
- `.venv/lib/python3.12/site-packages/deepspeed/runtime/utils.py:1426`
- `.venv/lib/python3.12/site-packages/torch/autograd/graph.py:176`

PyTorch的 `_get_grad_fn_or_grad_acc` 对叶子参数调用 `view_as` 来取得梯度累积节点。
在每个hook内重新构造/查找所有节点，导致大量Python和CPU autograd操作；GPU等待后续计算提交。
这与此前排除的cuDNN Conv3D回退是不同问题。

## 完整模型因果对照

脚本 `scripts/profile_nero_hook_overhead.py`；GPU0，A100，batch16，BF16，
4000步权重，原视频/动作联合模型及梯度检查点，完整参数梯度。
先算好冻结输入，再将冻结模块移至CPU节省显存；三组使用同一batch、权重与随机种子。
无优化器更新，无跨卡通信，无高频CUDA同步；只在前向/反向边界同步。
每组预热1步、测2步。数字是隔离实验，不是原六卡整步trace。

| 模式 | 前向+loss | 反向 | 两者合计 | 统计函数CPU累计时间 |
|---|---:|---:|---:|---:|
| 原生反向，无统计hook | 0.504s | 1.465s | 1.969s | 0 |
| 加入DeepSpeed原统计hook | 0.523s | 12.203s | 12.726s | 11.581s |
| 诊断变体：复用梯度节点，仍逐hook检查参与状态 | 0.586s | 1.463s | 2.050s | 0.367s |

两种hook模式每步都调用1651次，参与参数计数均为1651，每步只有一个graph task。
统计函数CPU累计时间包含在反向wall time内，且CPU提交和GPU计算可能重叠，不能再相加。
单独添加这段代码使反向增加约10.74秒，足以解释正式约15秒步时的大部分异常。
复用节点变体仅用于定位，没有写入DeepSpeed或训练入口，不是已验收的正式修复。

数值检查：三组loss均精确等于0.0993628054857254。两种hook模式与无hook参考梯度并非位级一致：
原统计最大绝对差3.0517578125e-05，复用节点最大绝对差1.52587890625e-05。
本实验没有建立重复原生反向的误差分布或全梯度相对误差界限，因此不能据此宣布生产修复数值验收通过。
正式修复仍需独立loss/梯度/优化器更新与六卡保存恢复检查。

产物：`outputs/nero_hook_profile_20260918/{summary.json,steps.json,run.log,gpu0_samples.csv}`。
GPU利用率采样有固定55秒边界，不等于覆盖所有实验阶段的均值；采样命令正常由timeout结束（124）。

## 编码与本地DeepSpeed补充测量

`scripts/profile_nero_single_rank.py` 先读取8个真实batch并准备输入，排除前2个warmup：

| 阶段 | 平均时间 |
|---|---:|
| T5文本编码 | 0.620s |
| 像素预处理/传GPU | 0.124s |
| VAE编码 | 0.617s |
| 全部prepare_inputs（含以上项） | 1.364s |
| 预取后的单rank取数等待 | 0.000251s |

完整单卡AdamW状态超出可用显存，因此实际DeepSpeed对照仅把1006253056个参数元素放入优化器，
接近正式每卡1003525848元素的状态分片（大约0.27%差异）；模型仍完整计算全部梯度。
该测试world_size=1，冻结输入预先编码，不能用作正式六卡训练吞吐或完整通信测量。

粗粒度计时：前向0.494s、反向3.686s、engine.step 0.094s、总计4.289s。
优化器阶段细分：FP32 AdamW约0.052s，梯度范数约0.026s，解缩放/裁剪约0.0055s。
world_size=1的all-gather不代表正式跨卡耗时。

细粒度诊断对每个梯度处理hook插入CUDA同步，反向增至4.875s；
这是测量扰动，比较端到端时使用上面的粗粒度结果，不把多出来的1.19秒当正式开销。
另一个带profiler的trace步也不计入均值。该trace发现约65.6万次aten::view_as，
与约809个优化器参数张量的重复遍历相符，为后续完整模型对照提供了线索。

产物：`outputs/nero_single_rank_profile_20260918/`，包含config、scope、steps、summary与436MiB trace。

## 六路CPU数据读取对照

`scripts/profile_nero_loader_ranks.py` 不使用GPU：6进程×4workers，原reader，batch16，
Accelerate分片采样；每次取数后模拟3.3秒计算时间，按rank最大等待统计。
16步排除前2步：稳态最慢rank取数等待平均0.506s，中位数0.462s，最大0.964s。
首次预取约3.22s，排除于稳态统计。当前主机存在其他作业；此实验不能还原历史I/O竞争。
CPU对照不含GPU pin-memory/H2D，且rank启动时间通过barrier对齐，不能直接加到原15秒中。
没有在该场景复现十秒级的稳定数据等待。

产物：`outputs/nero_loader_profile_20260918/{rank0..5.jsonl,run.log}`。

## 与正式15秒的对应关系

编码准备约1.36s，加完整模型+原统计hook约12.73s，仅这两项已约14.09s，
无需跨卡通信就能复现接近正式步时的数量级。另有真正ZeRO梯度处理、参数同步、日志等。
这是跨实验的量级核对，不能把15减去14.09的差额命名为“精确通信时间”。
六卡内部按rank计时仍是确定完整占比和修复后吞吐的最终依据。

优先解决重复统计，再评估缓存。单独预编码VAE只移除约0.62秒VAE和像素处理，
不会消除当前约11.6秒CPU统计。文本缓存也有收益，但应放在主要框架开销修复之后评估。

上游源码已存在按阶段决定是否刷新计数的实现，可作为正式修复评估的参考，
不应未经项目数值和多卡恢复验证直接替换依赖：
https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py
（2026-09-18查看的create_gradient_handling_hooks中包含should_refresh_expected_hook_count。）

## 复现完整统计开销

```bash
source scripts/activate_openwam.sh
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python scripts/profile_nero_hook_overhead.py \
  --checkpoint outputs/nero_multitask_6gpu/2026-09-17_18-22-36/checkpoint_step_4000.safetensors \
  --output outputs/nero_hook_profile_20260918
```

重新运行请换输出目录；要求单张可见GPU至少50GiB空闲。本次全部诊断使用隔离输出，无正式保存点写入。
