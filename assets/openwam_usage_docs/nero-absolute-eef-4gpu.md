# 本地 Nero 绝对 EEF：隔离的最小实现

本次增加独立绝对 EEF 实验，原相对 EEF reader、统计、配方以及目标机运行不变。
复用已有窗口索引、三路相机、FK 拼装、proprio 和通用训练器，不复制数据集、不新增按卡数切换策略。

## 数据定义

- 每臂 `xyz + rot6D + gripper`，双臂 20 维。XYZ 是 **body 坐标系下 gripper_flange 的绝对位置**，不是世界坐标；旋转取绝对旋转矩阵的前两列，按列展开。
- 一个窗口读 `t..t+32`；proprio 是 `t` 的绝对 EEF，动作是 `t+1..t+32` 的绝对 EEF，不减当前位置、不做相对旋转。
- 夹爪始终是绝对开口宽度，单位米，范围 `0..0.1`。源 `action` 等于当前关节状态，不能直接用作未来 EEF 标签。
- 左臂映射到 80 维槽位 `0..9`，右臂 `34..43`；其余维度由 mask 排除。
- 相机仍为 `[head_fpv, left_hand, right_hand]`，上方头部、左下左腕、右下右腕；384×320，33 原始帧以步长 4 取 9 帧。
- 独立 `eef` 动作统计覆盖所有完整窗口的全部未来标签，`eef_state` 覆盖当前帧状态。XYZ 分别使用 min-max；rot6D 保持恒等归一化；夹爪固定 `0..0.1 → -1..1`。
- 独立统计带绝对语义 contract 和数据指纹。相对/绝对 reader 互换统计会报错；准备脚本拒绝覆盖相对统计。
- 部署复用通用 `eef` 路径：反归一化后已经是绝对值，不再加回 anchor。新增仅针对本 reader 的统计检查和输出语义声明。机器人执行端仍需正确处理 rot6D 与执行约束，本次不增加执行器逻辑。

## 文件

```text
configs/
├── dataloader/nero_multitask_absolute.yaml       # 继承相机/窗口配置，仅覆盖类型、动作模式、统计路径
└── train_nero_absolute_4gpu.yaml                # fresh base、四卡 B64/GAS1、独立保存路径
openwam/
├── dataloader/
│   ├── nero_multitask_absolute.py              # 复用原 reader 的读取方法，替换动作生成与统计加载
│   ├── registry.py                            # 注册新 reader
│   └── utils/
│       ├── nero_absolute.py                   # 绝对语义及统计校验
│       └── stats_computation/nero_absolute_stats.py
│                                              # 复用全量源数据审计，再计算绝对动作统计
└── deploy/model_loader.py                     # 仅针对新 reader 的部署校验/语义
scripts/
├── data/prepare_nero_absolute.py               # 统计准备入口
└── train_nero_absolute_4gpu.sh                  # 显式指定四卡，启动前检查空闲显存
tests/dataloader/test_nero_multitask_absolute.py # 标签、统计、相机、部署及隔离回归
outputs/                                       # 以下为本地生成产物，不入 Git
├── nero_absolute_data/                        # normalization_stats.npy、audit.json
├── nero_absolute_preparation/                 # 真实 reader 抽查、旧权重清理记录
├── nero_absolute_4gpu/<时间>/                 # 正式训练时生成
└── nero_absolute_weights/<时间>/step_4000/    # 正式训练时生成
```

## 配方

| 项目 | 设置 |
|---|---|
| 初始化 | 本地官方 foundation `checkpoint_step_154000.safetensors`，不恢复旧相对权重 |
| GPU / 每卡 batch / 累积 | 4 / 64 / 1；全局 batch 256 |
| 数据 | 408 episodes，834400 帧，821344 完整窗口；全部训练，不做任务均衡 |
| 训练长度 | 5 epochs，3209 optimizer steps/epoch，共 16045 steps |
| 尾部补齐 | Accelerate 每遍补齐 160 个窗口，所以覆盖量略高于 5 遍 |
| 优化 | LR 1e-4、cosine、warmup 5%、bf16、ZeRO-2、gradient checkpointing |
| 性能设置 | 沿用已验证的 ZeRO-2 优化和冻结 T5 文本缓存，VAE 在线计算 |
| 保存 | 每 2000 步保存完整状态，先写新再清旧，训练中保留最近 1 套 |
| 独立权重 | 第 4000 步另存权重和加载资产，不带优化器，不随滚动清理删除 |

保存默认位于本项目 `/data2/xieweize/project/OpenWAM/outputs/`。沿用训练器现有结束行为：正常结束时清理恢复状态，保留最终权重。

目标机八卡 B64/GAS1 的全局 batch 是 512，这里为 256；虽然每卡设置相同，优化轨迹不保证等价。
采用相同 5 遍覆盖时本地更新次数约为两倍；本地 4000 步的样本数约等于目标机 2000 步。
绝对/相对标签和归一化范围本来就不同，因此不承诺训练效果相同，应作为独立实验比较。

## 使用

本次已生成全量绝对统计。重建时在项目根目录执行：

```bash
source scripts/activate_openwam.sh
python scripts/data/prepare_nero_absolute.py \
  --dataset-dir /data2/xieweize/project/dataset/lerobot_v2/nero_multitask_mooncake_hammer_charger_408_fk_lerobot_v2
bash scripts/train_nero_absolute_4gpu.sh --print-config
```

完成四卡显存/耗时短测并确定 GPU 后，启动命令为下述形式；GPU 编号只是示例，并非现在已空闲：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_MODE=online \
  bash scripts/train_nero_absolute_4gpu.sh
```

本次未启动正式训练或四卡 GPU 短测。启动脚本要求每卡至少 77 GiB 空闲，但这只是前置检查，不是不会 OOM 的保证。
现有目标机八卡 B64 数据为 allocated 峰值约 64.12 GiB、进程峰值 74.49 GiB；四卡 ZeRO-2 每卡优化器分片更大，不能直接沿用该显存结论。
本地既往四卡验证为 B32，不能作为 B64 实测依据；如需调整 batch/累积或训练策略，另行审核。
启动时使用本地当前 W&B 登录，独立 run name；脚本清除旧的 WANDB_RUN_ID/WANDB_RESUME，避免续写相对实验。

## 本次验证与清理（2026-09-19）

- 全量源数据审计通过，指纹 `47040ee3b169c6b245575d4db3a0adabc78d637e1a5e2f3932fd4c233762e386`。
- 三任务各抽 4 条 episode，每条取首尾窗口，共 24 个窗口。动作反归一化最大绝对误差 `5.960464477539063e-08`；抽查归一化范围 `[-0.9998783, 1.0]`，均为有限值。
- 抽查窗口的首尾视频帧分别独立解码三路相机，与部署拼图逐像素一致；状态、标签偏移、槽位和 mask 均通过检查。
- 相对统计 SHA256 仍为 `65d5877e34384817e18f77faf0f90feefe8439ff7c74de58893be0a194d93afe`。
- CPU 回归 **62 passed、2 skipped**，跳过的是需 GPU 的训练/部署集成测试；配置解析、shell 语法和新增 Python 文件 lint 通过。
- 已删除本地旧 optimizer step 4000 权重及六卡原状态、四卡恢复副本、双卡重分片副本、验证副本和相应软链接。双卡文件名是 step 24000，元数据确认仍为 optimizer step 4000。
- 实际释放约 **330.0 GiB**，删除后 `/data2` 可用约 **584.6 GiB**。保留 base、step 2000、日志、配置和统计；远端文件未修改。
- 详细记录：`outputs/nero_absolute_preparation/old_step4000_cleanup.json`、`reader_audit.json`；历史 resume 配方保留作记录，其中指向已删除 step 4000 的路径不可再使用。

验证命令：

```bash
.venv/bin/python -m pytest -q \
  tests/dataloader/test_nero_multitask_absolute.py \
  tests/dataloader/test_nero_multitask.py \
  tests/test_checkpointing.py tests/test_frozen_text_cache.py \
  tests/test_train_deploy_consistency.py tests/test_save_deploy_assets_hook.py
```
