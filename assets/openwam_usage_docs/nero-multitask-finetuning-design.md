# Nero 多任务：相对 EEF 微调

更新：2026-09-17。已实现数据准备、reader、Hydra 配置、部署还原及 CPU 检查工具，并完成官方 Foundation 的单卡 3-step 训练、更新和保存检查。详细证据与限制见 [训练链路审计](nero-multitask-audit-20260917.md)。按用户确认，全部 episode 用于训练，不划分验证集，不做任务均衡。

## 动作与观测契约

采用独立 `dataloader=nero_multitask`、`action_mode=eef_relative`，复用官方训练入口与默认模型。该模式与官方 Foundation 的张量形状和权重加载方式兼容，但把官方绝对 EEF 动作改为相对动作，因此不是官方原样动作语义，也不代表已经验证微调效果。

每个窗口的起点 t 是整个动作块唯一的参考帧，左右臂分别计算：

```text
position[k] = p[t+k] - p[t]                  # body 坐标系，米
rotation[k] = rot6d(R[t].T @ R[t+k])         # 矩阵前两列依次拼接
 gripper[k] = opening_width[t+k]            # 绝对宽度，米
k = 1 .. 32
```

旋转采用群运算，不是六个 rot6D 分量相减；零旋转为 `[1,0,0,0,1,0]`。各步始终相对 t，不逐步积分。状态输入 proprio 保持当前帧的绝对 body-frame EEF20。

Native 20D 顺序：`[L_xyz3, L_rot6d6, L_width, R_xyz3, R_rot6d6, R_width]`。归一化后映射至官方 80D 的 `0–9` 和 `34–43`，其余 60 维 mask=false。XYZ 动作统计按完整窗口里的所有 32 个未来差值计算；绝对 proprio 使用单独统计。rot6D 固定 min=-1/max=1，不拉伸。夹爪统一使用物理端点 `0–0.10 m → -1–+1`，连续值保留，不自动二值化，仅裁剪浮点端点误差；越界源数据直接报错。

数据中的原始 action 与当前 observation.state 相同，是 16D 关节状态。reader 使用已有 `observation.fk.fb.{left,right}.gripper_flange` 加夹爪宽度构造 EEF，显式取未来一帧开始的监督；没有把关节角当 EEF 或把 `fk.cmd` 当作独立控制命令。学习目标是未来已实现的轨迹，无法还原接触力或原始控制指令。

## 时间和三路相机

一个样本读取原始帧 `t..t+32`；proprio 取 t，动作取 `t+1..t+32`，视频取 `t,t+4,...,t+32`。只有完整 33 帧窗口，长度 L 的 episode 产生 L-32 个样本，禁止跨 episode。动作约 60 Hz、32 步覆盖约 0.533 秒；视频约 15 Hz。

| 源字段 | 训练画布 | 官方服务端请求字段 |
|---|---|---|
| observation.images.head_fpv | 上方 320×256 | images.head_camera |
| observation.images.left_hand | 左下 160×128 | images.left_wrist_camera |
| observation.images.right_hand | 右下 160×128 | images.right_wrist_camera |

输出画布为宽 320、高 384。reader 按视频实际帧索引解码，在原始分辨率取得图像，再使用与部署相同的 BILINEAR 拼图；不把左右路按字典顺序重排，不提前裁切或镜像。三路视频都必须存在，读取失败直接报错，不随机换样本或静默补黑图。服务端也要求三路相机。60.002 fps 的原视频保持原样，不重编码。

prompt 根据每条 episode 的 task_index 从 tasks.jsonl 读取，多行文本完整保留。月饼文本中的 `[box destination]` 仍保留；数据没有逐阶段语言标签。

## 数据评估与实际规模

数据目录：`/data2/xieweize/project/dataset/lerobot_v2/nero_multitask_mooncake_hammer_charger_408_fk_lerobot_v2`。

| 任务 | Episodes | 帧数 | 完整训练窗口 | 自然窗口占比 |
|---|---:|---:|---:|---:|
| 月饼 | 204 | 496,250 | 489,722 | 59.62% |
| 锤子 | 102 | 204,410 | 201,146 | 24.49% |
| 插座 | 102 | 133,740 | 130,476 | 15.89% |
| 合计 | 408 | 834,400 | 821,344 | 100% |

约 3.86 小时。第一版使用官方普通 DataLoader 的 shuffle，自然窗口采样，不改变任务比例。

此前独立审计扫描全部 Parquet：索引、形状、有限值、任务和 current_state 语义通过；全量 FK rot6D 正交误差不超过 1.20e-7。使用附带 URDF 独立重算 2448 个抽样位姿，最大分量误差约 2.98e-8。全部 1224 个视频容器帧数和分辨率通过；72 个抽样解码帧最近时间戳误差约 1.02e-6 秒。未逐条确认任务成功率，未重新完整解码所有视频，FK 数学一致性也不等于已验证实机标定。

月饼夹爪以连续宽度为主，锤子和插座约 99% 的宽度接近 0 或 0.10 m；保留该差异，不按任务重新拉伸宽度。物理解释依数据 opening_width 字段与当前 URDF 端点；本次没有恢复或重新采集夹爪命令。

## 准备与 CPU 检查

在项目根目录、安装了项目依赖的环境中执行：

```bash
python scripts/data/prepare_nero_multitask.py \
  --dataset-dir /data2/xieweize/project/dataset/lerobot_v2/nero_multitask_mooncake_hammer_charger_408_fk_lerobot_v2 \
  --output-dir outputs/nero_multitask_data

python scripts/data/check_nero_multitask.py
```

准备工具只读源数据，输出 normalization_stats.npy 和 audit.json。统计中包含 `eef_relative`（动作）与 `eef_relative_state`（绝对 proprio），以及动作定义、URDF/metadata/Parquet 内容指纹和视频 size/mtime 指纹；视频未做完整内容哈希。reader 启动时核对指纹，数据修订后必须重新准备。所有任务共享一套统计。统计只用于 min-max；位置 mean/std 来自样本，旋转和夹爪的变换参数固定，不提供经验分位数。

CPU 检查从 8 条 episode 的首、中、末选取 24 个窗口，覆盖各来源，检查实际视频解码、mask、未来 +1 对齐及物理动作还原。输出：

- `outputs/nero_multitask_data/reader_smoke.json`
- `outputs/nero_multitask_data/camera_preview.jpg`：每行同一 episode 的首、中、末窗口起始观测，图上标有 episode 与帧号。

当前真实数据检查通过，最大还原分量误差 `8.94e-8`。这是数据链路检查，不是模型训练或闭环效果评估。

若使用本机 pi5 环境，PyAV 需要其 libstdc++，可设置 `LD_LIBRARY_PATH=/data2/xieweize/.conda/envs/pi5/lib`。本次测试补充依赖放在 /tmp 隔离目录，未修改项目依赖清单；正式训练应使用安装完整官方依赖的环境。

## 官方微调入口

沿用官方 train.yaml 和 dual_system 模型配置：Wan2.2 TI2V 5B、joint_self_attn、80D action/state；AdamW lr=1e-4、betas=(0.9,0.95)、weight_decay=0.01、clip=1；5 epochs、每 GPU batch=16、累积=1；cosine/warmup=5%；bf16、ZeRO-2、gradient checkpointing；video/action loss 权重各 1。模块 LR 不单独覆盖，本地 bound_optimizer_groups 保持 false。

先用官方下载脚本取得 `OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model`：

```bash
python scripts/download_assets/download_openwam_checkpoints.py
```

选择 Foundation checkpoint，使用完整目录（含 config 和模型组件），然后运行官方训练脚本。以下 GPU 数 8 与权重路径是示例，需替换为实际资源：

```bash
NPROC_PER_NODE=8 bash scripts/train.sh \
  dataloader=nero_multitask \
  training.finetune_ckpt_path=/path/to/OpenWAM-Alpha-Pretrain-Foundation-Model \
  training.output_path=outputs/nero_multitask_alpha
```

多卡首次验证可在同一命令追加 `training.debug=true`，执行官方 20-step smoke，并使用单独输出目录；正式运行去掉该覆盖。单张 A100 80GB 本次已验证 batch=1 + CPU optimizer offload 的 3-step 测试，不能直接照搬默认 batch=16。精确命令见审计产物 `outputs/nero_audit_20260917/run_smoke.sh`。batch=16 是实际默认 YAML，官方说明中的 batch=24 为示例。8 GPU、每卡 16、累积 1 时，约 6417 steps/epoch，5 epochs 约 32,085 steps。实际以 Accelerate 构建后的 DataLoader 长度为准。

输出路径使用 training.output_path。默认每 2000 steps 保存，仅保留最后一个 checkpoint，不保存 optimizer 等完整续训状态；若需要意外中断后的周期点续训，显式设 `training.save_full_states_for_resume=true`。当前 trainer 正常完成时仍清理所有完整状态，仅保留权重，因此不保证完成后沿用最终 optimizer 继续训练。

沙盒外单卡已完成实际 Foundation 加载、forward/backward、optimizer 更新和最终 checkpoint 保存。三步约 67 秒（不含初始化和保存），仅是 smoke 吞吐，不代表默认多卡吞吐或收敛表现。模型结构和训练优化流程未修改。

## 部署契约

客户端发送相同 body/base、gripper_flange/tip、URDF 定义下的绝对 native EEF20 state，包含物理米制夹爪宽度；不能发送原始 16D 关节 state。服务端先用 absolute-state stats 归一化输入。模型生成后，官方 normalizer 先执行 80D gather 和动作反归一化，再由推理引擎对该次请求保存的参考状态还原：

```text
p_absolute = p_anchor + delta_p
R_absolute = R_anchor @ delta_R
width_absolute = clip(width_predicted, 0, 0.10)
```

生成 rot6D 使用 Gram-Schmidt 投影成合法旋转，退化旋转直接报错。动作块只还原一次，异步生成也使用生成起点的快照，不能取更新后的机器人状态。服务端返回绝对 EEF20，客户端不能再次加参考位姿或再次反归一化。若底层控制器接受 TCP 而非 flange，需要在机器人适配端应用固定工具变换。

服务端表示元数据对外声明 `representation=eef`、`model_representation=eef_relative`，并注明输出 frame 与夹爪米制范围。此接入没有新增真机控制器或修改硬件执行逻辑。

## 修改位置与回归检查

- `openwam/dataloader/nero_multitask.py`：独立 reader，复用官方公共 reader 钩子。
- `openwam/dataloader/utils/nero.py`：表示、相对/绝对转换、数据契约与指纹。
- `openwam/dataloader/utils/stats_computation/nero_multitask_stats.py`：全量审计及统计。
- `configs/dataloader/nero_multitask.yaml`、registry：官方 Hydra 接入。
- `scripts/data/prepare_nero_multitask.py`、`check_nero_multitask.py`：准备及 CPU smoke。
- 部署 engine、model_loader、obs_preprocess：相对位姿还原、统计校验、三相机要求。
- `tests/dataloader/test_nero_multitask.py`：非交换旋转、未来对齐与边界、左右相机/部署像素一致、独立 state/action normalizer、配置兼容、数据修订拒绝，以及生成起点快照。

```bash
pytest -q tests/dataloader/test_nero_multitask.py
```

官方参考：[训练默认配置](https://github.com/OpenWAM-Official/OpenWAM/blob/059cbfbfc068355f3d60fe9b014ff6469d1e94a3/configs/train.yaml)、[Alpha 微调指南](https://github.com/OpenWAM-Official/OpenWAM/blob/059cbfbfc068355f3d60fe9b014ff6469d1e94a3/assets/openwam_usage_docs/openwam-alpha-finetuning.md)。
