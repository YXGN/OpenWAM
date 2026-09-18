# Nero 多任务训练链路审计（2026-09-17）

审计对象为当前工作区的 `dataloader=nero_multitask` 接入，使用官方 Alpha Foundation 原始权重、默认 dual_system / joint_self_attn 模型和官方训练入口。没有使用本地 MoonCake reader、delta 定义或 bounded optimizer groups。源数据未修改。

## 结论及范围

数据路径、字段语义、时间对齐、左右臂槽位、三相机映射和归一化的检查通过；已完成单张 A100 80GB 上的 3-step 真训练和 checkpoint 保存。该结论是实现及数值链路可运行，不是策略收敛或实机任务成功率结论。

本次发现并修复的防护缺口：原 `validate_info` 只检查 16D state / 9D FK 的形状，没有检查字段名称顺序。现在准备统计和 reader 初始化均要求精确的 state/action/FK 名称顺序，防止后续数据改版仍维数相同却静默错取夹爪或旋转列。当前真实数据的顺序原本正确，因此无需改变统计文件。

仍需知悉的事项：

- 月饼 prompt 保留 `[box destination]` 占位符。程序不能从该文案得知目的地；已向用户确认实际含义，确认前不猜测替换。三个任务分别为 120 / 79 / 51 个 tokenizer tokens，均未达到 512 token 截断上限。
- 本接入改变了官方预训练的动作语义：输出相对位姿、绝对夹爪，proprio 仍为绝对位姿。权重和张量形状兼容，但不能据此假定未经微调即可零样本执行。
- `action == observation.state`，本数据能监督未来已实现的轨迹，不能恢复原始控制指令、接触力或控制延迟。
- 当前验证不包括完整视频逐帧解码、重新验证物理相机接线/机器人标定、实机执行、长期收敛。按用户要求，没有验证集，也没有任务均衡。

## 路径、表示和槽位

| 项目 | 核验结果 |
|---|---|
| 数据根目录 | `/data2/xieweize/project/dataset/lerobot_v2/nero_multitask_mooncake_hammer_charger_408_fk_lerobot_v2` 存在，408 episodes / 834400 frames |
| 存储格式 | LeRobot v2.1，每 episode 独立 parquet / 三路 mp4；chunk 与 episode 路径按 metadata 模板解析 |
| 统计文件 | 项目根目录下 `outputs/nero_multitask_data/normalization_stats.npy`；官方 `scripts/train.sh` 会切至项目根目录 |
| 状态来源 | `observation.fk.fb.{left,right}.gripper_flange` 的 xyz + rot6D，夹爪取 `observation.state[7]` / `[15]` |
| 坐标/旋转 | body 坐标系，左右 gripper_flange；rot6D 是旋转矩阵前两列依次拼接 |
| Native 20D | 左 xyz3 / rot6D6 / width1，然后右 xyz3 / rot6D6 / width1 |
| 模型 80D 槽位 | 左 xyz `0:3`，rot6D `3:9`，gripper `9`；右 xyz `34:37`，rot6D `37:43`，gripper `43`（Python 半开切片） |
| 未使用槽位 | 干净输入填 0，action/proprio mask=false；action loss 分母为 32×20 有效单元，而非 32×80 |
| 部署状态 | 必须输入绝对 native EEF20；不是原始 16D 关节 state |

数据 fingerprint 校验覆盖 metadata / URDF / parquet 内容和视频 size/mtime。保存 checkpoint 内的统计文件与训练源统计 SHA256 完全一致：`65d5877e34384817e18f77faf0f90feefe8439ff7c74de58893be0a194d93afe`。部署从 checkpoint 目录读取统计，不依赖启动目录中的相对 stats 路径。

## 时间、相机和读数

对每个窗口起点 t：

```text
proprio        = EEF[t]，绝对位姿和绝对夹爪
action[k]      = [p[t+k]-p[t], rot6d(R[t]ᵀ R[t+k]), width[t+k]]
k              = 1..32，左右臂分别计算，所有步均相对同一个 t
video frames   = t, t+4, ..., t+32
first image    = video[0]
```

XYZ 是 body 轴上的差值，不额外乘 `R[t]ᵀ`；旋转相对定义则必须做矩阵乘法。不能把两个 rot6D 向量相减，也不能逐步累加动作块。动作 60 Hz，32 步约 0.533 秒；9 帧视频约 15 Hz。仅采完整 33 帧窗口，没有跨 episode 或末尾复制标签。

| 数据相机 | 画布槽位 | 部署字段 |
|---|---|---|
| `observation.images.head_fpv` | 上方 320×256 | `images.head_camera` |
| `observation.images.left_hand` | 左下 160×128 | `images.left_wrist_camera` |
| `observation.images.right_hand` | 右下 160×128 | `images.right_wrist_camera` |

总画布宽 320、高 384，原视频 640×360 RGB，统一一次 BILINEAR 缩放拼图。训练与部署像素一致的回归检查通过；读数错误不会换成随机样本，缺相机不会静默填黑。

真实数据读取覆盖所有 408 episodes 的首、中、末各 1 个窗口，共 1224 窗口、33048 次相机帧解码。逐窗口检查有效槽位、prompt、proprio、未来 +1 动作和形状，使用独立矩阵还原后与源 FK 比较，最大分量误差 `1.7881393e-7`。图片拼图另经人工查看。

自然窗口数为月饼 489722、锤子 201146、插座 130476，合计 821344。右臂在插座任务中接近静止，归一化后的右臂差值方差接近零与源轨迹一致，不是漏读或左右映射错误。

## 归一化全量检查

独立重新计算 821344 个窗口 × 32 个未来步，共 26283008 行相对动作。检查了 157698048 个 action XYZ 数值和 4928064 个当前 state XYZ 数值：

- action / state 的 min、max 与保存统计逐项完全一致，范围正且有限。
- 训练 action XYZ / state XYZ 超出范围而被裁剪的数量均为 0；不作为训练起点的 episode 尾部 state XYZ 也均在范围内。
- 相对 action 和绝对 proprio 分别使用 `eef_relative` / `eef_relative_state` 统计。reader 已归一化，模型 `prepare_inputs` 不再归一化一次。
- rot6D 使用固定 min=-1/max=1，维持恒等变换；全量相对旋转正交误差最大约 `2.68e-7`。
- 夹爪使用共享物理范围 0–0.10 m，映射至 -1–+1。连续夹爪不二值化，不按任务各自拉伸。宽度 0 / 0.05 / 0.10 m 对应 -1 / 0 / +1。
- min-max 范围不对称，零位移归一化后依次约为 `[0.0852,-0.2904,0.0693,-0.2378,0.0869,-0.2643]`。这是正常仿射偏置；模型输出 0 不能直接解释为物理零位移。
- 普通任务/活动臂的抽样归一化 XYZ 标准差约 0.025–0.091，未发现所有动作被异常极值压至同一个常量的问题。统计范围是全任务共享，任务间幅度差异保留。

另模拟模型实际使用的 bf16 标签转换，覆盖全部 26283008 行 XYZ 标签，再按相同 min-max 还原。仅这一精度转换造成的最大绝对误差为 **0.542 mm**，各轴平均绝对误差 **0.012–0.144 mm**。该误差不包含网络预测误差，不能当作最终定位精度。若之后目标明确要求亚毫米插接，需要单独评估精度和表示选择；本次没有擅自改变官方 bf16 或统计定义。

## 训练、权重和保存

Foundation 文件与官方 Hugging Face 仓库的 LFS SHA256 对照一致（另逐项核验 config/tokenizer 文件）。权重 SHA256：`180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d`。虽然本机缓存目录名字含 MoonCake，内容实际是已核验的官方 Foundation，未使用已有 MoonCake 微调权重。

沿用官方模型和训练算法：Wan2.2 TI2V 5B + ActionDiT，80D action/state；AdamW lr=1e-4 / betas=(0.9,0.95) / weight_decay=.01 / clip=1；bf16、ZeRO-2、gradient checkpointing，video/action loss 权重各 1。文本编码器和 VAE 冻结，Video DiT / ActionDiT / proprio encoder 全参数微调；实际 trainable 为 6021155088 参数。

单卡 smoke 使用 GPU 2（A100 80GB），仅覆盖以下设置以缩短测试并适配显存：batch=1、num_workers=0、CPU optimizer offload、initialize_model_on_cpu=true、3 steps、constant LR、最终保存。其余模型和 loss 保持默认；`bound_optimizer_groups=false`。这不是原样 batch=16 / 5 epochs / cosine 的完整训练测试，也没有验证八卡 DDP 或精确断点恢复。

| Step | Total loss | Video loss | Action loss |
|---|---:|---:|---:|
| 1 | .2048 | .1502 | .0547 |
| 2 | .1922 | .1907 | .0015 |
| 3 | .4864 | .3217 | .1647 |

3 步约 67 秒（不含模型初始化、保存），成功完成 forward/backward/optimizer 和最终 checkpoint 写入。训练采样噪声和窗口不同，这三个 loss 不用于判断收敛趋势。

逐张量核验保存权重：所有张量有限；ActionDiT 的 674 个、Video DiT 的 683 个、proprio encoder 的 2 个张量发生实际更新；文本编码器 242 个张量及 VAE 196 个张量全部保持完全相同。checkpoint 附带 config、tokenizer、normalization_stats，可由官方加载入口重新构建。

从新 checkpoint 使用 `load_from_checkpoint_dir` 重新加载成功，再使用 `merge_deploy_cfg` / `ObsPreprocessor` / `JointInferenceEngine` 完成三任务实际生成。episode 0、204、306 分别使用中间窗口，三相机画布与 reader 像素一致，部署归一化 state 与训练 proprio 在 bf16 量化容差内一致。每次输出均为 `[32,20]`，全部有限、绝对位姿 rot6D 正交、夹爪在 0–0.10 m 内。推理使用 2 个 denoise steps，关闭 compile / DiT cache / 视频解码，仅验证执行链路，不是正式 10-step 推理效果测试。结果见 `reload_inference.json`，三组实际输出保存在 `inference_episode_*.npy`。

默认正式配方仍是每卡 batch=16 / 5 epochs / cosine+5% warmup，应按实际 GPU 数和显存配置运行。单卡 smoke 成功不代表原默认 batch=16 可以单卡运行；本模型仅单卡 Adam FP32 master+一二阶状态便约需 67.3 GiB，再加模型、梯度及激活，CPU offload 是本次单卡测试的必要适配。`initialize_model_on_cpu=true` 当前只跳过 trainer 的后置搬运，不保证 checkpoint 构建全程不使用 GPU。

保存语义也需区分：默认只保留最近权重，不含 optimizer/scheduler/RNG；`save_full_states_for_resume=true` 会周期保存完整状态，但当前 trainer 正常结束时仍清理所有 `accel_state_step_*`。它支持意外中断后的周期点恢复，不保证正常完成后还能沿用最终 optimizer 继续训练。

## 复核产物

所有审计产物位于 `/data2/xieweize/project/OpenWAM/outputs/nero_audit_20260917/`（忽略的运行产物，不提交 24 GB 权重）：

```text
nero_audit_20260917/
├── numeric_audit.json / numeric_audit_script.py       # 全窗口位置统计、范围和旋转检查
├── reader_all_episodes.json / reader_audit_script.py  # 408 episodes 真实读取
├── bf16_audit.json / bf16_audit_script.py             # bf16 标签量化误差
├── prompt_tokens.json                               # 三任务文本长度
├── foundation_verification.json                     # 官方权重/配置/tokenizer 校验
├── foundation_remote_tree.json                      # 官方仓库文件哈希快照
├── run_smoke.sh / train.log                          # 精确启动命令与训练输出
├── checkpoint_audit.json / checkpoint_audit_script.py# 全量权重有限值/更新/冻结检查
├── reload_inference_script.py / reload.log           # 官方 checkpoint 重载和实际生成
├── reload_inference.json / inference_episode_*.npy   # 三任务输出形状与数值检查
└── train_smoke/2026-09-17_16-48-37/                   # step 3 checkpoint 和部署资产
```

CPU 回归：数据、归一化、部署及微调配置 103 项通过；action loss / masks / proprio / padding 52 项通过，总计 **155 passed**。字段顺序相关的新增 4 项测试检查维数不变但重排名称时必须拒绝。
