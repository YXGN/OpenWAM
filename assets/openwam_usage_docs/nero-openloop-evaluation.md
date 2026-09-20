# Nero episode 开环评估

独立进程评估已保存的相对 / 绝对 EEF checkpoint。评估输入仅为当前三路 RGB、当前绝对 EEF20 和数据集原始任务文本；GT 为随后 32 帧的 FK 位姿和夹爪宽度。评估没有优化器，不修改训练配置或数据集。

## 复用边界

沿用官方 OpenWAM 的模型加载与推理结构：

- [`load_from_checkpoint_dir`](../../openwam/deploy/model_loader.py)：自包含 checkpoint 加载、eval 模式、dtype、部署 normalizer。
- [`merge_deploy_cfg`](../../openwam/deploy/server.py)：训练配置与显式推理参数合并，保留视频 9 帧 / 动作原始窗口 33 帧的区别。
- [`ObsPreprocessor`](../../openwam/deploy/obs_preprocess.py)：训练同款三相机拼图。
- [`JointInferenceEngine.generate`](../../openwam/deploy/engine.py) → `architecture.generate`：完整联合去噪，返回 32 步动作。
- 我们已有的 [`nero.py`](../../openwam/dataloader/utils/nero.py)：EEF 组装、数据合同、相对旋转与绝对位姿还原。

官方对应文件已核对：[engine.py](https://github.com/OpenWAM-Official/OpenWAM/blob/main/openwam/deploy/engine.py)、[model_loader.py](https://github.com/OpenWAM-Official/OpenWAM/blob/main/openwam/deploy/model_loader.py)、[obs_preprocess.py](https://github.com/OpenWAM-Official/OpenWAM/blob/main/openwam/deploy/obs_preprocess.py)。运行使用当前项目内的这些模块，保留我们已审计的 Nero 语义扩展，不在运行时下载或切换官方分支。

不依赖 `velabot/`、旧 MoonCake probe、机器人 HTTP/WS adapter、动作执行缓冲或二值夹爪兼容逻辑。评估配置拒绝 delta、binary_action_dims、不同相机顺序或不同有效槽位。

## 命令

在项目根目录使用 `.venv/bin/python`。必须明确指定已完成写入的 checkpoint 文件，不能使用不断变化的 latest。`--output` 必须是新目录。

三任务完整 episode（stride=32，补最后一个完整窗口，共 176 个窗口）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 .venv/bin/python -m openwam.evaluation.nero_openloop \
  --ckpt-dir outputs/nero_absolute_4gpu/2026-09-20_02-15-44 \
  --ckpt-name checkpoint_step_2000.safetensors \
  --dataset /data2/xieweize/project/dataset/lerobot_v2/nero_multitask_mooncake_hammer_charger_408_fk_lerobot_v2 \
  --episodes 51 242 347 --stride 32 --verify-parity \
  --output outputs/nero_openloop/absolute_step2000
```

示例中的 GPU 和保存点需要在运行时确认仍可用；训练只保留最近保存点，旧 step2000 可能随后被正常清理。评估启动前应选取不会在加载期间被清理的已完成保存点。

单窗口短测：在命令中添加 `--starts 0`，对每个所选 episode 的起始观察分别推理。也可 `--starts 0 128 672`，这些锚点应用于每个所选 episode，超出范围直接报错。

CPU 数据预检：添加 `--audit-only`，检查合同、全数据集 fingerprint、GT 归一化往返和当前图像，保存 `audit.npz` 与拼图，不加载 GPU 模型。

相对 EEF：命令不变，替换为相对 checkpoint 的目录和文件。动作表示完全由保存的配置与统计文件决定，CLI 不允许覆盖。目标机最终保存点目录是 `/data3/xieweize/openwam_runs/nero_multitask_8gpu_b64/2026-09-19_00-15-03`，文件 `checkpoint_step_8025.safetensors`；需要在具有该权重及数据的机器执行。

## 训推语义

| 项目 | 约定 |
|---|---|
| 观察 | episode 时刻 t 的头部、左腕、右腕图像，以及绝对 EEF20 |
| 相机布局 | head_fpv 上方；left_hand 下左；right_hand 下右；384×320 |
| 时间 | 60 Hz；GT 为 t+1…t+32，不使用 source action16 作为未来标签 |
| 物理空间 | body 坐标系，left/right gripper_flange，位置与夹爪为 m |
| rot6D | 旋转矩阵前两列，按列排列 |
| 有效维度 | 左臂 0…9、右臂 34…43，共 20 维；不统计剩余 60 维 |
| 相对动作 | p_future−p_t，R_tᵀR_future；整个 chunk 固定锚点；夹爪绝对 |
| 状态统计 | 对应 checkpoint 的 `<action_mode>_state` |
| 动作统计 | 对应 checkpoint 的 `<action_mode>`；不重新统计 |
| 推理 | 同步去噪 10 步，CFG=1，seed=42，可显式指定 seed / 去噪步数 |
| 推理优化 | 官方文本 embedding 缓存；compile、DiT 跳步缓存关闭；视频解码关闭 |

关闭视频解码不代表去掉视频分支，联合视频—动作去噪仍执行。训练 VAE 的完整未来视频缓存不进入评估，当前图像按官方推理路径编码。只有当前状态、图像、任务文字和显式推理参数进入 `conditions`。

`PredictionRecorder` 只委托现有 normalizer 并复制输出用于诊断，同时透传 `_dst_index` / `_unify_dim`，不改变无效槽位的去噪约束。`--verify-parity` 会对第一个窗口额外执行一次没有记录器的真实推理，要求逐元素相等；结果写入 manifest。

原始 native 预测在引擎进行相对旋转投影 / 夹爪裁剪之前保存。主报告使用现有部署输出；不对绝对模型额外裁剪或平滑。SO(3) 指标投影 rot6D，原始退化旋转数量单独记录，不伪造零误差。

## 产物与解读

```text
<output>/
├── report.html          # 20 维曲线，episode/窗口/动作空间选择，SVG 导出
├── predictions.npz      # 所有窗口的 GT、原始/部署预测、归一化预测、状态与时间
├── predictions.csv      # 逐窗口、逐 horizon、逐维；物理单位 m
├── metrics.json         # 位置 mm、旋转角 °、夹爪 mm、逐维 MAE/RMSE 与诊断
├── manifest.json        # 权重/配置/stats/source 哈希、运行参数、窗口索引、耗时
└── observations/        # 实际输入拼图 PNG
```

用浏览器打开 `report.html`，复制分享时携带整个输出目录；页面不访问 CDN 或上传数据。图表中的位置和夹爪乘 1000 显示为 mm，rot6D 无单位；CSV 和 NPZ 保持 m。

Episode 总览在重叠区使用最新锚点的预测，不平均；稀疏锚点之间未覆盖的时间不连接曲线。NPZ 和单窗口视图保留全部原始 chunk。主指标按窗口×horizon 统计，末尾补窗口会产生重叠计数，manifest 明确记录；另按 episode、task 分组。

保持当前位姿/夹爪的基线与模型使用同一 GT。静止窗口上基线可能优于模型，不能只选择 episode 开头的静止窗口展示精度。夹爪越界比例需要与越界幅度 `gripper_violation_mm` 一起看，极小的负宽度也会计入比例。

当前选择的 episode 已参与训练，这是训练集轨迹拟合检查，不是留出泛化或机器人任务成功率。不同训练步数 / 数据遍数的相对与绝对模型不应直接按训练 loss 排名；开环物理误差也需要相同窗口和推理参数。

## 最小维护范围

新增 `openwam/evaluation/` 的入口、指标、报告和测试。本次同时纳入此前绝对 EEF 部署已使用的 `nero_absolute.py` 和 loader 合同校验作为必要依赖；不捆绑未提交的训练缓存、训练配方、训练 reader 注册等改动。

验证记录见 [nero-openloop-audit-20260920.md](nero-openloop-audit-20260920.md)。
