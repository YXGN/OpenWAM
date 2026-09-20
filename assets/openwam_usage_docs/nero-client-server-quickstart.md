# Nero 客户端与服务端 Quickstart

适用版本：OpenWAM `6cdfd88`、VelaBot `86a72b3` 或包含这些功能的后续版本。
协议为 `openwam_nero/v1`，客户端参数为 `--protocol openwam_nero`。

本文在两个仓库同步维护：OpenWAM 的
`assets/openwam_usage_docs/nero-client-server-quickstart.md` 与 VelaBot 的
`docs/inference/openwam-nero-quickstart.md`。修改命令或合同后同时更新两份。

## 1. 准备目录与环境

服务端运行在有 GPU 的模型机，客户端运行在连接 Nero 的机器人机；可以是两台机器。
示例中的 `/path/to/...`、`MODEL_HOST` 和 GPU 编号需替换为实际值。

服务端在 OpenWAM 仓库根目录使用已有 `.venv`。新机器尚无环境时，先按项目锁定版本安装：

```bash
cd /path/to/OpenWAM
bash scripts/setup_openwam_env.sh
```

部署目录必须来自同一保存点，包含完整模型权重及配套资产：

```text
checkpoint_directory/
├── checkpoint_step_N.safetensors
├── config.yaml
├── normalization_stats.npy
└── tokenizer/
```

使用已迁移并校验的完整目录；不需要优化器状态，也不需要把训练数据集放到机器人机。
不要混用两种动作表示的 config/stats，不要手工修改 `action_mode` 来切换模型。

客户端使用独立的 VelaBot 环境。已有部署环境可跳过安装；新安装参考客户端
`docs/operations/installation.md`，直连 RealSense 的 Nero 需要对应依赖组：

```bash
cd /path/to/velabot_agentic_os
uv sync --frozen --group realsense
export VELABOT_PYAGXARM_PATH=/path/to/pyAgxArm
```

机器人机需已有对应 SDK、CAN 配置及现场确认过的 Nero profile。下面运行命令用
`uv run --no-sync`，沿用已安装环境。模型机的 PyTorch 环境不代替机器人机的 SDK 环境。

## 2. 启动服务端

在模型机执行；示例使用已迁移的相对 EEF 最终权重：

```bash
cd /path/to/OpenWAM
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m openwam.deploy.nero_chunk_server \
  --ckpt-dir models/nero_relative_8gpu_b64_20260919/step_8025 \
  --ckpt-name checkpoint_step_8025.safetensors \
  --device cuda:0 --host 0.0.0.0 --port 8849
```

`CUDA_VISIBLE_DEVICES` 选择物理 GPU，`--device cuda:0` 指该进程内的第一张可见卡。
同机使用可将 `--host` 改为 `127.0.0.1`；跨机客户端填写模型机可达地址，不能填
`0.0.0.0`。首次加载会读取并校验约24 GiB权重，等待加载完成再握手。
此服务没有 `/health`；用下一步的真实协议握手判断是否就绪。

换绝对 EEF 权重时，只替换 `--ckpt-dir` 与 `--ckpt-name` 为其完整部署目录和文件名。
服务端从保存点识别 `eef_relative` 或 `eef`；客户端始终收发绝对 EEF20。

## 3. 客户端先做无硬件握手检查

在机器人机的新终端执行：

```bash
cd /path/to/velabot_agentic_os
export NERO_BACKEND=http://MODEL_HOST:8849
export NERO_PROFILE=profiles/robots/agilex-nero-dual-arm.yaml

uv run --no-sync vela-run plan \
  --robot "$NERO_PROFILE" --adapter agilex --device none \
  --backend "$NERO_BACKEND" --protocol openwam_nero \
  --arm-format rot6d --strategy receding_horizon --policy-hz 60
```

检查输出包含 `action_dim=20`、`backend=openwam_nero`、`horizon=32`、`rtc=False`，
且没有阻塞原因。`plan` 读取配置并向服务端握手，不打开机器人或相机，也不执行模型推理。
它不能证明现场相机接线、图像新鲜度和机器人控制可用。

**一个服务只维持一个客户端会话。** `plan` 也会建立新会话，使原会话失效；不要在
另一个客户端执行策略期间运行它。完成检查后再启动下面的正式客户端。

## 4. 设置任务，启动客户端

继续使用上一步终端。示例为数据集中插头任务的原文，保留换行：

```bash
NERO_TASK=$(cat <<'TASK'
Grasp and lift the plug with the left hand.
Align the position and orientation of the plug with the middle socket using the left hand.
Insert the aligned plug into the middle socket with the left hand until the plug is fully seated.
TASK
)

uv run --no-sync vela-run run \
  --robot "$NERO_PROFILE" --adapter agilex --device none \
  --backend "$NERO_BACKEND" --protocol openwam_nero \
  --arm-format rot6d --strategy receding_horizon \
  --policy-hz 60 --rate-hz 60 \
  --task "$NERO_TASK" \
  --console-host 127.0.0.1 --console-port 8080 \
  --out data/nero_openwam
```

`run` **会连接真实机器人与相机**。此命令未加 `--autostart`，策略初始为停止状态；
打开机器人机上的 `http://127.0.0.1:8080`，检查三路图像、任务和后端连接，再在控制台
启动策略。`--device none` 表示没有遥操接管设备；需要已有 Pico 接管链路时改为
`--device pico`，其连接方式沿用现场配置。

录制在控制台单独启动；`--out` 只指定目录，不代表已经录制。
`--arm` 才表示启动时开启一段录制，`--autostart` 才表示连接后自动启动策略。
`--dry-run` 仅禁止策略动作执行，仍建立硬件连接，不用于无机器人回放。

三路角色必须分别是头部 `head_fpv`、左腕 `left_hand`、右腕 `right_hand`。不要设置
`--obs-image-size`；新协议发送原分辨率 PNG，服务端按训练布局缩放拼图。
网格与执行频率均固定60 Hz；不要用变更频率来掩盖推理等待。

## 5. 停止、换任务与换权重

- 换任务：先在控制台停止策略，等待停止完成，修改任务原文，再启动。
- 换权重：先停止策略，再关闭客户端进程；在服务端终端 Ctrl-C，使用新保存点重启服务；
  然后重新执行 `plan` 和 `run`。这是最容易确认旧队列已清空的操作顺序。
- 已熟悉控制台时，同协议换权重也可以停止后重新连接后端；必须完成新握手再启动。
- 相对与绝对权重切换时，客户端协议、state、夹爪单位及频率均不变。
- 不要把 HTTP `/reset` 当作停止机器人；停止通过客户端已有策略生命周期完成。

## 6. 无机器人联合回放验证（可选）

在模型机执行以下脚本。它自行启动临时 localhost 服务并加载模型，**不连接第2步的
常驻服务**，需要单独可用的 GPU 显存；`--output` 必须是尚不存在的新目录。

```bash
cd /path/to/OpenWAM
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/audit_nero_client.py \
  --client-root /path/to/velabot_agentic_os \
  --ckpt-dir models/nero_relative_8gpu_b64_20260919/step_8025 \
  --ckpt-name checkpoint_step_8025.safetensors \
  --dataset /path/to/nero_multitask_mooncake_hammer_charger_408_fk_lerobot_v2 \
  --output outputs/nero_http_replay_new
```

成功产物为 `result.json`（`status=passed`、9个窗口均 `exact_equal=true`）与
`predictions.npz`。链路包含数据集观测、客户端编码、实际 HTTP、真实模型、响应校验、
动作队列及解码，并与官方引擎直调比较；没有真实执行后反馈的闭环。

## 7. 语义与常见问题

每臂为 `[xyz3, rot6D6, opening_width1]`，左臂在前，共20维；body坐标、
gripper_flange末端、位置与夹爪均用米，rot6D为旋转矩阵前两列按列展开。
服务端负责归一化与相对还原；客户端不重复还原、不做[-1,1]夹爪映射，也不额外重锚。

| 现象 | 检查 |
|---|---|
| 连接拒绝或超时 | 模型是否加载完成、服务监听地址、端口与跨机可达性 |
| wire/schema/identity不匹配 | 两端均使用新协议；权重、config、stats是否配套；换模型后重新连接 |
| stale session | 是否另开过plan或第二个客户端；停止后重新连接，勿并用同一服务 |
| geometry不匹配 | profile是否改变MDH、mount、tool、父链或关节顺序；不同几何需重新审计 |
| 无请求或观测被跳过 | 三路角色、状态与图像新鲜度、时间对齐、任务是否为空 |
| chunk之间有停顿 | 当前为同步receding-horizon；32帧跨度约0.533秒，实测每块请求约0.8秒 |
| 夹爪预测略超[0,0.1] | 绝对模型可能原始越界；协议保留原值，现有运动层约束负责实际目标 |

当前已有18个真实模型窗口的客户端/服务端回放一致性证据，**尚未完成真机闭环测试**。
启动命令已按CLI核对；实时跟踪、现场相机身份与任务成功率需通过真机验收。
详细协议与审计见OpenWAM的 `nero-client-protocol.md`、`nero-client-audit-20260920.md`，
以及客户端的 `docs/inference/openwam-nero.md`。
