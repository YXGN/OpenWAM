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
  --device cuda:0 --host 0.0.0.0 --port 8017
```

使用相对 EEF 的 **step4000 中间权重**时，改用下面的完整命令：

```bash
cd /path/to/OpenWAM
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m openwam.deploy.nero_chunk_server \
  --ckpt-dir models/nero_relative_8gpu_b64_20260919/step_4000 \
  --ckpt-name checkpoint_step_4000.safetensors \
  --device cuda:0 --host 0.0.0.0 --port 8017
```

step4000 与 step8025 命令二选一，使用同一 GPU 和端口；切换前先停止客户端策略，
再停止旧服务。新服务加载完成后客户端重新握手，后端地址仍为 `http://MODEL_HOST:8017`。
这里的step4000是已迁移的**相对EEF**权重，不是本机绝对EEF训练的同名保存点。
示例显式使用8017端口；省略 `--port` 时程序默认仍为8849。

`CUDA_VISIBLE_DEVICES` 选择物理 GPU，`--device cuda:0` 指该进程内的第一张可见卡。
同机使用可将 `--host` 改为 `127.0.0.1`；跨机客户端填写模型机可达地址，不能填
`0.0.0.0`。首次加载会读取并校验约24 GiB权重，等待加载完成再握手。
此服务没有 `/health`；用下一步的真实协议握手判断是否就绪。

换绝对 EEF 权重时，只替换 `--ckpt-dir` 与 `--ckpt-name` 为其完整部署目录和文件名。
服务端从保存点识别 `eef_relative` 或 `eef`；客户端始终收发绝对 EEF20。

包含推理性能优化的新版服务端默认启用请求内 context/KV 复用、静态动作索引和
CUDA Graph。首次推理含预热/捕获；之后日志的 `backend=cuda_graph` 表示该次最后一个
去噪步实际重放了图。需要原 eager 对照时，在服务端命令追加
`--disable-inference-fast-path`。客户端参数不变；停止策略、重启服务、重新握手后生效。
逐项输出等价性与耗时证据见 OpenWAM 的 `nero-inference-performance.md`。

Nero 服务和开环评估默认读取 OpenWAM 的 `configs/deploy_nero.yaml`，不同权重可共用：

```yaml
denoise_steps: 4
inference_fast_path: true
cuda_graph: true
request_context_cache: true
```

日常只需修改 `denoise_steps`。优化总开关关闭时，两个子开关不生效。
也可复制该文件，在服务端命令追加 `--config /path/to/deploy_nero.yaml`；
临时试 8 步时追加 `--denoise-steps 8`，优先于 YAML。原来的
`--disable-inference-fast-path` 同样优先于 YAML。配置只在启动时读取，修改后需停止策略、
重启服务并重新握手；启动日志的 `denoise_steps` 是实际值。
减少去噪步数会改变预测，需开环评估；不会改变每块 32 个动作或客户端 60 Hz 约定。
权重目录中的 `config.yaml` 及通用 `configs/deploy.yaml` 都不是这份调参文件。

### 分阶段耗时诊断

在上述服务端启动命令的环境变量前加 `WAM_PROFILE=1`，例如把
`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8` 改为
`WAM_PROFILE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8`，其余参数保持原样。
每次成功推理会在原有 prompt/总耗时日志外，多打印一行 `timing_ms`，单位均为毫秒：

| 字段 | 范围 |
|---|---|
| `request` / `response` | 服务端读取请求体与 JSON 解析 / JSON 序列化与写入 socket |
| `queue` | 等待服务端推理锁，含锁前的轻量校验 |
| `images` / `obs` | Base64、PNG 解码 / 三相机与 state 观测预处理 |
| `setup` | 模型输入准备、噪声与调度器初始化等 |
| `t5` | 文本编码或缓存读取；`t5_cache=hit/miss` 标明本次是否命中 |
| `vae` | 当前图像准备与 VAE 编码（Nero TI2V 路径） |
| `denoise` | 全部去噪步，含请求内 context/KV 准备；首次也含图预热与捕获 |
| `action` | 动作回传 CPU、反归一化及相对 EEF 还原 |
| `other` | 其余校验、动作列表转换、日志等未单列开销 |
| `total` | 上述各阶段总和；`graph_captures` 是本请求新增的图捕获次数 |

`total` 从服务端开始读取请求体到响应写入 socket；不包含请求头接收、客户端采图、
编码、完整网络往返或机器人执行。它比原日志 `server_ms` 多含 HTTP 收发处理。
阶段采用 GPU 同步后的墙钟时间，可包含资源等待，并非纯 GPU 内核时间。
详细计时会引入额外同步开销，用于定位问题；比较日常延迟时移除 `WAM_PROFILE=1`
或设为 `0`。开关在服务启动时读取，不改变去噪步数、缓存策略或动作语义。

## 3. 客户端先做无硬件握手检查

在机器人机的新终端执行：

```bash
cd /path/to/velabot_agentic_os
export NERO_BACKEND=http://MODEL_HOST:8017
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

- 换任务：先在控制台停止策略，等待停止完成，在**推理面板 → 改任务描述**保存任务原文，再启动。
- 换权重：先停止策略，再关闭客户端进程；在服务端终端 Ctrl-C，使用新保存点重启服务；
  然后重新执行 `plan` 和 `run`。这是最容易确认旧队列已清空的操作顺序。
- 已熟悉控制台时，同协议换权重也可以停止后重新连接后端；必须完成新握手再启动。
- 相对与绝对权重切换时，客户端协议、state、夹爪单位及频率均不变。
- 不要把 HTTP `/reset` 当作停止机器人；停止通过客户端已有策略生命周期完成。

推理面板发送 `inference.set_task`，修改后下一次请求使用新的 prompt，通常不需重启
服务端或重新握手。录制面板的同名按钮发送 `record.set_task`，只修改录制任务标签，
**不会切换模型 prompt**。新协议运行中改任务会被明确拒绝；必须先停止并确认修改成功。

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
| chunk之间有停顿 | 当前为同步receding-horizon，动作耗尽再请求；优化减少计算耗时，但没有引入执行与推理重叠 |
| 夹爪预测略超[0,0.1] | 绝对模型可能原始越界；协议保留原值，现有运动层约束负责实际目标 |

当前已有18个真实模型窗口的客户端/服务端回放一致性证据，**尚未完成真机闭环测试**。
启动命令已按CLI核对；实时跟踪、现场相机身份与任务成功率需通过真机验收。
详细协议与审计见OpenWAM的 `nero-client-protocol.md`、`nero-client-audit-20260920.md`，
以及客户端的 `docs/inference/openwam-nero.md`。
