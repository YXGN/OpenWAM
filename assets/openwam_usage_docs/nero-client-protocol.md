# Nero 客户端协议：openwam_nero/v1

首次部署请先阅读[客户端与服务端 Quickstart](nero-client-server-quickstart.md)。

本入口用于 Nero 的 `nero_multitask/eef_relative` 与
`nero_multitask_absolute/eef` checkpoint。相对与绝对是模型内部表示；两者
对外都使用 **body 坐标系、gripper_flange 末端的绝对 EEF20**。
客户端选择 `--protocol openwam_nero`。旧 MoonCake `openwam_native` 不适用。

## 服务端

在 OpenWAM 独立环境中运行；一个进程加载一份完整 checkpoint，不需要原训练路径：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m openwam.deploy.nero_chunk_server \
  --ckpt-dir models/nero_relative_8gpu_b64_20260919/step_8025 \
  --ckpt-name checkpoint_step_8025.safetensors \
  --device cuda:0 --host 127.0.0.1 --port 8849
```

跨机使用时显式选择监听地址。模型服务器不拥有机器人，也不会发送控制指令。
停止客户端策略、替换服务端 checkpoint、重新连接并握手后，再启动策略；不要热替换
执行中的动作来源。更换绝对 EEF 权重只修改 checkpoint 参数。

加载使用官方 `build_server_from_config`；预处理使用官方 `ObsPreprocessor`；
每次请求直接调用一次 `JointInferenceEngine.generate`。服务端没有逐步动作队列。
默认 10 步同步去噪、CFG=1、无 DiT 近似缓存、不解码预测视频，保留冻结 T5 prompt 缓存。
这与已审计的开环评估基线一致；没有改模型、采样算法或训练权重。
现已显式开启静态动作索引、请求内 context/KV 复用和联合 Transformer CUDA Graph。
优化范围与等价验证见[推理性能说明](nero-inference-performance.md)。追加
`--disable-inference-fast-path` 可对照原 eager 路径；首次推理需要图预热和捕获。
必要调参集中在 `configs/deploy_nero.yaml`，支持 `--config` 指定副本及 `--denoise-steps`
优先覆盖步数；服务和开环评估共用。改步数会改变预测，历史 10 步等价与性能结论不代表新步数。

### 终端日志

服务就绪时打印监听地址、checkpoint 路径与模型内部动作表示。每次成功推理只打印一行，
相同 prompt 连续使用时以编号引用；首次使用或切换时打印实际传入模型的完整原文。
例如（数字仅为格式示例，不是本次实测）：

```text
14:20:01 INFO infer id='8d249930aecd' prompt=P1 infer_ms=460.0 server_ms=485.0 backend=cuda_graph text='Pick up the hammer.'
14:20:02 INFO infer id='a115567199c2' prompt=P1 infer_ms=450.0 server_ms=473.0 backend=cuda_graph
14:20:04 INFO infer id='e97ff41ef1c9' prompt=P2 infer_ms=510.0 server_ms=537.0 backend=cuda_graph text='Pick up the charger.'
```

- `id`：客户端 request_id 的前 12 个字符，便于对照请求。
- `prompt=P…`：此服务进程中的 prompt 版本；A→B→A 会依次成为 P1、P2、P3。
  `text` 来自模型 conditions 中的 prompt，保留空格和中文；换行显示为 `\n`，避免日志多行展开。
  这是日志表示，不会修改模型输入；编号也不会改变 T5 缓存键。
- `infer_ms`：从调用官方引擎到动作已返回 CPU 并校验完成的墙钟耗时，包含实际执行的
  T5/VAE、去噪和动作后处理；不是单个 GPU kernel 的计时。
- `server_ms`：进入服务请求处理至响应数据构造完成，包含等待引擎锁、PNG 解码、
  观测预处理和上述推理；不含 HTTP 请求体接收/JSON 解析、响应 JSON 编码/发送、网络和客户端处理。
  响应字段 `server_total_sec` 同步采用该范围（秒）；旧版本仅从引擎调用前开始计时。
- `backend`：本次最后一个去噪步实际使用 `cuda_graph` 或 `eager`，不只是配置开关。

成功 `/infer` 不再重复打印 HTTP 200 访问日志；握手、重置与失败访问日志保留。
日志改动不增加 GPU 同步或额外模型调用。已运行的进程需停止客户端策略并重启服务端，
然后客户端重新握手才会使用新版本；原启动命令无需增加日志参数。

## 输入和输出

每臂顺序为 `[x,y,z,R00,R10,R20,R01,R11,R21,opening_width]`，左臂在前。
XYZ/夹爪用 m，rot6D 是旋转矩阵前两列按列排列。

- 输入 state：同一观测时刻的实测绝对 EEF20，未经归一化。
- 图像：原分辨率 RGB PNG。客户端角色 `head_fpv/left_hand/right_hand` 映射到
  `head_camera/left_wrist_camera/right_wrist_camera`。官方 BILINEAR 拼图为
  320×384，上方头部、下左左腕、下右右腕；客户端不得先缩放。
- prompt：实际任务原文，每次请求必填，不附加模板。
- 输出 actions：32×20 **绝对** EEF，对应观测之后第 1…32 帧，间隔 1/60 秒。
- 相对权重在官方引擎内按 `p=p_anchor+delta_p`、`R=R_anchor@delta_R`
  还原一次，anchor 是生成该块的原始 state；夹爪不做相对还原。
- 归一化、80 维槽位、反归一化只由模型自身 normalizer 处理。客户端不加载 stats。

`gripper_bounds=[0,0.1]` 是物理范围，非再次量化/归一化规则。上行实测宽度必须在范围内。
下行保持官方模型输出：相对模型沿用其已有还原/投影；绝对模型原始预测可能略超范围，
协议不新增隐藏裁剪。现有机器人运动层负责显式范围/速度约束，模型原始值应保留。

## HTTP 合同

JSON POST，`wire_format="openwam_nero/v1"`，最大请求 32 MiB：

1. `/handshake`：提交完整 `velabot.io/v2` schema 和固定合同。服务端独立校验双臂
   顺序、rot6D、opening_width、米制及范围，回显校验后的设备约束；返回完整模型
   identity、合同、H32、dt、session_id，明确无 RTC/hidden 功能。
2. `/infer`：提交当前 session_id、唯一 request_id、state、images（base64 PNG）、prompt。
   返回相同 request_id、生成 anchor_state、identity、actions。客户端核对身份、anchor、
   维度、有限值和旋转可解码性；不猜测协议、不补零或截断。
3. `/reset`：旋转 session_id。服务无动作/历史视频队列；客户端仍必须通过既有停止路径
   清空本地动作和作废在途旧结果。单个 reset 请求不能替代机器人停止。

identity 包含实际 checkpoint 名称、模型内部表示及 checkpoint/config/stats SHA256。
同一连接身份变化会拒绝；主动重连创建新 codec。模型计算串行，一个服务只用于一个
客户端会话；新握手使旧会话失效。

## 客户端执行约束

新协议的 composition 固定 PNG、三路角色；保留 Soma 同快照、时间对齐及新鲜度规则。
默认来源为 BGR 原始帧，PNG 解码后的 RGB 必须逐像素一致。仅有 JPEG 的来源再编码
为 PNG 不会恢复丢失信息，数据来源的压缩应单独记录。

关闭 epoch 动作重锚和 chunk-install bridge；已有 `None` 配置可表达这两项，不增加
第二套运动循环。动作队列、解码、权限、IK 和显式控制保护沿用现有链路。60 Hz 网格
和执行频率均固定为 60 Hz，逐个 pop 原始模型节点，不使用额外插值改变首个动作。
计划与运行入口校验 Nero MDH、mount、tool、关节顺序和固定 BASE 坐标，拒绝未经
重新审计的几何；仅 EEF20 schema 一致不能证明坐标语义一致。
模型节点、控制采样、限幅后的关节目标与实测反馈是不同结果，不能混用。

切换任务前必须停止策略；清空动作队列及 generation 后再用新任务启动。
新旧协议的观测/执行设置不同，进入或离开 `openwam_nero` 需新建 Runtime。
相对/绝对权重切换不需要更换协议或 Runtime，只需停止、重连新模型、再启动。

当前是同步 receding-horizon，动作耗尽后请求新块，仍会有等待区间。原 eager 基线
约 0.8 秒；新增优化的实测见推理性能说明。即使生成时间低于 32/60≈0.533 秒，
也不会自动变成推理与动作执行重叠的调度。现有调度会记录饥饿/迟到，不通过改变
动作时间单位、补未来图像或伪造动作来消除延迟。此版不承诺连续实时闭环控制。

## 可重复验证

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/audit_nero_client.py \
  --client-root /path/to/velabot_agentic_os \
  --ckpt-dir /path/to/checkpoint_directory --ckpt-name checkpoint_step_N.safetensors \
  --dataset /path/to/nero_multitask_lerobot_v2 --output /path/to/new_audit_directory
```

脚本通过 localhost 实际 HTTP，使用真实客户端 PNG 编码、codec、协商、动作队列和
ActionDecoder，与直接调用官方引擎逐元素比较；覆盖三任务、每任务三个 anchor。
不加载硬件 adapter、不创建机器人控制实例。CPU 回归另覆盖 observation builder、
运行时 composition、过期响应、错误角色/单位以及旧协议保持原行为。
