> 迁入说明：本文从 VelaBot monorepo 的 OpenWAM 验收记录原样迁入集中仓；实现入口位于 `../openwam_front.py`，训练记录位于本目录的训练文档。历史 source 身份与 GPU 证据按原文保留。

# OpenWAM 推理接入记录

日期：2026-09-16。状态：**独立 HTTP front、完整 schema 握手、client/runtime 接线、
双 mode 短训练、旧权重三窗 HTTP parity 及两种 saved-weight GPU reload 已完成**。
本文件记录当前实现与证据边界，不把短测称为模型精度或真机验收。

## 当前实现

`OpenWAMCodec` 注册为 `openwam`，wire 名为 `openwam_native`，通过独立 HTTP front
连接 `POST /handshake` 和 `POST /infer`。front 位于
[`../openwam_front.py`](../openwam_front.py)，不替换
已有服务。客户端和服务端共同校验完整的 `velabot.io/v2` schema，服务端不会根据客户端
声明自行合成 schema。

固定 profile 为双臂、无底盘、左臂在前，每臂 `rot6d(9)+opening_width(1)`，共 20 维；
动作 H32、60 Hz，`wire_format=openwam_native`，语义为
`absolute_flange_rot6d_20d`。支持 `action_mode=eef` 与 `action_mode=delta`：两种 mode
对外都返回 absolute EEF20，delta 仅由服务端 engine 对六个 XYZ 维执行一次
`state[0]` addback；codec 不重复加回。

图像角色固定为 `head_fpv/left_hand/right_hand`，映射到服务端的
`head_camera/left_wrist_camera/right_wrist_camera`。客户端只把原始 JPEG bytes 做
base64 编码，不解码、缩放或重新压缩；服务端按训练 profile 生成 320×256 头部 slot、
两个 160×128 腕部 slot，再拼成 320×384 canvas。夹爪只做固定单位换算：
`g=2*w/0.1-1`、`w=0.05*(g+1)`，无 codec clip。

握手和每次响应冻结 checkpoint/config/stats 名称及 SHA256、固定 source repository/base
revision、patch SHA、源码文件 SHA、action mode 和完整 model contract。delta 必须声明
非空 patch/files fingerprint；eef 可以使用干净 base 或合法的已应用 patch。front 校验
固定 base HEAD、补丁可反向应用、登记源码实际 hash，以及未登记 tracked/untracked 变化。

front 读取实际 loaded cfg，要求 `dataloader.type=moon_cake`、
`normalize_mode=min-max`、所选 action mode、33 帧、stride 4、320×384 多视角、统一
80D 映射和三路 camera layout。eef 旧 stats 可以只有 `eef` block；delta 还要求
`delta` 和 `delta_state` 的六个 20D 统计项，且 `delta_state` 与 `eef` 逐项相等。
已有夹爪裁剪通过独立 `GripperRangeSafety` 保留并记录 requested/effective；这项保护
与模型归一化分开，不由 codec 隐式执行。

runtime 在 assemble、connect/configure 与 edge CLI 复核 OpenWAM 的固定 60 Hz，并拒绝
`obs_image_size`，避免客户端二次 resize。现有 `ObservationDeltaRemapper` 保留：它是
执行链的观测重锚，不能当作 server delta addback。raw absolute 动作的离线开环比较
不套 remapper。

## 使用前置与语义边界

profile 的 `task_pose` control frame 必须与训练 body frame 等价，并且 TCP 外参必须
确实对应 gripper flange；io schema 只描述列、单位和通道，不能证明这些外参。真实机器
人、GPU 精度、收敛、留出泛化和任务成功率仍需独立验收。HTTP front 使用官方 OpenWAM
checkout 的 `PolicyServer` 私有 `_policy/_executor/_action_buffer` 兼容接口，不复制
模型网络、优化器或训练代码。

官方 source 固定为 `OpenWAM-Official/OpenWAM`，base revision
`c171854a18a2acbeaadf46a746c4118c0e0c00f7`。本轮训练 patch 的 source fingerprint 和
实际文件清单由训练文档维护；patch artifact 必须位于 checkout 外。

## 已完成验收

- codec、front、model boundary 和真实 localhost HTTP fixture 共 39 项定向测试通过，
  覆盖双 mode、完整 schema、握手顺序、dt、identity、JPEG 原字节、单位换算、精确
  32×20 输出、单飞、队列清理和错误原因。
- runtime OpenWAM 定向测试 3 项通过，Ruff、文档布局和 codespell 通过。
- 独立 front import smoke 通过；wheel 构建并在临时独立目录解包导入通过，包内 front
  源码 hash 与工作树一致。

旧 absolute step3000 与 delta step4000 的三窗 HTTP parity 已完成：episode 78 的
t0/128/672 三窗各 9 次 generation，native 与修正基线、HTTP 原始 JPEG 与 direct bytes、
HTTP effective action 与显式 safety 结果均为 `max_abs=0`；delta 三窗的夹爪事件分别为
8/8/0，均与显式 safety 结果精确匹配。旧 absolute GPU probe 使用 seed=42、NFE=10、
compile/cache 关闭，GPU4 A100 进程 exit 0，peak allocated 为 25,096,087,552 bytes。
这些结果只证明指定 checkpoint 和窗口的数值闭包，不证明全量模型效果或真机行为。
两种 saved-weight GPU reload 已完成限定工程验收：stage14 generator 分别重载新 eef 与
delta step2，两个进程均 exit 0；episode 78 t0 输出均为 shape `[1,32,20]` 且 finite。
两种 mode 的 canvas SHA 为 `8696596...`、state SHA 为 `762def...`，实际参数均为
seed42、NFE10、video9、action_num_frames33、decode_video=false、cache=None、compile
关闭。full raw config 与仅 `project.output_dir` 的模板分离记录均精确匹配。

stage13 真实 preflight 已覆盖 4 个窗口 indices `0,2843,1165627,1167939`，118 个
video metadata 检查通过。stage14 的 generator 快照兼容仅分离已确认的 Hydra 训练日志
模板并保留完整 raw；两个 saved config 的 CPU 检查通过，其余无效 runtime 插值仍明确报错。

## 相关实现

- [客户端协议说明](../README.md)
- [独立 front README](../README.md)
- [训练接入记录](2026-09-16-openwam-training-design-draft.md)
- [验证规则](../README.md)
