> 迁入说明：本文从 VelaBot monorepo 的 OpenWAM 训练记录原样迁入集中仓；训练入口位于 `../training/`，推理记录位于本目录的推理文档。历史 source 身份与 GPU 证据按原文保留。

# OpenWAM MoonCake 训练接入记录

日期：2026-09-16。状态：**数据、eef/delta 双 mode 短训练和旧权重三窗 HTTP parity
已完成；两种 saved-weight GPU reload 的限定工程验收也已完成**。本文只记录当前工具和实际证据，不宣称全
参数验证、全视频解码、新模型精度或真机效果通过。

## 固定实现与入口

本轮采用官方 OpenWAM base revision
`c171854a18a2acbeaadf46a746c4118c0e0c00f7`，训练 patch SHA256 为
`99ddce32bc18f3fd974193dd5c69f95a49bae475c438faf1f84780253c8faed0`。patch 涉及 9 个
登记源码文件；部署身份使用 `repository`、`revision`、`patch_sha256` 和完整
`files_sha256`，不能用任意 dirty checkout 代替已应用 patch。patch 文件放在官方
checkout 外，避免把 patch 自身计入源码 hash。

当前入口为：

- `../training/prepare_moon_cake.py`：数据、stats 和 CPU 预检；
- `../training/train_moon_cake.py`：受限训练启动；
- `../training/generate_openloop_moon_cake.py`：checkpoint 开环生成；
- `../training/openloop_validate_moon_cake.py`：固定窗口误差检查。

工具固定双臂 body-frame gripper flange、XYZ/rot6d/gripper 20D、统一 80D 映射
`0-9/34-43`、33 行窗口、32 行动作、60 Hz 和视频行 `0,4,...,32`。`eef` 使用绝对
动作；`delta` 只将左右臂 XYZ 相对窗口 state[0]，rot6d 和 gripper 保持绝对。proprio
两种 mode 都使用 eef stats，服务端只执行一次 XYZ addback。当前工具范围是固定窗口的
XYZ、左右臂轴、SO(3)、gripper 和 offset 检查。

运行环境复用已核验的 Python 3.10.0 环境；torch CUDA build 实测为 2.7.1+cu126，未
另建环境。私人机器、数据、权重路径不写入公开文档或 manifest 示例。

## 当前数据证据

全量 494 episodes 的数值、action/state 对齐和视频 metadata 检查已通过。另对 64 个
代表窗口完成视频解码；独立 delta episode 78、104、154 的首/中/末 9 窗归一化检查已
通过。上述解码是指定窗口证据，不等于全视频逐帧解码，也不等于全量模型输入验证。

数据不因本轮工具自动删除 episode；104 与 154 的保留、留出划分和效果比较属于后续
模型评估决定。reader 继续使用整数累计视频 offset，不用固定帧率换算 episode 起点。

## 当前训练证据

eef 已完成双卡、每卡 batch 1、两步的真实训练 smoke。已观察并记录 global/optimizer
step=2、finite loss/gradient、video/action/proprio 抽样更新、frozen 抽样保持恒定，
以及 `saved=poststep`。这证明受限训练链路能完成一次真实 forward/backward/update 和
保存边界，不证明全参数更新、收敛或新模型精度。

eef 与 delta 均已完成双卡、每卡 batch 1、两步真实训练；两 rank 的 optimizer/global
step 均为 2，loss/grad 有限，video/action/proprio 抽样 trainable 组更新、frozen 抽样
不变，且两 mode 的 rank0 `saved_samples_equal_post_step=true`。这证明受限训练链路的
更新和保存边界，不证明全参数验证、收敛或新模型精度。

旧 absolute step3000 与 delta step4000 的三窗 HTTP parity 已通过：episode 78 的
t0/128/672 三窗各 9 次 generation，native 与修正基线、HTTP 原始 JPEG 与 direct bytes、
HTTP effective action 与显式 safety 结果均为 `max_abs=0`；delta 三窗夹爪事件为 8/8/0，
均精确匹配。旧 absolute GPU probe 使用 seed=42、NFE=10、compile/cache 关闭并正常退出
（GPU4 A100，peak allocated 为 25,096,087,552 bytes）。这些结果仅覆盖指定 checkpoint
与窗口。两种 saved-weight GPU reload 已完成：stage14 generator 分别重载新 eef 与
delta step2，两个进程均 exit 0；episode 78 t0 输出均为 shape `[1,32,20]` 且 finite。
两种 mode 的 canvas SHA 为 `8696596...`、state SHA 为 `762def...`，实际参数均为
seed42、NFE10、video9、action_num_frames33、decode_video=false、cache=None、compile
关闭。full raw config 与仅 `project.output_dir` 的模板分离记录均精确匹配。

stage13 真实 preflight 覆盖窗口 indices `0,2843,1165627,1167939`，118 个 video
metadata 检查通过。stage14 generator 快照兼容只分离已确认的 Hydra 训练日志模板并
保留完整 raw；两个 saved config CPU 检查通过，其余无效 runtime 插值仍报错。

## 归一化、保存与后续评估

profile 固定 `min-max`；`q01/q99` 是旧产物兼容字段时的 min/max 复制，不表示真实
分位数。delta stats 的 action 只对六个 XYZ 维做窗口起点变换，`delta_state` 保持与
eef state stats 一致。夹爪单位和显式 safety 记录与归一化算法分开。

当前工具只支持固定窗口和有限的动作/状态诊断。三集全面重评、固定留出、静/动分层、
chunk 接缝、repeat-state baseline 以及模型效果阈值属于后续模型效果评估，不是本轮
训练接入的通过条件。严格 resume、全参数验证、全视频解码、真机闭环和新模型精度也
尚未声明通过。

## 相关文档

- [推理接入记录](2026-09-16-openwam-inference-design-draft.md)
- [独立 front README](../README.md)
- [训练目录说明](../training/README.md)
- [验证规则](../README.md)
