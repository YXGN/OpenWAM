> 历史开发/迁移记录，不是当前入口。此源码副本已继续开发，旧 Git/patch 身份与机器运行记录不再代表当前状态。新训练读模型根 README 和 docs/training.md；机器人服务统一使用外层推理项目。

# Nero 4090 推理迁移

把服务器当前 Atom 6248 模型搬到 Linux 单卡 4090，继续使用 `openwam` 协议、8017 端口。流程只有三步：**拉取独立快照 → 安装并校验 → 启动模型服务**。本地 agent 先读 [交接要求](AGENT_HANDOFF.md)。

快照包含当前工作区源码及未提交修改、完整 checkpoint、tokenizer、归一化统计、环境锁定文件和少量录制观测。模型文件约 24.8 GB；环境在本地重建。无需复制训练数据、训练缓存、其他权重或服务器 `.venv`。

## 在 4090 机器拉取

将下面的服务器地址换成平时 SSH 登录这台服务器所用的 IP 或别名。使用新的本地目录，避免与旧协议代码混合；命令可重复执行续传。

```bash
OPENWAM_REMOTE='用户名@服务器IP或SSH别名'
OPENWAM_HANDOFF_DIR='/path/to/nero_4090_handoff'
mkdir -p "$HOME/workspace/OpenWAM-nero-4090"
rsync -avh --partial --info=progress2 \
  "${OPENWAM_REMOTE}:${OPENWAM_HANDOFF_DIR}/" \
  "$HOME/workspace/OpenWAM-nero-4090/"
cd "$HOME/workspace/OpenWAM-nero-4090"
python3 scripts/check_nero_4090.py files
```

非默认 SSH 端口时给 rsync 增加 `-e 'ssh -p 实际端口'`。文件校验会读取整个 checkpoint，等待完成；退出码必须是 0。

## 安装并校验

先确认本地 Linux、NVIDIA 驱动、`nvidia-smi` 和 `uv` 可用。使用项目独立环境，按锁定依赖安装 Python 3.12.13、PyTorch 2.7.1/cu126 等，不复用全局环境。安装过程需要联网；模型推理使用包内资产离线运行。

```bash
bash scripts/setup_openwam_env.sh
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/check_nero_4090.py replay
```

分别检查两个命令的退出码。回放必须显示 `status: passed`，结果保存在 `outputs/nero_4090_validation/report.json` 和同名 `.npz`。先完成回放再启动服务，避免在 24 GB 卡上同时装载两份模型。CPU 需要容纳约 10.6 GiB 的 T5 权重及模型加载临时内存，建议准备 64 GB 主存，并在本机确认加载峰值；这不是已测得的最低配置。

回放走与 HTTP 服务相同的 `load_service`、观测处理和 `/infer` 边界，但在进程内调用，不连接机器人或现有 8017。它核对模型身份、参数放置、协议、夹爪映射、输出形状、重复输入和物理动作误差，并记录实际 GPU、加载时间、冷／热请求时间和显存峰值。

参考是 **2026-10-08 A100 的 episode 010 第 32／144 帧**，第 32 帧重复一次；三路原始 PNG、state 和 prompt 全部固定。它早于夹爪减 2 mm 改动，因此原始预测保留，比较时仅补当前的 `clip(raw - 0.002, 0, 0.1)`。这份历史记录的 checkpoint/config/stats 哈希与当前模型相同，**不是本轮重新采集的在线响应，也不是任务成功率测试**。

默认迁移检查上限：左右手每一步的位置差 ≤ 2 mm、姿态角差 ≤ 1°、夹爪差 ≤ 1 mm；本机同输入重复最大差 ≤ 1e-6。它们是保守的工程回归门槛，尚未通过 4090 校准，不代表机器人安全或任务容差。若超限，保留报告，先定位 CPU T5／GPU 内核差异，不自动调宽门槛或宣称已经对齐。

## 启动服务

```bash
bash scripts/serve_openwam_atom_4090.sh
```

等待 `ready ... wire=openwam/v1 ... denoise_steps=4 ... gripper_close_offset_m=0.002 text_encoder_cpu=True`。需要后台运行时可把同一条命令放进本地 tmux。单卡默认为 GPU 0；多卡机器用 `NERO_ATOM_GPU=实际编号` 选择卡。

客户端继续选择 `--protocol openwam`，服务器地址改为 4090 机器的 IP，端口 `8017`；同机客户端用 `http://127.0.0.1:8017`。省略 `--obs-image-size`，使用已完成协议更名的客户端。没有引入新 codec，也不需要另装一份机器人客户端到纯模型服务器。

T5 在 CPU 上首次处理新 prompt 可能较慢，同 prompt 命中缓存后不再编码。回放与正式服务是两个进程，不共享内存中的 prompt 缓存。若首次请求超时，按实测编码耗时调整客户端请求超时，不能通过删改 prompt 或切换动作格式解决。

## 与服务器保持一致的内容

| 项目 | 固定值 |
|---|---|
| 模型资产 | `outputs/nero_atom_4gpu/2026-10-04_15-00-03/` 的 6248 权重、config、stats、tokenizer，逐文件 SHA256 校验 |
| 推理实现 | 同一 `openwam.deploy.nero_chunk_server` → 官方引擎 → 模型 `generate` |
| 推理条件 | BF16、4 步同步联合 video/action 去噪、seed 42、CFG 1、fast path 和请求内缓存开启 |
| 图像 | 三路原分辨率 RGB PNG，沿用既有拼接／裁切／缩放逻辑，模型输入高 384、宽 320 |
| 状态／动作 | body/gripper_flange 绝对 EEF20；左手在前；Rot6D 为旋转矩阵前两列；米制位置和夹爪 |
| 时间 | 从下一帧起 32 步，`dt_sec=1/60`；这不是 HTTP 请求频率 |
| 夹爪 | 双手连续宽度只减一次 0.002 m，再裁剪到 `[0, 0.1]`；关闭旧二值阈值 |
| 推理输出 | 仍联合预测动作和视频 latent；`decode_video: false` 仅省去未来 RGB 解码 |

4090 入口只有两处执行差异：**T5 驻 CPU 并把 embedding 送回 GPU；CUDA Graph 关闭**。checkpoint 精度、模型层、预处理、动作重建及归一化实现均不替换。CPU 与 CUDA、A100 与 4090 不承诺浮点逐位一致，因此必须保留数值回放检查。开启 CUDA Graph、量化、改变分辨率／步数属于后续优化，应另存配置和验证结果。

## 状态和维护

服务器当前 8017 进程未被本轮重启或握手；已读到的运行设置仍为 6248／4 步／原 GPU 路径。4090 的实际显存、速度和数值回放需在目标机器完成，不能以本机单元测试替代。已执行的软件检查见随包携带的 `source_validation.json`，本地结果写入 `outputs/nero_4090_validation/`。

服务端维护者如需生成新快照，在完整源码项目运行：

```bash
.venv/bin/python scripts/package_nero_4090.py --output outputs/一个新的快照目录
```

快照目录必须不存在；代码独立复制，大权重在源服务器内使用硬链接节省磁盘，通过 rsync 到 4090 后是普通文件。**不要在源服务器的快照中原地修改或截断 checkpoint**。`manifest.json` 固定了本次文件和环境；文件变化后应生成新版本，不修改 manifest 来绕过校验。
