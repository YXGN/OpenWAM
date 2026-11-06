> 历史开发/迁移记录，不是当前入口。此源码副本已继续开发，旧 Git/patch 身份与机器运行记录不再代表当前状态。新训练读模型根 README 和 docs/training.md；机器人服务统一使用外层推理项目。

# OpenWAM VelaBot 训练与推理集成入口

本目录集中维护 OpenWAM 的 VelaBot 训练、开环工具和独立 HTTP front：

- [`training/README.md`](training/README.md)：stats/preflight、eef/delta 训练和开环工具；
- [`training/prepare_moon_cake.py`](training/prepare_moon_cake.py)：生成 stats 与预检报告；
- [`training/train_moon_cake.py`](training/train_moon_cake.py)：dry-run 与 torchrun 训练入口；
- [`training/generate_openloop_moon_cake.py`](training/generate_openloop_moon_cake.py)：从 checkpoint 生成开环 NPZ；
- [`training/openloop_validate_moon_cake.py`](training/openloop_validate_moon_cake.py)：评估开环 NPZ；
- [`client/README.md`](client/README.md)：VelaBot client/runtime patch 的应用说明；
- [`docs/2026-09-16-openwam-inference-design-draft.md`](docs/2026-09-16-openwam-inference-design-draft.md)：推理接入证据；
- [`docs/2026-09-16-openwam-training-design-draft.md`](docs/2026-09-16-openwam-training-design-draft.md)：训练接入证据。

本集成布局使用 OpenWAM root 下的 `velabot/` 目录，不替换已有 VelaBot monorepo 服务。
当前集成 source 是 `YXGN/OpenWAM` base `059cbfbfc068355f3d60fe9b014ff6469d1e94a3`；
文档中保留的 `OpenWAM-Official/OpenWAM` c171 及旧 GPU 结果属于历史 legacy checkout，
不应改写为当前集成 revision。

入口是 `openwam_front.py`，提供 `GET /health`、`POST /handshake` 和 `POST /infer`。
HTTP wire 中的三路 base64 JPEG 使用 native 角色 `head_camera`、`left_wrist_camera`、
`right_wrist_camera`；客户端 codec 将 `head_fpv`、`left_hand`、`right_hand` 映射到这三个
名字。服务直接用 LANCZOS 生成 320×256 的头部 slot 和两个 160×128
腕部 slot，再放入 320×384 画布；不会先把所有图像缩到画布大小。

启动前复制 `config.example.json`，并用下方 `fingerprint_source` 命令生成并填入本次 source
身份，再填入 checkpoint、config、stats 文件名及其 SHA-256 摘要。示例中的全零摘要只是
模板，不能直接用于启动：
然后在固定 base revision 的 OpenWAM checkout 根目录中运行：

```bash
# 在已经安装固定 OpenWAM revision 及其 GPU/model 依赖的独立环境中执行
pip install -e /path/to/OpenWAM/velabot
python /path/to/OpenWAM/velabot/openwam_front.py \
  --config /path/to/openwam-front.json \
  --openwam-root /path/to/OpenWAM \
  --ckpt-dir /path/to/checkpoint \
  --ckpt-name checkpoint_step_3000.safetensors
```

启动时会对 checkpoint、保存的 config、normalization stats 逐文件计算 SHA-256，并核对
实际 checkout 的 source identity；配置中的摘要不能代替实际文件验证。集成布局使用
`repository=YXGN/OpenWAM`、`base_revision=059cbfbfc068355f3d60fe9b014ff6469d1e94a3`、
实际 HEAD revision，以及 `velabot/source_files.json` 登记的完整文件 hash。legacy 外部
布局仍兼容 `OpenWAM-Official/OpenWAM`/c171 四字段身份。训练 patch 的 SHA-256 为
`99ddce32bc18f3fd974193dd5c69f95a49bae475c438faf1f84780253c8faed0`；使用该 patch 的
`eef` 或 `delta` 配置都必须在 source 中登记补丁 SHA-256 和实际源码文件 SHA-256。集成布局
默认使用仓库内登记的 `velabot/patches/openwam-moon-cake-training.patch`；传入 `--patch` 时
必须指定同一文件。legacy 外部布局才通过 `--patch` 传入 checkout 外的补丁文件。服务拒绝清单外的 tracked 或
untracked 源码变化。服务还会读取加载后的官方 cfg，确认 `moon_cake`、`min-max`、所
登记的 `eef` 或 `delta`、33 帧、320×384、多视角设置与本入口一致。两种 mode 的 wire
都返回绝对 flange EEF20；delta 的 XYZ addback 由 engine 恰好执行一次。模型输出中的
有限 native 夹爪值交给独立 `GripperRangeSafety` 模块裁剪到 `[-1,1]`，响应逐项记录
requested、effective、row、channel、reason 和 selected intervention；shape/NaN/Inf
仍是协议错误。旧服务保持不变。

本地测试使用 CPU fake adapter，验证完整 schema、三路图像、20 维 state、32×20 输出、
握手顺序、真实 HTTP JSON 边界和实际错误原因。它不证明 GPU 模型精度或机器人闭环行为。

历史 legacy `OpenWAM-Official/OpenWAM` c171 + patch 验收记录：eef 与 delta 均已完成双卡、每卡 batch 1、两步真实训练；两种 mode 的
rank0 均报告 `saved_samples_equal_post_step=true`。旧 absolute step3000 与 delta step4000
在 episode 78 的 t0/128/672 三窗 HTTP parity 中，native/HTTP/safety 对比均为
`max_abs=0`，delta 三窗夹爪事件 8/8/0 均精确匹配。stage13 的 4 个 preflight 窗口与
118 个 video metadata 检查通过；stage14 两个 saved config 的 CPU generator 快照兼容
检查通过。两种 saved-weight GPU reload 的限定工程验收已完成：stage14 generator 分别
重载新 eef 与 delta step2，进程均 exit 0，episode 78 t0 输出均为 `[1,32,20]` 且 finite；
canvas/state SHA 分别为 `8696596...`/`762def...`，full raw config 与仅
`project.output_dir` 的模板分离记录精确匹配。上述 GPU 结果属于 legacy c171 + patch；这些证据不代表全参数验证、全视频解码、
新模型精度、长训练或真机闭环通过。

来源边界：HTTP front 本身只调用官方 OpenWAM 仓库 `OpenWAM-Official/OpenWAM` 的
`PolicyServer` 兼容接口；它不复制优化器或训练代码。官方 base revision 为
`c171854a18a2acbeaadf46a746c4118c0e0c00f7`。本目录调用其私有 `_policy`、`_executor`、
`_action_buffer` 实现一次完整动作段的兼容接线；这些是当前固定官方 revision 的实现
细节而非上游公开 API。训练 patch、reader 和训练入口的边界见
[`训练接入记录`](docs/2026-09-16-openwam-training-design-draft.md)。

集成 checkout 的 source 身份由 `velabot/source_identity.py` 从实际 Git checkout 生成，不能
手填 revision 或文件摘要。训练、服务配置和产物 manifest 应使用该命令输出的 JSON；不要把
它预先写回 `config.example.json`，否则会形成自引用摘要：

```bash
PYTHONPATH=/path/to/OpenWAM/velabot python - <<'PY'
import json
from pathlib import Path
from source_identity import fingerprint_source

root = Path('/path/to/OpenWAM').resolve()
identity = fingerprint_source(
    root,
    root / 'velabot/patches/openwam-moon-cake-training.patch',
    (),
)
print(json.dumps(identity, ensure_ascii=False, indent=2, sort_keys=True))
PY
```
