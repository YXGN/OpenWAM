> 历史开发/迁移记录，不是当前入口。此源码副本已继续开发，旧 Git/patch 身份与机器运行记录不再代表当前状态。新训练读模型根 README 和 docs/training.md；机器人服务统一使用外层推理项目。

# OpenWAM MoonCake 训练与开环工具

本目录提供 MoonCake LeRobot v3 数据转换、CPU 预检和显式训练启动入口。集成仓的 source
identity 使用 `YXGN/OpenWAM`、base `059cbfbfc068355f3d60fe9b014ff6469d1e94a3` 和实际
HEAD；legacy 外部 checkout 仍使用官方 c171 四字段身份。集成 checkout 已包含完整修改，位于
[`patches/openwam-moon-cake-training.patch`](../patches/openwam-moon-cake-training.patch)。
该 patch 新增 MoonCake reader/config、delta 部署 addback 和按需启用的 128M
optimizer 分组，并在官方 registry/trainer/engine 的明确位置接线。集成 checkout 中 patch
已应用，可用下面的命令验证：

```bash
cd /path/to/OpenWAM
git apply --reverse --check velabot/patches/openwam-moon-cake-training.patch
git rev-parse HEAD
git diff --check
```

若从固定 base 重新构建集成 checkout，只在干净 base 上 apply 一次该 patch，再复制或生成
`velabot/` 目录。服务和训练入口会核对 `velabot/source_files.json`、patch 摘要以及所有
登记文件。使用实际 checkout 生成完整 source JSON 的方式见总入口
[`velabot/README.md`](../README.md)；JSON 中的 revision 和 `files_sha256` 必须使用命令输出，
不要从示例配置抄写。

`action_mode=eef` 使用绝对 body-frame flange EEF20；`action_mode=delta` 仅对双臂 XYZ
相对窗口第 0 行 state，rot6d 和夹爪保持绝对值。两者都使用 33 行窗口、32 行动作、60 Hz、
视频行 `0,4,...,32` 和统一 80D 映射 `0-9/34-43`。proprio 永远使用 `eef` stats，delta
输出在反归一化并 gather 回 20D 后用同一 raw state 只给六个 XYZ 维做一次 addback。

先生成新 stats 和预检报告；脚本不会加载 foundation model，也不会启动训练：

```bash
python velabot/training/prepare_moon_cake.py \
  --dataset /datasets/moon_cake_eef_flange_lerobot_v30 \
  --stats-output /runs/moon-cake-eef/normalization_stats.npy \
  --report-output /runs/moon-cake-eef/data-verified.json \
  --action-mode eef --code-root /path/to/OpenWAM
```

该 stats artifact 同时写入 `eef`、`delta`、`delta_state` 三个 blocks；eef 与 delta 训练示例可复用
同一份已完成预检的 stats。

启动入口要求 dataset、stats、foundation、output、GPU 和 mode 全部显式给出。`--dry-run`
会先验证 source/artifact，再运行官方 `scripts/train.py --cfg job --resolve` 做实际 Hydra
配置解析；该模式不加载模型、不启动训练进程、不创建 output。本首切片固定 gradient accumulation 为 1，
不支持 resume：

```bash
python velabot/training/train_moon_cake.py \
  --dataset /datasets/moon_cake_eef_flange_lerobot_v30 \
  --stats /runs/moon-cake-eef/normalization_stats.npy \
  --foundation /weights/OpenWAM-foundation/checkpoint_step_154000.safetensors \
  --output /runs/moon-cake-delta \
  --gpu 4,6 --code-root /path/to/OpenWAM --action-mode delta \
  --batch-size 1 --max-steps 2 --save-steps 3 --keep-last-k-ckpts 1 --dry-run
```

`--save-steps` 默认为 2000，`--keep-last-k-ckpts` 默认为 1；上例显式设置的是两步
GPU smoke 配方。保存内容只保留可部署权重，`save_full_states_for_resume=false`，本切片不支持
resume；foundation 通过 `finetune_ckpt_path` warm-start，不将其称为 resume。

stats 报告保存数据集解析路径、episode/action/state hash、derivation hash、固定窗口与
视频索引、统计代码身份、有限性/shape/alignment 检查和夹爪 safety intervention 的
episode/requested/effective 值。`q01/q99` 仅为旧 artifact 兼容字段，等于 min/max，不表示真实分位数。
报告中的 `alignment_anomalous_episodes` 只表示 action[t] 与 state[t+1] 对齐检查；原始轨迹保留，
不会自动删除 104 或 154 集，其他轨迹异常不在本工具的检测范围内。

训练入口调用官方 `scripts/train.py`，仅传入本 profile 所需的 Hydra overrides。输出目录
应保存官方 config、stats、deploy 权重以及本次 checkout revision 和 patch hash；不应将
机器、数据或权重的私人绝对路径提交到公共源码。

训练 manifest 和预检报告使用统一的 `source` 结构。legacy 使用四字段
`repository`、`revision`、`patch_sha256`、`files_sha256`，revision 固定为官方 c171；集成
仓额外包含 `base_revision`，它固定为 059 base，而 `revision` 必须是当前 checkout 的实际
40 位 HEAD。缺少 patch 或任一已修改文件会直接失败。
`openloop_validate_moon_cake.py` 评估外部 predictions NPZ（`actions`、`episode_index`、
`start`），沿用 reader 的整数累计视频 offset，并以原始 `action[start:start+32]` 作为 GT。
默认 predictions 已经是引擎反归一化后的 absolute EEF20；只有显式指定
`--prediction-semantics xyz_delta_eef20` 才会做一次 XYZ addback。该工具不启动模型、不宣称
训练质量，误差计算不做 clip/filter。需要固定窗口时可加
`--episodes 78 --starts 0,128,672`（单 episode 会展开到多个 starts）；省略时使用 NPZ
自身的 episode/start metadata。指定这两个参数时必须与 NPZ metadata 完全一致；若只想抽取
NPZ 中的若干行，使用 `--window-indices`，工具不会重标 episode/start。
评估器命令行里的 seed、denoise steps、compile 只会标记为 `caller_declared`；使用生成器
输出的同名 manifest 可追溯实际 adapter 配置。

从已保存 checkpoint 生成这三个窗口的最小 probe（输出 NPZ 及同名 `.manifest.json`）：

```bash
python velabot/training/generate_openloop_moon_cake.py \
  --ckpt-dir /weights/moon-cake-eef --ckpt-name checkpoint_step_2000.safetensors \
  --front-config /weights/moon-cake-eef/service-config.json \
  --code-root /path/to/OpenWAM --source-patch /path/to/OpenWAM/velabot/patches/openwam-moon-cake-training.patch \
  --dataset /datasets/moon_cake_eef_flange_lerobot_v30 \
  --stats /runs/moon-cake-eef/normalization_stats.npy \
  --episodes 78 --starts 0,128,672 --output /runs/moon-cake-eef/openloop.npz \
  --device cuda:0
```

该 probe 复用 front 的 `load_config`/`OfficialAdapter`，固定 seed=42；每窗只将 reader
返回的第 0 张当前 canvas 和 state 交给模型。manifest 将 CLI `declared` 参数和实际
`architecture.generate` kwargs、完整的 `loaded_config_unresolved` 和 resolved deploy config
分开保存，并分别记录 `reader_stats` 与 checkpoint 的 `model_stats/model_config`。Hydra
训练日志模板 `project.output_dir=outputs/${now:%Y-%m-%d_%H-%M-%S}` 只在 metadata 副本中
移除并记录到 `excluded_non_runtime_fields`，不会生成当前时间替代值；其它配置插值仍严格
解析并在失败时终止。输出语义固定为 `absolute_eef20`，不会额外 delta addback、clip 或
filter。
