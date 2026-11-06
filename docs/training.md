# 最小训练链路

`inspect → recipe → prepare → stats → torchrun → 原生训练 → 完整产物 → 外层推理 runtime`。

## 数据与配方

`prepare_training.py inspect --dataset-root ...` 展示 metadata、有效窗口、字段及相机合同。原生 `nero_fk_v2` 使用 `observation.state`/`action` 16 维当前反馈（左 7 关节＋夹爪，右同理），左右 `observation.fk.fb.<left|right>.gripper_flange` 9 维 FK，及 `observation.images.head_fpv`、`left_hand`、`right_hand` 三路 RGB。坐标为 body→gripper_flange；位置及夹爪米制、Rot6D 为旋转矩阵前两列。原始 action 必须符合当前反馈语义，训练标签由未来 FK 构造，不能将任意控制命令列混入。

固定 60 Hz；每窗 33 帧，从第 1 帧开始预测 32 个未来状态，视频按 0,4,…,32 取帧。末尾不补齐虚假未来。相对模式使用同一窗口起点的 `p_future-p_anchor` 和 `R_anchor.T @ R_future`，绝对模式直接使用未来位姿；两者夹爪均绝对，proprio 也绝对。EEF20 按原生映射进入 80 维模型：0–9 与 34–43。

复制 `configs/training_recipes/nero_fk_v2.json` 到 `local_configs/`，设置基础权重、dataset_root、全新 output_dir、GPU 数、每卡 batch 和 GAS。路径支持环境变量；缺失变量直接拒绝。`action_mode` 可为 relative/absolute，默认 relative；`epochs=1`。固定字段/相机/窗口不支持通过任意覆盖改变。月饼 v3 原始数据需要其专用数据链路，见 tasks.md。

## 生成与执行

```bash
python scripts/prepare_training.py prepare --recipe local_configs/task.json
```

输出 `recipe.json`、`training_plan.json`、`<name>.yaml`、`train.sh` 和绑定文件摘要的 `prepared_run.json`。准备阶段不构建模型或启动训练，但会完整读取基础权重计算 SHA，大文件会产生磁盘读取成本。基础 checkpoint 同目录须存在原生 `config.yaml`，且保留其完整模型配置。当前新入口限 dual_system/joint_self_attn、wan22_ti2v_5b、80 维 action/state、冻结 T5/VAE，并需要本地 components/tokenizer；其他模型结构直接拒绝，不自动替换为默认模型。审阅文件后用选定的 `CUDA_VISIBLE_DEVICES` 执行脚本。脚本先 verify 绑定文件与来源，再检查 GPU、统计和 torchrun；固定解释器与源码位置，不透传改变统计域的任意参数。手改 recipe/YAML/脚本或基础权重会拒绝启动，需要重新 prepare。verify 同样会读取基础权重计算摘要。

统计可独立运行 `python scripts/prepare_training.py stats --recipe <run>/recipe.json`，生成 `stats/normalization_stats.npy`、审计及准备记录。复用原生 Nero 相对/绝对全窗口统计和数据验证，不改统计算法，不从历史任务复制 stats。

一轮以有效窗口计数：W=Σmax(episode_length−32,0)，每卡微 batch 为 B，GPU 数 G，梯度累积 A。Accelerate 每轮微步 M=ceil(W/(B×G))，会补齐 M×B×G−W 个窗口；优化器步为 ceil(M/A)，最后一组可能不足 A 个微步。计划分别记录这些量，不将 native `max_steps`（微步）误称优化器步。默认 `num_epochs=1, max_steps=null`，尾组遵守原生累积行为；GAS1 是最直接的一轮设置。

训练目录在 `checkpoints/<native timestamp>/` 下保存 `config.yaml`、统计、`checkpoint_step_N.safetensors` 及配置要求的配套资产。复制完整目录，并保留 recipe/plan/源码版本。默认新入口 warm-start，不自动选择某个历史 run 恢复。

## 加速与资源

在 recipe 设置 `speed_preset: "fast"` 可启用官方加速；默认 null 会写入生成 YAML，避免意外继承旧 shell 的 OPENWAM_PRESET。具体开关、重计算和 T5 offload 见 optimizations.md。输出配置仍需结合实际显存选择，官方四 H200 的数值不能当作本机吞吐。

保留的离线 VAE 缓存入口 `scripts/data/preencode_nero_video.py` 当前只支持 absolute Nero reader，固定 B64 的构建要求每卡至少 50 GiB 空闲显存，并校验完整 cache manifest。它使用原生 `--config-name`；不要直接把新生成 YAML 当成已注册的内置名字。此步骤需按真实资产另行准备配置并显式执行，不是默认训练准备的一部分，也不能在 4090 上照抄历史预编码命令。相对模型不复用 absolute-only 缓存入口。

现有 `velabot/training` 是更早的月饼源码身份绑定流程，保留用于解释历史模型；它要求旧 Git/patch 身份，不能在本次继续开发的普通源码副本中直接当新训练入口。
