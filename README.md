# OpenWAM 机器人训练与模型实现

本仓库保留 Nero 月饼、原子任务的模型和数据实现，新增可复用的训练准备入口。机器人推理统一交给外层 `stable_real_world_inference` 项目；本仓库提供模型、数据变换、训练和原生引擎。

先读 [AGENTS.md](AGENTS.md) 和 [训练说明](docs/training.md)。历史任务差异见 [任务约定](docs/tasks.md)，优化来源和开关见 [加速说明](docs/optimizations.md)。

本仓库可独立克隆和训练，不依赖父推理仓库：

```bash
git clone --branch agent/training-workflows https://github.com/YXGN/OpenWAM.git
cd OpenWAM
```

已有克隆直接使用当前仓库的环境脚本和 prepare 入口。对于已有 `nero_fk_v2` 合同，新任务只需核对数据、动作域和资源，填写本地 recipe 并生成训练脚本，无须新增 trainer。其他机器人或数据合同需先补独立适配与验证，不能只重命名字段。

最少需要准备：

- 完整本地 Nero LeRobot v2 FK 数据：60 Hz、原生三相机、state16/action16 当前反馈、左右 FK9，以及任务 metadata 和匹配的 URDF。
- 完整 OpenWAM Foundation 目录中的 checkpoint、原生 `config.yaml` 和本地 tokenizer；当前入口要求 Wan2.2 TI2V 5B、dual-system joint-self-attention、统一 action/state 80 维。
- 独立 OpenWAM Python 环境、新输出目录、GPU 数、每卡 batch 和 GAS；明确 relative/absolute。默认 1 epoch、不开 fast preset 或 VAE cache，不保证特定显存可容纳模型。


## 最小训练

复用本模型独立环境；新环境可使用 `scripts/setup_openwam_env.sh` 和 `envs/requirements-cu126.lock`。该脚本会安装依赖，训练准备入口本身不安装、不下载模型、不执行训练。

```bash
# 以下在模型仓库根目录运行，python 为 OpenWAM 环境。
python scripts/prepare_training.py inspect --dataset-root "$DATASET_ROOT"
```

当前入口支持 `nero_fk_v2`：本地 LeRobot v2、双臂 FK、三相机、60 Hz。它会展示并核对原生字段及相机映射；其他数据格式需先使用匹配的转换器，不会自动猜测。

```bash
export DATASET_ROOT=/path/to/exported_nero_dataset
export OPENWAM_FOUNDATION_CHECKPOINT=/path/to/foundation/checkpoint_step_154000.safetensors
export TRAIN_OUTPUT_DIR=/path/to/new_training_run
# foundation checkpoint 同目录需有原生 config.yaml。
python scripts/prepare_training.py prepare --recipe configs/training_recipes/nero_fk_v2.json
# 审阅生成的 recipe.json / training_plan.json / YAML / train.sh 后：
CUDA_VISIBLE_DEVICES=0 bash "$TRAIN_OUTPUT_DIR/train.sh"
```

默认相对 XYZ＋相对 Rot6D、绝对夹爪，`action_mode=absolute` 切换绝对位姿。固定 33 帧窗口产生 32 步未来动作，默认 1 epoch、每卡 batch=1、梯度累积=1。资源参数按实际显存及用户资源选择；batch=1 也不是 24 GB 显存可训练的承诺。

脚本先调用现有原生全窗口统计，再通过 torchrun 启动原生训练。训练加速已集成但 preset 默认关闭；在本地 recipe 中显式指定 `"speed_preset": "fast"` 启用。没有在本轮执行实际 GPU 训练或测量性能。

## 主要目录

```text
scripts/prepare_training.py          inspect / prepare / stats
configs/training_recipes/            新训练配方模板
configs/train*.yaml                  原生及历史任务配置
openwam/dataloader/                  Nero、月饼等数据与统计语义
openwam/model/                       模型、MoT、Wan、推理缓存与加速内核
openwam/train/                       原生训练循环
openwam/deploy/                      外层 runtime 复用的引擎与合同
presets/fast.env                     显式训练加速预设
velabot/                            旧月饼集成源码证据，非新任务入口
```

旧采集工作簿、会话日志、过时服务状态记录已从源码树移出并保留迁移前备份。历史月饼训练代码与原子配置仍保留，动作语义不会被静默改写。`openwam/deploy/nero_chunk_server.py` 中被外层 runtime 复用的加载逻辑继续保留。

上游：[OpenWAM-Official/OpenWAM](https://github.com/OpenWAM-Official/OpenWAM)。保留 LICENSE、CITATION.cff、上游模型与仿真工具。训练加速代码来自官方 PR #58，ResearchBoost-Agent 是改进来源，不是运行依赖。
