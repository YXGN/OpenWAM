# OpenWAM 独立训练环境

环境位于 `/data2/xieweize/project/OpenWAM/.venv`，激活提示符为 `openwam`。
Python 解释器独立安装在项目的 `.cache/python`；虚拟环境关闭系统包和用户包继承。
项目以 editable 方式安装，修改源码后直接生效。只有官方 Foundation 大权重通过链接复用，
Python、训练依赖和编译扩展均不借用其他项目的环境。

## 使用

```bash
cd /data2/xieweize/project/OpenWAM
source scripts/activate_openwam.sh
python --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'

# 检查最终配置；不开始训练。
bash scripts/train_nero_8gpu.sh --print-config

# 8 张 GPU 资源齐备后启动正式训练。
bash scripts/train_nero_8gpu.sh
```

直接运行 8 卡脚本时也会自动选择 `.venv`。`OPENWAM_PYTHON` 可显式覆盖解释器，
通常无需设置。环境不是 Conda 环境，不使用 `conda activate openwam`。
激活脚本还将模型缓存、CUDA/JIT 缓存放在项目 `.cache` 下，并按用户要求设置 WandB 默认为 online。
现有凭据已通过联网验证；如需离线运行，可显式设置 `WANDB_MODE=offline`。

## 固定版本与安装

| 组件 | 版本 |
|---|---|
| Python | 3.12.13 |
| PyTorch / torchvision | 2.7.1+cu126 / 0.22.1+cu126 |
| Accelerate / DeepSpeed | 1.12.0 / 0.18.5 |
| Transformers / Diffusers | 5.16.1 / 0.40.0 |
| NumPy / PyArrow / PyAV | 1.26.4 / 22.0.0 / 15.1.0 |

基础版本保持此前训练审计使用的主要依赖组合。完整传递依赖锁在
`envs/requirements-cu126.lock`，不是每次安装时重新选择最新版本。

```bash
# 需要可用的 uv；安装脚本可从任意工作目录运行。
bash scripts/setup_openwam_env.sh
```

脚本使用官方 PyPI 和 PyTorch CUDA 12.6 wheel 源，下载独立 Python，先安装构建依赖及
PyTorch，再以 `DS_BUILD_OPS=0`、`--no-build-isolation` 安装 DeepSpeed 和完整锁定依赖。
可通过 `OPENWAM_PYPI_INDEX` 指定镜像，但镜像必须包含锁定版本；本机镜像曾缺少
Transformers 5.16.1，因此实际完整安装使用官方 PyPI。

CUDA 运行库由 PyTorch 依赖提供，不使用系统 `/usr/bin/nvcc` 11.5。
本机可选 JIT 构建使用 `/usr/local/cuda-12.4`，通过 `OPENWAM_CUDA_HOME` 可覆盖。
CUDA Toolkit 12.4 与 PyTorch CUDA 12.6 有次版本差异，DeepSpeed 会提示该差异并接受该组合；
当前环境已通过 CPUAdam 实际编译与单卡训练验证。

重新生成锁定文件时，先激活已安装环境，再执行：

```bash
DS_BUILD_OPS=0 uv pip compile envs/train-cu126.in \
  -c envs/constraints-cu126.txt --python .venv/bin/python \
  --torch-backend cu126 --no-build-isolation \
  --index-url https://pypi.org/simple -o envs/requirements-cu126.lock
```

## Foundation 权重复用

默认目录为 `models/OpenWAM-Alpha-Pretrain-Foundation-Model`。
`config.yaml` 和 tokenizer 已复制到项目目录，权重文件为只读使用的符号链接：

```text
checkpoint_step_154000.safetensors
  -> /data0/jikangye/research/runs/openwam-moon-cake-1k-20260914/foundation/checkpoint_step_154000.safetensors
```

虽然缓存上层目录名称含 moon-cake，此文件是官方原始 Foundation 权重。
重新读取全文件计算的 SHA256 与此前取得的官方发布哈希一致：

```text
180a02653118b0f96da28a9cae9ec7b4c1c1e6cd0e4608b56c7b4884f8001d3d
```

核验记录在该目录的 `foundation_verification.json`。链接目标需要持续可读；
如移动权重，可设置 `OPENWAM_FOUNDATION_DIR=/absolute/path/to/foundation`，
或覆盖配置的 `training.finetune_ckpt_path`。
环境安装脚本只管理 Python 依赖，迁移项目时需另行准备该模型目录和数据集。

## 验证记录

新环境检查的日志和报告保存在 `outputs/openwam_env_validation/`：

- `environment.json`：独立解释器、包版本和包路径，确认无外部 Python 包路径。
- `pytest.log`：155 项数据、归一化、掩码、损失及微调配置回归通过。
- `resolved_8gpu.yaml`：实际解析的正式配置与模型、数据路径。
- `distributed_cpu_audit.json`：8 个 CPU rank 的数据分片和完整学习率调度通过；
  每 epoch 6417 steps，覆盖 821344 个窗口，尾部补齐 32 个窗口。
- `reader/reader_smoke.json`：24 个真实窗口的数据读取、三相机解码、动作还原检查通过，
  最大还原误差 8.94e-8。
- `smoke.log` 与 `run_smoke.sh`：GPU 2 的单卡真实训练通过，Foundation 初始化、bf16、
  batch=1、ZeRO-2 + CPU optimizer offload，2 步 loss 为 0.2048 / 0.1921，退出码为 0。
  关闭 checkpoint 保存，未启动正式长程训练。
- `setup_rerun.log`：安装脚本重复运行通过；116 个安装包依赖兼容。

CPU 分布式验证和单卡 smoke 不等于 8 卡 NCCL/ZeRO-2 正式训练验证；
完整 8 卡吞吐、显存和中断恢复仍需资源齐备后测量。

```text
OpenWAM/
├── .venv/                              # 独立 openwam 虚拟环境（忽略）
├── .cache/python/                      # 独立 Python（忽略）
├── .cache/torch_extensions/             # 当前 Python 的 JIT 扩展（忽略）
├── envs/
│   ├── constraints-cu126.txt           # 主要训练依赖约束
│   ├── train-cu126.in                  # 环境安装入口和构建依赖
│   └── requirements-cu126.lock         # 完整依赖版本锁定
├── scripts/
│   ├── setup_openwam_env.sh            # 可重复安装
│   ├── activate_openwam.sh             # 环境、缓存、Foundation 路径
│   └── train_nero_8gpu.sh              # 默认选择本项目环境
├── models/OpenWAM-Alpha-Pretrain-Foundation-Model/  # 官方权重入口（忽略）
└── outputs/openwam_env_validation/     # 验证产物（忽略）
```
