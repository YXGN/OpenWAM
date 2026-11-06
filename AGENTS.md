# 给接手 OpenWAM 的 agent

这是独立模型仓库：从当前克隆的 README 和原生环境开始，不要求父推理仓库或 workspace.sh。已有数据/profile 合同的新任务只修改本地 recipe 并生成脚本，不新增或复制 trainer；仅在确有新数据/动作合同或模型结构时实现适配并补测试。
先读 README.md、docs/training.md、docs/tasks.md。新 Nero 训练复用 scripts/prepare_training.py；模型算法和原生训练循环已经存在，不为新任务复制整套 trainer。

- 先 inspect 实际数据及代表性样本。新入口只支持明确的 nero_fk_v2 合同：LeRobot v2、state16/FK9×2、60 Hz、原生三相机。字段、坐标、单位、动作时间偏移和相机视角都需核对。缺字段时不能改个名字当作完成转换。
- 默认 relative XYZ + relative SO(3) Rot6D、absolute gripper；absolute 模式保留绝对位姿。状态始终绝对。保留原生 33 帧→32 动作窗口，不套用 pi05 的 H50/末尾补齐。
- 月饼 v3 历史 delta 仅 XYZ 相对，旋转仍绝对；不得拿它的 checkpoint、stats 或部署加回逻辑当作新版完整相对旋转。
- 资源按用户提供的 GPU 数量、显存和数据规模确定。默认 1 epoch、GAS1、每卡 batch1；从 training_plan 解释实际窗口、补齐样本、微步及优化器步。prepare 只生成，不启动训练。
- train.sh 先调用原生统计检查，后调用 torchrun。统计、数据、动作域使用同一 recipe；不要用另一任务的 normalization_stats.npy。新 run 不覆盖旧输出。
- 基础权重需带原生 config.yaml。权重 warm-start 和 full-state resume 不同，不把 finetune 叫 resume。迁移产物须保留 config.yaml、统计、tokenizer 和相关资产。
- 优化已融合本地推理缓存和官方训练改进。speed_preset 默认 null，fast 显式选择。GPU 内核、重计算、CPU offload 和预编码缓存都不能未经资源检查自动启用；CPU 单测不证明硬件吞吐或等价性。
- 不再使用 velabot/ 或 migration/ 里的历史服务启动方式。真实机器人服务从外层推理项目启动；保留原生引擎接口供该项目调用。新训练完成后需单独验收 checkpoint 与 runtime 合同。
- 各模型独立依赖，原 ~/project 源码及环境只读，不运行安装/训练/模型下载来完成文档或准备任务。数据、权重、机器配置和 run 输出不入库。

相关 CPU 测试放 tests/。修改底层推理或 CUDA Graph 行为时需另做固定输入比较和真实 GPU 验收；不得把未执行的实验写成通过。优化来源固定 SHA 在 docs/optimization_sources.json。
