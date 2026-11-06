# 历史任务与动作域

| 流程 | 数据来源 | 动作含义 | 当前用途 |
|---|---|---|---|
| 旧月饼 v3 | LeRobot v3，双臂 xyz+xyzw+夹爪，共 16 维 | `eef` 为绝对 EEF20；`delta` 仅 XYZ 相对，旋转和夹爪绝对 | 保留 moon_cake reader、配置及 velabot/training，解释旧权重 |
| Nero 多任务/月饼等 v2 FK | LeRobot v2，关节反馈16＋左右FK9 | XYZ 与旋转均相对窗口起点，夹爪绝对 | 新 `nero_fk_v2` relative 入口复用 |
| 原子 v2 FK | 相同 FK 合同，原始 atom 数据需先导出 | 绝对 EEF20、绝对夹爪 | 新 absolute 入口复用；保留历史缓存/配置 |

三个流程都使用 33 帧窗口和 32 步动作，但仅窗口大小相同不代表统计或动作解释兼容。新数据的任务文本来自其真实 metadata；不能从目录名猜 prompt。

原子导出工具 `scripts/data/export_nero_atom.py` 依赖明确提供的机器人采集仓库与 descriptor/URDF，生成 FK、数据和 provenance；它不是把任意 parquet 改名即可转换。导出、统计、训练、部署分别保存来源及合同。

旧机器路径已改为可替换路径，历史 GPU 数、batch、训练步数留作实验参考。不要照抄 `train_nero_*gpu.yaml` 的硬件设置作为新用户默认。原采集工作簿、研究笔记、运行状态和旧交接包已归档到整理前源码备份，训练所需实现仍在本仓库。
