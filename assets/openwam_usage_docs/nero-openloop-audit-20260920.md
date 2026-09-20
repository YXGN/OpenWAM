# Nero 开环评估验收：2026-09-20

## 实现与复用

新增 `python -m openwam.evaluation.nero_openloop`，调用官方结构的 checkpoint loader、ObsPreprocessor、JointInferenceEngine 和 architecture.generate。我们已有的 Nero 合同负责相对/绝对 EEF 的差异；没有导入旧 MoonCake、VelaBot 或第三方机器人兼容入口，没有修改训练循环、推理去噪循环或动作执行策略。

官方源码复核包括 `engine.py`、`model_loader.py`、`obs_preprocess.py` 和 `architectures/base.py`；80D gather/scatter 与无效动作槽位的解析噪声路径在官方代码中已有。评估记录器透传这些属性，并用真实 GPU 重复推理检验输出一致性。

## 数据与语义验证

- 当前项目相关测试：**83 passed**，包括新增评估、动作归一化、相对 Nero reader、绝对 Nero reader。
- 隔离检查：从已提交 HEAD 导出代码，仅覆盖本次拟提交文件；新增评估与相对 reader 测试 **37 passed**。确认本功能不依赖工作区其余未提交训练配方/缓存改动。
- 真实数据：episode **51、242、347**，各自开头、中间、最后完整窗口，分别按相对和绝对合同验证，共 **18 个窗口**。
- 所有评估首帧拼图与训练 reader 首帧 **像素完全相同**。
- 当前状态经部署 normalizer 后，与训练 proprio 在 `atol=3e-7` 内一致。
- 训练 action 反归一化并按需还原相对位姿后，与评估 `t+1…t+32` 的 GT 一致；最大逐元素误差 **5.96046448e-8**（20D 混合单位数组的数值误差，不应作为单一物理指标）。
- 三路图像只请求当前索引 t；推理 conditions 不含未来图像、GT 动作或训练 VAE 视频缓存。
- 覆盖非交换旋转、episode 尾部、跨 task 边界、槽位/相机顺序/旧 delta/二值夹爪配置拒绝，以及记录器异常后恢复 normalizer。

原始验证证据：`outputs/nero_openloop_validation/reader_parity.json`。

## GPU 短测和三条完整 episode

使用本地空闲 GPU 0，未更改在 1、3、5、6 上运行的训练配置。环境 PyTorch 2.7.1+cu126、bf16、10 步同步去噪、seed42、CFG1；T5 embedding 缓存开启，compile/DiT 跳步缓存关闭。

权重为绝对 EEF **step2000**，不是完成训练的模型：

```text
outputs/nero_absolute_4gpu/2026-09-20_02-15-44/checkpoint_step_2000.safetensors
sha256: 47dfebe14a61eaf7da6bd33e39c3ed6e24172e61828abdd71d962788b910b587
```

短测对三个任务各一个窗口生成 32×20 输出；首窗口记录器开启/关闭的结果逐元素一致，**max_abs=0**。

随后完整评估 episode 51 / 242 / 347，stride32 加最后完整窗口，分别 75 / 60 / 41 个窗口，合计 **176 个窗口**。全部成功、输出有限，原始预测退化旋转数量为 0；再次通过真实推理 max_abs=0 的一致性验证。

| 资源/耗时 | 实测 |
|---|---:|
| 预热后 175 个窗口平均推理 | 0.7809 秒 |
| 预热后 P95 | 0.8008 秒 |
| 含首窗口平均 | 0.7868 秒 |
| 峰值 allocated 显存 | 23.3804 GiB |

计时包括模型当前图像编码、文本条件、联合去噪和动作解码；不包括模型加载、CPU 视频读取、报告写入。全量 dataset fingerprint 和 checkpoint SHA256 在加载前执行，因此启动耗时明显大于单窗口推理。显存数字为 PyTorch allocated，不等同 nvidia-smi 总占用。

## 初步物理误差

下面是训练集开环误差，按窗口×horizon 统计；位置为三维距离均值，旋转为 SO(3) 角误差均值，夹爪为宽度 MAE。

| Episode / 任务 | 左/右位置 mm | 左/右旋转 ° | 左/右夹爪 mm |
|---|---:|---:|---:|
| 51 / 月饼 | 5.61 / 6.40 | 1.39 / 1.58 | 2.45 / 1.40 |
| 242 / 锤子 | 10.62 / 9.15 | 3.52 / 2.26 | 0.92 / 3.25 |
| 347 / 插接 | 9.81 / 2.04 | 2.77 / 0.54 | 3.46 / 0.24 |

保持当前状态的基线说明，模型并非每个指标都更好。例如插接 episode 的右臂几乎静止，保持当前的位置误差仅约 0.0034 mm，模型约 2.04 mm；锤子左臂旋转也未优于保持当前基线。这些现象可以在逐维曲线中检查，不能只凭总体平均宣称模型效果优秀。

未裁剪的绝对夹爪输出约 33.92% 超出标定区间，越界幅度在所有夹爪样本上的均值 0.170 mm、P95 0.781 mm、最大 2.344 mm。评估如实记录，不悄悄裁剪来改善分数，不修改现有推理行为。

这次 GPU 验证针对本地绝对 checkpoint；相对 EEF 完成了单元、合同、真实 reader 和还原语义检查，尚未在目标机最终相对 checkpoint 上运行全 episode GPU 评估。

## 产物

- 三窗口短测：`outputs/nero_openloop_validation/20260920_absolute_step2000_smoke/`。
- 完整三 episode：`outputs/nero_openloop_validation/20260920_absolute_step2000_episodes/`。
- 完整报告：上述目录的 `report.html`；原始逐维结果为 `predictions.csv` 和 `predictions.npz`。
- 每次运行保存 checkpoint/config/stats/source 哈希、PyTorch/CUDA/cuDNN/GPU 信息、实际输入拼图哈希、采样锚点和推理参数。

HTML 以本地 SVG 渲染，不需要额外绘图库。已验证报告的 episode、窗口、绝对/原始动作视图生成 20 个维度图和 6 个 horizon 指标图，并检查 SVG 坐标有限。SVG 可导出；数据集任务文本作为纯文本展示并转义脚本边界。

使用方式与指标口径见 [nero-openloop-evaluation.md](nero-openloop-evaluation.md)。
