# Nero VAE 预编码与训练接入设计

状态：设计稿。2026-09-18 已完成真实16窗口的VAE编码/序列化验证，尚未实现全量缓存训练路径。
本文不启动预计算、不修改训练配方、不恢复正式训练。

后续耗时定位：GPU0已复现DeepSpeed参数统计hook约11.58秒/步的CPU开销，
详见 `nero-step-time-root-cause-20260918.md`。应优先处理该开销，再评估缓存后的正式步时。

## 目标与适用范围

在现有官方 Wan2.2-TI2V-5B + dual_system 微调链路中，将冻结VAE的确定性编码移到训练前。
视频DiT、动作DiT和proprio encoder继续训练；视频预测loss保持开启。
第一版仅支持当前Nero固定三相机、关闭像素增强、33原始帧/stride4、BF16配方。
在线输入路径继续可用，其他reader与其他Wan variant默认不启用缓存。

数据量821344窗口；每窗口latent `[48,3,24,20]`，BF16有效载荷138240字节。
全量有效载荷105.745GiB。预算建议至少130GiB空闲，包含临时分片、索引和审计输出；
该预算只覆盖单份缓存，不含训练checkpoint或多版本缓存。

## 数据流

预计算：原MP4 → 原reader三相机拼图/采帧 → 原像素归一化 → 冻结VAE → BF16分片。
训练：窗口索引 → 缓存latent + 原parquet的action/proprio/mask/prompt →
首帧条件与文本context → 每步重新采样timestep/噪声 → 原联合前向/loss/反向/优化器。

缓存包含全部3个latent时间帧。第0帧用于干净条件，后两帧继续参与视频扩散训练。
不得缓存加噪结果、训练target、可训练proprio token、DiT特征或loss。

## 存储结构与契约

建议放在 `outputs/nero_multitask_data/latent_cache/<contract_hash>/`：

```text
manifest.json                  # schema、输入/权重/预处理契约、完整性状态
source_files.json              # 源文件路径、大小、时间、内容hash
index.npy                      # dataset_idx → episode_id/raw_start/shard/row
shards/
  shard_000000.safetensors      # latents[N,48,3,24,20]，约4096窗口/540MiB
  shard_000001.safetensors
  ...                          # 约201片；尾片可以不足4096
text_contexts.safetensors      # 可选：三条prompt的context和seq_lens
audit.json                     # 完整性、数值与训练等价性验证结果
```

分片按原dataset顺序划分，不能跨episode改变窗口语义。manifest记录：

- dataset长度、排序/索引版本、episode长度及每个窗口的原始起点映射；
- 原始视频/metadata/parquet指纹；现有dataset_fingerprint只对视频使用size/mtime，
  因此生成时补充视频内容SHA256，训练启动快速校验size/mtime，独立verify可做全量hash复核；
- VAE实际state_dict内容hash（不是整个训练checkpoint hash），VAE结构、mean/std、dtype；
- 相机head/left/right顺序，384×320画布，resize插值与[-1,1]转换实现版本；
- 窗口33、视频stride4、窗口stride1、独立窗口编码、latent shape；
- torch/cuDNN/解码器版本，分片文件hash、行数，以及缓存schema版本；
- 动作/状态仍走原reader，训练配置另记录动作契约及normalization_stats hash。

不要对latent应用动作统计归一化，也不要重复执行VAE输出mean/std归一化。
主契约变化使用新目录；不能原地覆盖已发布的可训练缓存。

## 预计算器

新增 `scripts/data/precompute_nero_latents.py`，接受配置、checkpoint、output、batch_size和worker数。
读取checkpoint中的VAE权重，eval/no_grad/BF16，仅加载VAE；不加载DiT、不创建优化器。

- 单卡默认batch16，沿用已经实测的形状；多卡一进程一卡，各自领取互不重复的分片，无DDP梯度通信。
- 第一版复用真实reader和像素预处理，保持窗口/画布一致；若以后优化重复视频解码，需要独立像素等价验证。
- 每个窗口重置VAE因果状态；禁止整episode编码后切latent，也禁止先分别编码三相机再拼latent。
- 尾batch记录实际batch大小；跨batch大小可能有BF16舍入差异，列入数值验证。
- 每分片先写独占临时文件，检查shape/dtype/finite/行数与hash后原子发布；成功分片支持续算。
- 不同进程不得写同一分片；每次续算校验已有文件和完整契约，不只看文件是否存在。
- 任一失败保持manifest为incomplete；仅全部821344窗口覆盖、无重复/缺失且审计通过后设为complete。
- 正式全量前先做约1万窗口的端到端试跑，覆盖三任务/长短episode/尾batch，实测解码、编码、写盘和峰值显存。

## Reader和模型接口

新增 `NeroMultitaskCachedDataset`，注册为 `nero_multitask_cached`。
采用原reader的同一窗口定位、parquet切片、动作/状态归一化和mask构造。
建议在共享reader中提取一个视觉读取hook，默认维持当前视频路径，缓存reader仅替换该hook；
避免复制整套action/proprio读取逻辑。该重构需验证其他reader的默认行为不变。

缓存sample额外提供 `video_latents`、`video_geometry`、`has_first_frame_condition`、缓存契约ID。
保留prompt/action/proprio/各mask和与原dataset完全相同的长度、index顺序；不返回虚构PIL占位图。
按worker懒加载分片，使用有界mmap句柄缓存，按row取tensor；不在每次getitem加载整片到内存。
合batch后再传GPU，避免原MP4解码、拼图和像素H2D；在线/缓存样本混用第一版明确拒绝。

修改 `BaseWAMArchitecture.prepare_inputs`，区分原始视频和预编码输入：

- 缓存路径跳过要求PIL视频的FirstFrameConditioningTransform，继续共用action/proprio/mask汇总。
- 修改 WanBackbone.preprocess_input_for_train，接受可选latent及显式几何元数据，严格验证形状/契约。
- 当前TI2V variant增加显式首帧条件flag；从latent第0帧clone出first_frame_latents，
  保持fuse_vae_embedding_in_latents、首帧干净替换及视频loss排除逻辑与在线一致。
- 不用 `ref_images=None` 暗示无首帧条件；这是无PIL缓存reader最容易引入的语义错误。
- 不对mmap原始tensor进行原地加噪/修改；组成batch后在训练设备构造噪声输入。
- 不改compute_loss中的timestep采样、随机噪声、视频/动作权重、联合注意力和可训练proprio context。

第一版保留冻结模块的注册、state_dict键名及现有设备生命周期，优先确保4000步完整状态可恢复。
绕过前向不会自动省掉常驻权重显存；将VAE/T5移到CPU或延迟加载是后续单独优化，
需要审核DeepSpeed初始化、完整状态恢复和自包含部署checkpoint，不能直接删除冻结模块。

## 文本缓存（推荐独立开关）

三条prompt的冻结T5 context约12MiB，保存context、seq_lens及prompt/tokenizer/T5权重指纹。
训练仍按原方式构造padding mask并追加可训练proprio token。文本缓存不包含该token。
可先做文本缓存，再做视频缓存；两个开关独立，便于在线/缓存A/B归因。
未知prompt或模型指纹不匹配时明确失败，不静默使用错误context。

## 配方和恢复

新增6卡/8卡cached YAML，各自继承当前官方兼容配方，仅替换reader与缓存开关/路径。
保留batch16、梯度累积1、LR、5epochs、采样/随机种子、保存间隔和ZeRO设置。
完整恢复4000步时，保留原6卡world_size、优化器分组、scheduler状态、数据索引顺序与skip逻辑。
记录缓存契约和接入配置变更，保留原config快照，不能把warm-start误称完整resume。
若切换8卡，不能默认认定与原六卡的完整状态/学习率轨迹等价；另行设计恢复。
部署继续从原始相机数据运行，不依赖训练缓存目录。

建议代码组织（均为待实现项）：

```text
scripts/data/precompute_nero_latents.py
scripts/data/verify_nero_latents.py
openwam/dataloader/nero_multitask_cached.py
openwam/dataloader/utils/latent_cache.py
configs/dataloader/nero_multitask_cached.yaml
configs/train_nero_6gpu_cached.yaml
configs/train_nero_8gpu_cached.yaml
tests/dataloader/test_nero_latent_cache.py
tests/model/test_wan_cached_inputs.py
```

另修改registry、共享reader视觉hook、architecture.prepare_inputs、Wan训练预处理和TI2V条件接口。

## 验收与效果边界

1. 文件层：全量shape/dtype/finite/索引/hash覆盖检查；故障中断后只重做未完成分片。
2. 数据层：三任务、各episode首尾/随机窗口、跨分片边界抽样对照。
   action/proprio/prompt/masks/index必须与在线一致，不能只检验latent。
3. 编码层：BF16落盘往返必须位级一致；相同batch在线/缓存编码对照。
   跨batch大小的差异单独报告max/mean/相对L2，不能以宽松容差掩盖时间或相机错位。
4. 模型层：固定同一batch、权重、RNG状态、timestep和噪声，比较首帧条件、context/mask、
   video/action/total loss与可训练参数梯度。使用同一数值输入时期待一致；其他情况以重复在线基线衡量数值误差。
5. 多卡层：原6卡配置在线/缓存各预热后至少100步，在独占GPU条件下记录各rank阶段计时、
   p50/p95 step、数据等待、吞吐、显存；测试一个完整保存与恢复周期，验证部署加载兼容。
6. 全量缓存训练前先做以上检查；验收完成再恢复正式训练。当前正式训练保持暂停。

在冻结确定性编码、无随机像素增强且上述输入一致时，缓存保持同一个训练目标，
理论上不因缓存本身改变训练效果。但浮点算法、batch形状和分布式执行可能引入舍入差异，
不能承诺长训练权重逐bit一致，也不能以数值测试替代任务成功率评估。

## 收益估计

本地实测VAE约0.618秒、像素处理约0.079秒、T5约0.615秒/每卡batch16。
仅视频缓存理论上省约0.70秒减去缓存I/O；视频+文本约1.31秒减去缓存I/O。
相对正式约15秒/步，若其余阶段不变，约为5%或9%的时间减少，不能承诺成倍加速。
单卡本地流水线3.32秒中这部分占比更高，但该比例不能直接套到六卡正式训练。
如果六卡存在明显视频解码等待，缓存可能额外获益；现有单卡取数等待很小，尚无六卡证据。

全量纯编码约8.8GPU小时，6卡理想下限1.47小时；此前2–3小时为端到端规划估计，未实测。
训练5遍数据时，缓存把重复5遍的VAE计算变成一次预计算和多次读取；
评估总收益时应扣掉预计算开销，不能只看训练阶段缩短。
