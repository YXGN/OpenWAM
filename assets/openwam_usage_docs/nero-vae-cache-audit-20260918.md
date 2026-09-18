# Nero 冻结编码器缓存评估：2026-09-18

结论：当前配方可以预计算 VAE latent，也可以缓存三条任务文本的 T5 输出。
冻结参数只省去梯度和更新，现有训练仍逐 batch 执行这两个编码器。
但本地同形状测量中 VAE 仅约0.62秒，不能据此解释正式六卡约15秒的整步耗时。
六卡内部计时尚待资源可用；不能把未归属差额直接称为通信耗时。

## 已完成的真实样本验证

`scripts/audit_nero_vae_cache.py` 从4000步保存点仅加载 VAE，使用当前官方数据预处理和 batch_encode。
GPU0运行，限制本进程PyTorch显存分配器12GiB；实际allocated峰值9.30GiB。
没有更新权重、改训练配置或生成全量缓存。

- 16个窗口覆盖月饼、锤子、插头三个任务。
- 输入 `[16,3,9,384,320]`，输出 `[16,48,3,24,20]`，BF16。
- 保存为 safetensors 再加载，逐元素完全一致。
- 相同batch再次在线编码，最大绝对差0。
- 预热后4次编码均值0.61740秒/16窗口。
- 小样本热页缓存读取并传GPU约0.000453秒/16窗口；这不是全量随机读取性能保证。
- 全部821344窗口的latent有效载荷113542594560字节，约105.745GiB，另加索引、元数据和临时文件。
- 由本次吞吐估计编码计算约8.80 GPU小时；六卡理想并行下限约1.47小时。
  实际还包括视频解码、CPU预处理、写盘和并行I/O竞争，应通过分片试跑再确定完成时间。

数值证据：`outputs/nero_vae_cache_audit_20260918/summary.json`。
样本文件：同目录 `sample_latents.safetensors`，只含16个窗口。

## 必须保留的语义

1. **按训练窗口独立编码。** VAE 每次 encode 重置因果缓存；窗口第一帧单独编码，后续每4帧压缩。
   实测将13帧长片段编码后切出对应3个latent帧，与独立9帧窗口编码相比，
   平均绝对差0.208718、最大2.25。因此不能“每个episode编码一次，再随意切latent”。
2. **先拼三相机再编码。** 严格沿用 head上方、left左下、right右下的384×320画布和相同插值。
   分别编码相机再拼latent不等价。输入仍按当前代码转换到[-1,1]。
3. **保存最终标准化latent。** 缓存 batch_encode 返回值（已应用VAE自身mean/std），
   读取后不能再重复做VAE标准化；动作/状态归一化继续走原reader，二者是不同层面的处理。
4. **保留首帧条件。** 当前TI2V首帧条件来自 `input_latents[:,:,0:1]`；缓存reader虽不返回PIL图像，
   也必须显式传递“存在参考首帧”的信息，保持 `fuse_vae_embedding_in_latents=True` 和首帧loss排除。
   不能因ref_images缺失而退化为无首帧条件训练。
5. **仅缓存固定编码。** 不能缓存扩散噪声、timestep、训练target随机部分或可训练proprio encoder输出。
   三条文本可保存冻结T5的context和seq_lens，之后照常动态追加可训练proprio token及mask。
6. 当前关闭颜色增强且相机/窗口固定，因此缓存成立。新增随机颜色/裁剪等像素增强时，
   固定latent缓存会改变训练分布；必须拒绝不兼容配置或回到在线编码。

小样本证明序列化无损和同batch重复编码一致，不代表任意batch形状、GPU型号或库版本都位级一致。
全量生成应固定BF16、VAE权重及预处理版本，并在不同任务/episode/窗口边界做抽样在线对照。

## 建议实现边界

- 预计算器：只加载冻结VAE，按episode/起点生成窗口，保存分片；断点续算、临时文件原子改名，
  最终manifest核验覆盖821344窗口，无缺失/重复。优先分片，避免821344个小文件。
- manifest：记录VAE权重hash、数据集身份、episode/窗口起点映射、相机顺序、画布尺寸、
  33原始帧/stride4/9输入帧、预处理实现版本、dtype、latent形状和生成软件版本。
- reader：以episode_id + raw_start_frame取latent，继续按同一窗口读取action/proprio/masks/prompt。
  使用缓存时跳过MP4解码和PIL画布生成；缓存不存在或契约不匹配应明确报错，避免静默混用。
- 架构：增加可选的预编码输入路径，原在线路径保留；兼容当前官方finetune checkpoint。
  Wan训练预处理接受latent及明确的尺寸/首帧条件，绕过冻结编码器，后续联合扩散loss不变。
- 文本：仅三种prompt，BF16的3×512×4096 context约12MiB，另有seq_lens；
  key应包含prompt、tokenizer、T5权重和预处理版本。优先做此项，空间成本远低于视频缓存。
- 验证：固定噪声和timestep，对照在线/缓存的输入latent、首帧条件、mask、loss和可训练参数梯度，
  然后做六卡短跑与随机分片读取吞吐验证。不能只检查缓存文件能打开。

预计收益：VAE约0.62秒，T5约0.62秒，像素搬运约0.08秒，加上可能消除的数据解码等待；
正式六卡实际收益须实测。若瓶颈在DeepSpeed/跨rank等待，缓存不会自动消除剩余耗时。

## 复现

```bash
source scripts/activate_openwam.sh
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled python scripts/audit_nero_vae_cache.py \
  --checkpoint outputs/nero_multitask_6gpu/2026-09-17_18-22-36/checkpoint_step_4000.safetensors \
  --output outputs/nero_vae_cache_audit_20260918
```

运行前检查所选GPU资源；再次执行应更换output目录，保留原测量。
六卡计时脚本 `scripts/profile_nero_distributed.py` 配合
`outputs/nero_distributed_profile_20260918/run.sh`，隔离输出、禁用保存和WandB，12步后结束。
目前它仅通过配置解析/静态检查，未取得六卡实测数据；正式训练保持暂停。
