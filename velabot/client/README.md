# OpenWAM VelaBot client patch

本 patch 以 VelaBot 基线 `0432753f25f88caf2fb29d0904739ac48cdfbf74`（短标识
`0432753f`）为应用目标，包含 OpenWAM client、protocol codec、runtime 接线、相关
测试和客户端文档。它不包含独立 HTTP front、训练代码、模型 patch，也不包含 RoboDojo、
Agent、Console、memory 或其他并行改动。

应用方式：

```bash
git checkout 0432753f
git apply --check /path/to/openwam-velabot-client.patch
git apply /path/to/openwam-velabot-client.patch
```

legacy source identity 保持官方 `OpenWAM-Official/OpenWAM` 与 c171 base 的四字段行为，
必须省略 `base_revision`；显式 `base_revision: null` 也会拒绝。integrated source
额外允许 `base_revision=059cbfbfc068355f3d60fe9b014ff6469d1e94a3`，repository 固定为
`YXGN/OpenWAM`，revision 要求完整小写 40 位 commit，并要求非空
`patch_sha256` 与 `files_sha256`；未知字段仍拒绝。服务端/训练侧需另行在 core 中应用
模型 patch，并登记 core 与 `velabot/client` 的实际文件 hash。

本 patch 的 codec 定向测试已通过 24 项；Ruff lint/format 通过。它证明客户端协议和
runtime 接线边界，不证明模型质量或真机效果。
