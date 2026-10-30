# Nero 4090 部署 agent 交接

用户目标：在本地 4090 最小化部署当前 Nero Atom 6248 推理，与源服务器保持同一模型、输入输出合同及动作处理。默认执行必要安装、检查和模型服务启动，不需要重复确认已授权步骤。

- 先读 `migration/nero_4090/README.md` 和 `migration/nero_4090/manifest.json`。这是包含未提交修改的工作区快照；不要用重新 clone、git reset 或通用 OpenWAM WebSocket 入口替换它。
- 依次执行文件校验、环境安装、离线回放；全部通过后使用 `scripts/serve_openwam_atom_4090.sh`。仅部署模型服务，机器人运动由用户已有客户端负责。
- 模型、config、normalization_stats 和 tokenizer 必须成套；checkpoint 中的训练数据／缓存旧路径不是部署依赖，不要为了匹配本机路径修改训练 config。
- 只使用已更名的 `openwam` / `openwam/v1` PNG、EEF20、米制协议。内部 `nero_chunk_server` 文件名保留是正常的；不要恢复旧 JPEG／归一化夹爪兼容协议或 `openwam_nero` 别名。
- 4090 使用已有 `--text-encoder-cpu`，其余模型驻 GPU；保持 BF16 和 4 步，CUDA Graph 先关闭。T5 的 prompt cache 与 VAE／数据集训练缓存是不同事物，无需导出训练缓存。
- 不增加第二次夹爪偏移、归一化、坐标变换或客户端缩图。双手偏移固定减 2 mm，原二值阈值关闭。
- `replay` 为离线模型进程：在服务启动前运行，避免两份模型争抢显存。不要对仍在被机器人使用的服务器发送握手／reset，它们会改变会话。
- 数值超限或显存不足时，保存错误、报告和实际环境，定位原因；不得直接放宽阈值、改 prompt／分辨率／模型精度后报告“与服务器一致”。对新增修复记录变更并重新验证。
- 软件测试和历史 A100 样例不能证明 4090 的真实性能或真机成功率。记录实际 GPU、显存峰值、首次／缓存请求时间、回放误差以及最终启动命令、PID、日志位置。
- 结果维护在 `outputs/nero_4090_validation/LOCAL_STATUS.md`，区分完成项、失败项和待实机观察。向用户简短汇报连接地址和验证结果，不要求用户重新设计部署参数。
