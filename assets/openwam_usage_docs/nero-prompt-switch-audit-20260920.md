# Nero 客户端切换 prompt 联合验证（2026-09-20）

结果：**推理面板的任务修改确实进入服务端并参与真实模型推理**。同会话切换不要求
重启服务或重新握手，必须先停止策略、修改并确认成功，再启动。

## 实际测试

使用本机空闲0号A100、相对EEF step8025完整权重；固定 episode242 中间观测与seed42。
依次使用数据集原文：A为锤子任务，B为插头任务，再回到A。

链路：`InferencePanel.handle(inference.set_task)` → `PolicyController.configure`
→ 同一个 `ObservationBuilder.payload_getter()` → 新协议codec → localhost实际HTTP
→ 官方预处理 → `JointInferenceEngine` → architecture → T5条件 → 动作生成。

| 验证点 | 结果 |
|---|---|
| 服务端收到的完整prompt与architecture入参 | 严格等于A、B、A，换行保留 |
| T5实际编码调用 | A一次、B一次；回到A命中以文本为键的缓存 |
| A与B的embedding | SHA256不同，均为 `[1,512,4096]` |
| A与B动作 | 640个原始元素中596个不同；跨单位元素最大绝对差0.02701644 |
| 第一次A与回到A的动作 | 逐元素完全一致 |
| 策略运行中改任务 | 命令被拒绝，原prompt保持不变 |
| 停止后切B、旧A请求随后返回 | generation改变，旧响应被拒绝安装，队列为空 |
| 录制面板的 `record.set_task` | 只改录制任务标签，不改变模型prompt |
| 服务端CPU回归 | 8项通过，含同会话A/B/A、空格、换行与中文原文透传 |

动作差异用于确认prompt进入计算，不代表B在A场景下的动作正确或任务成功率提高。
测试调用真实控制台命令处理器，但没有自动点击浏览器；Soma输入来自数据集，运动
ownership/output回调为桩，停止/清队列调用真实PolicySession方法。未连接机器人，
没有验证运动完成后的反馈闭环。此次未修改生产推理逻辑或UI行为。

## 使用时选择正确的入口

1. 在推理面板停止策略，等待停止完成。
2. 在**推理面板 → 改任务描述**修改并保存，确认命令成功。
3. 启动策略，下一次请求使用新prompt。

录制面板也有“改任务描述”，但它用于数据集标签，不用于切换推理任务。
当前推理面板的任务按钮可能仍显示可点击；新Nero协议运行中提交修改会返回明确拒绝，
不要把填入文本或点击按钮本身当作配置已生效。

## 可重复命令

在OpenWAM环境执行；脚本会自行加载模型、创建临时HTTP服务，无须预先启动服务。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python scripts/audit_nero_prompt_switch.py \
  --client-root /path/to/velabot_agentic_os \
  --ckpt-dir models/nero_relative_8gpu_b64_20260919/step_8025 \
  --ckpt-name checkpoint_step_8025.safetensors \
  --dataset /path/to/nero_multitask_mooncake_hammer_charger_408_fk_lerobot_v2 \
  --output outputs/nero_prompt_switch_new
```

output必须是新目录。实测证据保存在
`outputs/nero_prompt_switch_validation/relative_8025/result.json`、`predictions.npz`、`audit.log`。
result包含真实权重/config/stats身份、全量prompt、embedding指纹和断言结果。
