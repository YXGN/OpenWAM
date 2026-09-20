# Nero 新协议链路审计（2026-09-20）

实现范围：官方模型加载/预处理/JointInferenceEngine 之外的 HTTP 动作块边界；
客户端增加独立 openwam_nero codec 与必要的 PNG、运行时绑定。没有替换官方模型算法。
官方引擎仍联合生成动作与视频 latent；关闭视频解码不改变动作生成，不向客户端传视频。

## 已验证结果

| 检查 | 结果 |
|---|---|
| 客户端协议、观测、composition、控制配置及旧协议回归 | 91 passed |
| OpenWAM 服务与开环评估回归 | 27 passed |
| 扩展动作队列/插值/重映射回归 | 131 passed，1个既有失败 |
| 控制生命周期回归（CasADi 3.7.2，纯 fake hardware） | 85 passed，1个既有失败 |
| 修改文件 Ruff 与 diff 空白检查 | 通过 |
| 客户端真实源码架构检查 | 474文件通过 |
| 相对 EEF，step8025，三任务各3窗口 | 9/9，HTTP与官方直调最大差0 |
| 绝对 EEF，step4000，三任务各3窗口 | 9/9，HTTP与官方直调最大差0 |

真实模型使用本机A100的0/2卡，未占用正式训练的1/3/5/6卡。GPU验证为 loopback HTTP，
包含真实codec/PNG编码/协商/动作队列/ActionDecoder，不包含真实机器人执行。
三任务 episode 51、242、347，各取起点、中点和最后有效起点；检查图像像素、官方
拼图、实测state、prompt、一次模型生成、32个动作节点和解码的pose/夹爪。
相对权重HTTP平均0.7986秒（0.7855–0.8495）；绝对平均0.8241秒（0.8015–0.9121）。
这是单机回环请求时间，实际跨机传输与在线观测编码还需计入。

FK独立审计覆盖三任务300帧、两个Nero profiles，对比训练URDF：最大位置差
2.956e-7m，最大旋转差3.285e-5度；独立URDF计算与数据集已存FK最大差2.98e-8。
`base_link`是profile标签，已审计数值实际对应训练body，不再额外变换一次。
AgileX上下行夹爪均为米制opening_width，未做旧MoonCake的[-1,1]映射。

## 本次消除的语义风险

- 同为20维不能证明坐标一致：新协议验证已审计的MDH、mount、tool、关节顺序、
  无父链和BASE任务坐标；不同几何明确拒绝。
- 相对权重只在官方引擎还原一次；线上的state/actions一律物理绝对EEF20。
- 关闭客户端epoch remap和跨块bridge，策略与执行固定60 Hz，首节点原值进入解码。
- 三路相机按角色显式映射；原分辨率PNG，官方预处理负责训练布局与缩放。
- 会话、请求ID、生成anchor、完整schema和checkpoint/config/stats SHA256严格校验。
- 停止后才能换任务/模型；沿用generation作废在途旧响应，不复用旧队列。

## 边界与已知问题

两个扩展测试失败均在未修改客户端HEAD `3bb58fe`上复现，本次未掩盖或修改：
`test_batched_rot6d_matches_the_scalar_twin[random-rows0]`：标量/批量浮点差
3.68e-16，旧测试要求差严格为0；`test_policy_starvation_zeroes_and_accepts_a_recovered_chunk`：
移动底盘恢复预期0.2、得到0。新Nero无底盘且禁用旧remap。
CasADi仅安装到临时测试目录；未改正式环境。没有运行全仓测试或硬件SDK测试。

绝对模型原始夹爪预测在576个值中93个超出[0,0.1]，最大超界0.000781m。
这是checkpoint的原始输出；协议保留，与官方直调一致。机器人运动层的现有范围、
速度约束仍然适用，因此模型原值与实际执行目标需区分。相对模型沿用官方已有投影。

同步receding-horizon每块32/60≈0.533秒，生成约0.8秒，耗尽后存在等待；接口一致
不等于连续实时或任务成功保证。本次没有启动真实机器人，也没有实测现场相机身份、
同步、IK跟踪与闭环成功率。

## 证据与复现

- `outputs/nero_protocol_validation/relative/result.json` 和 `predictions.npz`
- `outputs/nero_protocol_validation/absolute4000/result.json` 和 `predictions.npz`
- `outputs/nero_protocol_validation/fk_audit.json` 和 `fk_audit.py`
- 两类扩展测试日志位于同一目录，保留失败信息。
- [协议、启动参数及复现命令](nero-client-protocol.md)

相对权重SHA256：`9f363994f4bf85d09e0dcc6e9ce9b7f9980384f87d4e1d50115ab595022af6e0`。
绝对权重SHA256：`354e042335d03e2c520d43fbd4baea644df5e4582214e910d065d1aa5dcb76b9`。
完整配置/统计身份记录在各result.json。GPU验证采用待发布客户端源码，随后几何和频率
约束通过CPU专项验证；发布后验证文件哈希相同并复测导入，不重复加载大权重。
