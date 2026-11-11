# 第 08 章：教学代码与 EngineAI 原生 SDK 的对应关系

> 上一章：[第 07 章：T800 抗扰动站立训练](07_T800抗扰动站立训练.md)  
> 本阶段终点：理解训练端与部署端的契约，不进行真机部署。

## 1. 本章目标

1. 把前七课的概念准确映射到 `engineai_robotics_native_sdk`；
2. 理解 SDK 中已有的是策略**部署**，不是 PPO 训练；
3. 读懂 25 关节全身顺序、22 active joints、72 维单步 observation 和 15 步历史；
4. 知道一个策略在部署前必须固定哪些输入输出契约；
5. 明确本教程 12-action 站立策略为什么不能替换 SDK walking MNN。

## 2. 关键路径

以下路径均相对于仓库根目录：

| 内容 | SDK 文件 |
|---|---|
| T800 limb/joint 顺序 | `engineai_robotics_native_sdk/assets/config/t800/model/default.yaml` |
| MuJoCo actuator 和 ctrlrange | `engineai_robotics_native_sdk/assets/resource/robot/t800/xml/serial_actuators.xml` |
| MuJoCo sensor 顺序 | `engineai_robotics_native_sdk/assets/resource/robot/t800/xml/serial_sensors.xml` |
| 仿真 PD 力矩回调 | `engineai_robotics_native_sdk/simulation/mujoco/main.cc` |
| PD 站立配置 | `engineai_robotics_native_sdk/assets/config/t800/pd_stand/default.yaml` |
| PD 站立 runner | `engineai_robotics_native_sdk/src/runner/pd_stand/src/pd_stand_runner.cc` |
| 被动/阻尼安全状态 | `engineai_robotics_native_sdk/src/runner/passive/src/passive_runner.cc` |
| walking policy 配置 | `engineai_robotics_native_sdk/assets/config/t800/rl_walking_example/default.yaml` |
| walking 推理 runner | `engineai_robotics_native_sdk/src/runner/rl_walking_example/src/rl_walking_example_runner.cc` |

## 3. 教学层与 SDK 的逐项对应

| 教程概念 | SDK 实现 | 含义 |
|---|---|---|
| 关节名字/顺序 | `model/default.yaml` | 全身 command/state vector 的契约 |
| `data.ctrl` 力矩输入 | `serial_actuators.xml` | `<motor>` 的 control 是力矩，不是目标角 |
| PD + 前馈 | `main.cc::TorqueController` | `kp*(qdes-q)+kd*(qddes-qd)+tau_ff` |
| 五次轨迹进入站姿 | `PdStandRunner::Run()` | 平滑从当前姿态到站姿 |
| 站姿、Kp/Kd | `pd_stand/default.yaml` | 25 关节目标和增益 |
| 零刚度/纯阻尼 | `PassiveRunner::Run()` | 失效时的安全输出思想 |
| observation 拼接 | `CalculateObservation()` | 输入项顺序、缩放和 clipping |
| action 到目标角 | `CalculateMotorCommand()` | `default_q + action*action_scale` |
| PD command 下发 | `SendMotorCommand()` | q/qd/Kp/Kd/feedforward 命令 |

## 4. SDK 的 PD 不是完整 PID

`simulation/mujoco/main.cc` 的核心公式是：

```cpp
tau_cmd = sim_command.kp.cwiseProduct(sim_command.q - q)
        + sim_command.kd.cwiseProduct(sim_command.qd - qd)
        + sim_command.tau_ff;
```

等价于：

```text
tau = kp * (q_des - q)
    + kd * (qd_des - qd)
    + tau_ff
```

没有积分状态，也没有 `ki`。因此应称为 **PD + feed-forward**。第 03 章的积分项只用于理解稳态误差和 anti-windup，不应误称 SDK 为 PID 控制。

## 5. SDK 的 25 关节全身顺序

顺序必须固定：

```text
0..5   左腿  J00_HIP_PITCH_L ... J05_ANKLE_ROLL_L
6..11  右腿  J06_HIP_PITCH_R ... J11_ANKLE_ROLL_R
12     腰部  J12_TORSO_YAW
13..17 左臂  J13_SHOULDER_PITCH_L ... J17_ELBOW_YAW_L
18..22 右臂  J18_SHOULDER_PITCH_R ... J22_ELBOW_YAW_R
23..24 头部  J23_HEAD_PITCH, J24_HEAD_YAW
```

策略 action 顺序、PD 增益顺序、默认姿态顺序任一错位，都可能让错误关节收到命令。不要只靠向量长度判断兼容。

## 6. Walking policy 的 22 active joints

`rl_walking_example/default.yaml` 中 active joints 包含：

- 双腿 12 个；
- 双臂 10 个；
- 不包含腰 J12；
- 不包含头 J23/J24。

因此 action 数量是 22。SDK 在 `Enter()` 中把每个 active joint 名字转换为全身 25 关节索引：

```cpp
for (const std::string &name : param_->active_joint_names) {
  active_joint_idx_(i++) = model_param_->joint_id_in_total_limb.at(name);
}
```

这种按名字构建 mapping 的方式比假定某个连续切片安全。

## 7. 72 维单步 observation

SDK `CalculateObservation()` 的单步顺序为：

```text
22 维 (q_real - default_q)[active_joint_idx]
22 维 qd_real[active_joint_idx]
22 维 previous_action
 3 维 body-frame angular velocity
 3 维 projected gravity
------------------------------------
72 维
```

公式：

```text
22 + 22 + 22 + 3 + 3 = 72
```

随后每个元素按配置缩放并 clip。**顺序、缩放和 clip 都属于模型的一部分**，不能只保存神经网络权重。

## 8. 15 步历史和速度指令

配置指定：

```yaml
num_observations: 72
num_include_obs_steps: 15
control_dt: 0.01
```

历史 observation 为 `72 × 15 = 1080` 维。runner 再追加 3 维速度 command：

```text
[obs(t-14), ..., obs(t-1), obs(t), command(vx, vy, yaw_rate)]
```

最终网络输入是 1083 维。代码会先转置 history matrix，再按时间步 flatten；如果 Python 训练端 flatten 顺序不同，即使 shape 相同，输出也会错误。

## 9. Action mapping

SDK 的推理输出先 clip，再映射：

```cpp
q_des_ = default_joint_q_;
q_des_(active_joint_idx_) +=
    mlp_net_action_.cwiseProduct(action_scale_);
```

即：

```text
q_des = q_default + clipped_action * action_scale
```

发送到底层的目标速度和前馈力矩均为零：

```cpp
qd_des_ = Zero(...);
tau_ff_des_ = Zero(...);
SetCommand(q_des_, qd_des_, joint_kp_des_, joint_kd_des_, tau_ff_des_);
```

所以 RL policy 不是直接替代关节闭环，而是在 PD 之上生成目标位置。

## 10. 为什么第 07 章模型不能直接替换 SDK walking MNN

| 契约 | 第 07 章 | SDK walking |
|---|---:|---:|
| 任务 | 抗扰动站立 | 速度指令行走 |
| action | 12 个腿关节 | 22 个腿+臂关节 |
| 单步 observation | 47 维 | 72 维 |
| observation history | 无 | 15 步 |
| command | 无 | vx、vy、yaw rate |
| 网络格式 | SB3/PyTorch `.zip` | MNN `.mnn` |
| reward | 教学站立 reward | 训练仓库未包含 |

维数、顺序、归一化、任务和文件格式全部不兼容。简单改扩展名或只导出网络都不能解决这些差异。

## 11. 只读兼容性检查程序

下面程序不会运行 SDK，也不会控制机器人；它只读取 SDK YAML 与 MJCF，检查最关键的关节契约。复制为临时文件 `/tmp/compare_sdk_contract.py`，从仓库根目录执行。

```python
from pathlib import Path

import mujoco
import yaml


ROOT = Path.cwd()
SDK = ROOT / "engineai_robotics_native_sdk"
MODEL_XML = ROOT / "models/t800/t800.xml"
MODEL_YAML = SDK / "assets/config/t800/model/default.yaml"
PD_YAML = SDK / "assets/config/t800/pd_stand/default.yaml"
RL_YAML = SDK / "assets/config/t800/rl_walking_example/default.yaml"

EXPECTED = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R", "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
    "J23_HEAD_PITCH", "J24_HEAD_YAW",
]


def flatten(groups):
    return [value for group in groups for value in group]


def main():
    for path in (MODEL_XML, MODEL_YAML, PD_YAML, RL_YAML):
        if not path.exists():
            raise FileNotFoundError(path)

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    actuator_joints = []
    for aid in range(model.nu):
        actuator_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid
        )
        transmission_jid = int(model.actuator_trnid[aid, 0])
        joint_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_JOINT, transmission_jid
        )
        actuator_joints.append(joint_name)
        assert actuator_name == f"motor_{joint_name}"

    model_cfg = yaml.safe_load(MODEL_YAML.read_text())
    sdk_joint_order = flatten([limb["joints"] for limb in model_cfg["limbs"]])

    pd_cfg = yaml.safe_load(PD_YAML.read_text())
    pd_q = flatten(pd_cfg["desired_joint_position"])
    pd_kp = flatten(pd_cfg["stiffness"])
    pd_kd = flatten(pd_cfg["damping"])

    rl_cfg = yaml.safe_load(RL_YAML.read_text())
    active = rl_cfg["active_joint_names"]

    assert model.nu == 25
    assert actuator_joints == EXPECTED
    assert sdk_joint_order == EXPECTED
    assert len(pd_q) == len(pd_kp) == len(pd_kd) == 25
    assert len(active) == 22
    assert rl_cfg["num_observations"] == 72
    assert rl_cfg["num_include_obs_steps"] == 15
    assert rl_cfg["control_dt"] == 0.01
    assert all(name in EXPECTED for name in active)

    print("model actuator order: OK (25)")
    print("SDK full joint order: OK (25)")
    print("PD vectors: OK (25)")
    print("RL active joints: OK (22)")
    print("RL input: 72 * 15 + 3 =", 72 * 15 + 3)
    print("All read-only contract checks passed.")


if __name__ == "__main__":
    main()
```

本章的只读契约检查脚本需要可选的 YAML 解析依赖 PyYAML；它不属于课程基础环境。安装后运行：

```bash
python -m pip install PyYAML
python /tmp/compare_sdk_contract.py
```

预期输出：

```text
model actuator order: OK (25)
SDK full joint order: OK (25)
PD vectors: OK (25)
RL active joints: OK (22)
RL input: 72 * 15 + 3 = 1083
All read-only contract checks passed.
```

## 12. 从训练到部署必须保存的 metadata

至少记录：

```text
robot/model version
full joint order
active joint order
observation names and order
observation scales and clipping
history length and flatten order
command order and scale
action order, scale and clipping
default joint pose
Kp/Kd
torque/target limits
physics dt and policy dt
network input/output names, shapes and dtype
```

缺少这些信息的权重文件不是可部署策略。

## 13. 后续路线（不在本轮执行）

### 13.1 SDK-compatible 训练 profile

先把环境改成：

- 22 actions；
- 72 维单步 observation；
- 15 步 history；
- 3 维速度 command；
- 0.01 s policy dt；
- 完全一致的 scale、clip、joint order 和 flatten order。

这只是兼容输入输出契约，还不等于学会行走。

### 13.2 速度指令行走

需要新增 gait/velocity reward、足部滑动、摆动脚、能耗、指令采样、地形与 reset 设计。不要只把站立 reward 加一个前进速度项就宣称完成行走。

### 13.3 模型转换

可能路径：

```text
PyTorch/SB3 policy
→ 导出 ONNX
→ 验证 ONNX 与 PyTorch 同输入输出
→ 转换 MNN
→ 验证 MNN 与 ONNX 同输入输出
```

SDK runner 加载 MNN，不能直接加载 SB3 `.zip` 或 ONNX。每一步都必须用固定 observation 测量最大/平均输出误差。

### 13.4 Sim-to-real 和真机

至少还需：

- 延迟、噪声、摩擦、质量和电机参数随机化；
- 状态估计与坐标系验证；
- action/target/torque 多层限幅；
- 推理频率和最坏延迟测试；
- policy watchdog；
- 实体急停和 passive fallback；
- 吊架、低功率、空载到负载的分级测试；
- 由熟悉 T800 硬件与现场安全的人审核。

本教程没有完成这些工作，因此不提供真机运行命令。

## 14. 常见误区

1. **有 `.mnn` 就有训练代码**：错误；当前 SDK 展示的是推理部署。
2. **shape 一样就兼容**：错误；顺序、缩放、单位和历史 flatten 也必须一致。
3. **把 SB3 `.zip` 改名为 `.mnn`**：文件格式完全不同。
4. **仿真站稳就能上真机**：仿真没有覆盖真实延迟、误差、故障和安全流程。
5. **RL 替代底层 PD**：SDK walking policy 仍输出 PD 目标角。
6. **训练 reward 高就安全**：reward 只反映定义过的目标。

## 15. 小练习

1. 在检查程序中打印 22 active joints 对应的全身索引。
2. 从 YAML 读取 `action_scale` 并验证其长度等于 22。
3. 手算 15 步 history 的时间覆盖范围：`15 × 0.01 = 0.15 s`。
4. 对比第 07 章 47 维 observation 和 SDK 72 维 observation，说明哪些项不同。
5. 设计一份 policy metadata JSON 字段表，但暂不编写部署程序。

## 16. 验收清单

- [ ] 能说出 SDK PD + 前馈公式。
- [ ] 能列出 25 关节分组和 22 active joints 的区别。
- [ ] 能推导 `22+22+22+3+3=72`。
- [ ] 能推导最终输入 `72×15+3=1083`。
- [ ] 理解 history flatten 顺序属于模型契约。
- [ ] 理解 SDK 示例是部署端而非训练端。
- [ ] 能解释第 07 章模型不能替换 walking MNN 的至少四个原因。
- [ ] 只读兼容检查通过。
- [ ] 明确当前阶段不能直接控制真机。

## 17. 阶段总结

至此，学习路径形成完整闭环：

```text
MuJoCo 模型
→ joint / actuator / sensor
→ torque
→ PD / PID / feed-forward
→ T800 PD 站立
→ Gymnasium
→ PPO 单关节
→ PPO 抗扰动站立
→ SDK 部署契约
```

下一阶段应先做 **SDK-compatible 22-action 仿真训练 profile** 和数值一致性测试，再考虑行走，最后才是真机安全验证。