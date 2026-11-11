# 第 05 章：强化学习基础与 Gymnasium 环境

> 上一章：[第 04 章：T800 PD 站立](04_T800_PD站立.md)  
> 下一章：[第 06 章：PPO 单关节训练](06_PPO单关节训练.md)

## 1. 本章目标

完成本章后，你将能够：

1. 用控制工程语言解释 observation、action、reward 和 episode；
2. 理解策略为何不直接输出大力矩，而是输出 PD 目标角偏移；
3. 实现符合 Gymnasium 接口的单关节 `JointReachEnv`；
4. 正确区分 `terminated`（任务失败）和 `truncated`（时间到）；
5. 用 `check_env` 和随机策略检查环境。

本章只做**单关节目标跟踪**。先把 RL 接口学清楚，再进入 T800 全身训练。

## 2. 从 PID 闭环到强化学习闭环

传统控制器由人写控制律：

```text
状态 q, qd → PID/PD → 力矩 tau → 被控对象 → 新状态
```

强化学习把其中一部分换成可训练策略：

```text
observation → policy → action → 安全映射 + PD → tau → MuJoCo
```

对应关系：

| RL 名称 | 控制工程中的直观含义 |
|---|---|
| observation | 控制器可读取的反馈量 |
| action | 上层控制器输出的命令 |
| reward | 每一步的性能指标，越大越好 |
| episode | 一次有限时长的闭环实验 |
| policy | 从 observation 到 action 的映射 |
| value function | 对未来累计 reward 的预测 |
| training | 调整策略参数 |
| inference | 固定参数后，仅执行策略 |

策略不会绕开安全控制。本课程令 action 表示**目标角增量**：

```text
q_des = q_target + action * action_scale
```

然后仍由底层 PD 计算力矩。这样 action 有明确单位和范围，也自然对应 T800 SDK 的 `default_q + action * action_scale`。

## 3. 环境设计

### 3.1 Observation

本环境使用 5 维 observation：

```text
[q, qd, q_target, q_target - q, previous_action]
```

所有元素均为 `float32`，shape 永远是 `(5,)`。

### 3.2 Action

- shape：`(1,)`
- 范围：`[-1, 1]`
- 物理意义：围绕目标角的微小目标偏移，而非原始力矩

### 3.3 Reward

```text
reward = -2.0 * position_error²
         -0.02 * velocity²
         -0.01 * action_rate²
         -0.0005 * torque²
```

位置误差是主项，其他项只用于减少振荡、命令突变和过大力矩。

### 3.4 Episode 结束

- `terminated=True`：状态出现 NaN/Inf 或关节越过安全范围；
- `truncated=True`：达到固定时间上限；
- 到达目标本身不立即结束，策略需要持续稳定地保持目标。

## 4. 完整代码

将下面完整代码复制为临时文件 `/tmp/joint_reach_env.py`：

```python
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env


SINGLE_HINGE_XML = r"""
<mujoco model="joint_reach">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <light pos="0 0 2"/>
    <body name="link" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"
             range="-1.5 1.5" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5"
            size="0.04" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor" joint="hinge" gear="1"
           ctrllimited="true" ctrlrange="-5 5"/>
  </actuator>
</mujoco>
"""


class JointReachEnv(gym.Env):
    """单关节目标跟踪：策略给出 PD 目标角偏移。"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"unsupported render_mode: {render_mode}")

        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(SINGLE_HINGE_XML)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        # q, qd, target, error, previous_action
        self.observation_space = spaces.Box(
            low=np.array([-1.5, -50.0, -1.0, -2.5, -1.0], dtype=np.float32),
            high=np.array([1.5, 50.0, 1.0, 2.5, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.physics_dt = float(self.model.opt.timestep)  # 0.002 s
        self.frame_skip = 5                              # policy 100 Hz
        self.max_steps = 400                             # 4 seconds
        self.kp = 12.0
        self.kd = 1.2
        self.action_scale = 0.25                         # rad
        self.torque_limit = 5.0                          # Nm

        self.target = 0.0
        self.previous_action = 0.0
        self.step_count = 0
        self._renderer = None
        self._viewer = None

    def _get_obs(self) -> np.ndarray:
        q = float(self.data.qpos[0])
        qd = float(self.data.qvel[0])
        return np.array(
            [q, qd, self.target, self.target - q, self.previous_action],
            dtype=np.float32,
        )

    def _get_info(self, torque: float = 0.0) -> dict:
        q = float(self.data.qpos[0])
        return {
            "position_error": abs(self.target - q),
            "torque": float(torque),
            "simulation_time": float(self.data.time),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 必须使用 self.np_random，才能让 Gymnasium seed 生效。
        self.data.qpos[0] = self.np_random.uniform(-0.35, 0.35)
        self.data.qvel[0] = self.np_random.uniform(-0.05, 0.05)
        self.target = float(self.np_random.uniform(-0.8, 0.8))
        self.previous_action = 0.0
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        a = float(action[0])
        action_rate = a - self.previous_action

        # RL action -> 目标位置；底层 PD -> 力矩。
        q_des = self.target + self.action_scale * a
        last_torque = 0.0
        for _ in range(self.frame_skip):
            q = float(self.data.qpos[0])
            qd = float(self.data.qvel[0])
            torque = self.kp * (q_des - q) - self.kd * qd
            last_torque = float(np.clip(torque, -self.torque_limit, self.torque_limit))
            self.data.ctrl[0] = last_torque
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self.previous_action = a

        q = float(self.data.qpos[0])
        qd = float(self.data.qvel[0])
        error = self.target - q
        reward_terms = {
            "tracking": -2.0 * error**2,
            "velocity": -0.02 * qd**2,
            "action_rate": -0.01 * action_rate**2,
            "torque": -0.0005 * last_torque**2,
        }
        reward = float(sum(reward_terms.values()))

        finite = bool(
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(reward)
        )
        out_of_bounds = abs(q) > 1.49
        terminated = (not finite) or out_of_bounds
        truncated = self.step_count >= self.max_steps

        info = self._get_info(last_torque)
        info["reward_terms"] = reward_terms
        if terminated:
            info["termination_reason"] = (
                "non_finite" if not finite else "joint_limit"
            )

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()
        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def random_policy_demo(seed: int = 0) -> None:
    env = JointReachEnv()
    obs, info = env.reset(seed=seed)
    total_reward = 0.0

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print("final observation:", obs)
    print("total reward:", round(total_reward, 3))
    print("final position error:", round(info["position_error"], 4))
    print("terminated:", terminated, "truncated:", truncated)
    env.close()


if __name__ == "__main__":
    # skip_render_check=True 避免在无显示器机器上创建窗口。
    check_env(JointReachEnv(), skip_render_check=True)
    print("check_env: passed")

    # 同一 seed 的 reset 必须一致。
    env = JointReachEnv()
    obs_a, _ = env.reset(seed=123)
    obs_b, _ = env.reset(seed=123)
    np.testing.assert_allclose(obs_a, obs_b)
    env.close()
    print("seed check: passed")

    random_policy_demo(seed=0)
```

## 5. 运行

```bash
python /tmp/joint_reach_env.py
```

预期看到：

```text
check_env: passed
seed check: passed
final observation: [...]
total reward: ...
final position error: ...
terminated: False truncated: True
```

随机策略的最终误差每次可能不同。这里不追求随机策略表现好，只要求环境完整运行、数据有限、结束语义正确。

## 6. 逐段理解

### 为什么每个 action 做 5 个 `mj_step`？

物理步长为 0.002 s，即 500 Hz；策略每 5 个物理步才输出一次 action，即 100 Hz。底层 PD 在每个 2 ms 子步重新读取状态并计算力矩，而不是把一次计算的力矩保持 10 ms。

### 为什么 reset 调用 `super().reset(seed=seed)`？

它初始化 Gymnasium 的 `self.np_random`。不要在 reset 内调用全局 `np.random`，否则相同 seed 不能复现初始状态。

### 为什么 reward 多数为负？

这是 cost 的负数。最好的理论状态接近 0，误差、速度、突变和力矩越大，reward 越负。PPO 仍会最大化它。

## 7. 常见错误

1. **返回 4 个值**：旧 Gym API 的 `step` 返回 4 项；Gymnasium 必须返回 5 项。
2. **把超时当 terminated**：固定时间上限应是 `truncated=True`。
3. **shape 漂移**：即使 action 只有一个元素，也应保持 `(1,)`。
4. **返回 float64**：observation space 是 `float32`，实际 observation 也必须是 `float32`。
5. **action 直接乘 415 Nm**：这会把学习问题变成危险的冲击控制。
6. **只在策略频率算一次 PD**：底层 PD 应在每个物理子步刷新。

## 8. 安全提示

本环境中的 5 Nm 是教学模型限幅，不能外推到 T800。仿真 reward 高不代表真机安全，也不代表策略可直接部署。

## 9. 小练习

1. 把 `action_scale` 从 0.25 降到 0.05，观察随机策略的行为。
2. 暂时删除 `action_rate` 惩罚，比较 action 是否更跳跃。
3. 把 `frame_skip` 改为 10，并说明策略频率如何变化。
4. 加一个 `is_success = abs(error) < 0.03` 到 info，但不要立刻终止 episode。

## 10. 验收清单

- [ ] 能解释 observation 每一维的物理意义。
- [ ] `check_env` 通过。
- [ ] 相同 seed 的第一次 reset 完全一致。
- [ ] action 始终限制在 `[-1, 1]`。
- [ ] 每个 RL step 执行 5 个 MuJoCo step。
- [ ] reward 等于 `reward_terms` 各项之和。
- [ ] 能区分 terminated 和 truncated。

## 11. 与 SDK 的关系

本章已经采用 SDK RL runner 的核心分层思想：策略输出经过缩放的目标角偏移，底层继续使用 PD。下一章将真正训练策略；第 08 章再逐项对照 SDK 的 22 关节 walking policy。