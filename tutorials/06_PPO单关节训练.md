# 第 06 章：用 PPO 训练单关节策略

> 上一章：[第 05 章：强化学习与 Gymnasium](05_强化学习与Gymnasium.md)  
> 下一章：[第 07 章：T800 抗扰动站立训练](07_T800抗扰动站立训练.md)

## 1. 本章目标

1. 直观理解 PPO 的 rollout、actor、critic、advantage 和 clipping；
2. 用 Stable-Baselines3（SB3）训练第 05 章的 `JointReachEnv`；
3. 保存、重新加载并确定性评估模型；
4. 用**平均位置误差**而不是只看训练 reward 判断效果。

## 2. PPO 的最小直觉

- **actor/policy**：根据 observation 生成 action 分布；
- **critic/value function**：预测从当前状态开始未来能拿多少累计 reward；
- **rollout**：用当前策略收集一段 `(obs, action, reward, next_obs)`；
- **advantage**：某个 action 比 critic 的预期好多少；
- **PPO clipping**：限制每轮参数更新幅度，避免策略一次变化太大。

训练和推理必须分开理解：

```text
训练：环境交互 + 反向传播 + 更新网络参数
推理：固定网络参数，obs -> action
```

## 3. 安装

```bash
python -m pip install "mujoco>=3.1" "gymnasium>=0.29" \
  "stable-baselines3>=2.3" "torch>=2.1" numpy
```

## 4. 完整训练程序

本章代码是**独立可运行程序**，为方便手工复制，重复了必要的环境定义。复制为临时文件 `/tmp/train_joint_ppo.py`：

```python
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


XML = r"""
<mujoco model="joint_reach">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
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
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-1.5, -50.0, -1.0, -2.5, -1.0], np.float32),
            high=np.array([1.5, 50.0, 1.0, 2.5, 1.0], np.float32),
            dtype=np.float32,
        )
        self.frame_skip = 5
        self.max_steps = 400
        self.kp, self.kd = 12.0, 1.2
        self.action_scale = 0.25
        self.previous_action = 0.0
        self.target = 0.0
        self.step_count = 0

    def _obs(self):
        q, qd = float(self.data.qpos[0]), float(self.data.qvel[0])
        return np.array(
            [q, qd, self.target, self.target - q, self.previous_action],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = self.np_random.uniform(-0.35, 0.35)
        self.data.qvel[0] = self.np_random.uniform(-0.05, 0.05)
        self.target = float(self.np_random.uniform(-0.8, 0.8))
        self.previous_action = 0.0
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {"position_error": abs(self.target - self.data.qpos[0])}

    def step(self, action):
        a = float(np.clip(np.asarray(action, np.float32), -1.0, 1.0)[0])
        action_rate = a - self.previous_action
        q_des = self.target + 0.25 * a
        torque = 0.0
        for _ in range(self.frame_skip):
            q, qd = float(self.data.qpos[0]), float(self.data.qvel[0])
            torque = float(np.clip(self.kp * (q_des - q) - self.kd * qd, -5, 5))
            self.data.ctrl[0] = torque
            mujoco.mj_step(self.model, self.data)

        self.previous_action = a
        self.step_count += 1
        q, qd = float(self.data.qpos[0]), float(self.data.qvel[0])
        error = self.target - q
        terms = {
            "tracking": -2.0 * error**2,
            "velocity": -0.02 * qd**2,
            "action_rate": -0.01 * action_rate**2,
            "torque": -0.0005 * torque**2,
        }
        reward = float(sum(terms.values()))
        finite = bool(np.isfinite(self.data.qpos).all()
                      and np.isfinite(self.data.qvel).all()
                      and np.isfinite(reward))
        terminated = (not finite) or abs(q) > 1.49
        truncated = self.step_count >= self.max_steps
        info = {
            "position_error": abs(error),
            "torque": torque,
            "reward_terms": terms,
        }
        return self._obs(), reward, terminated, truncated, info


def mean_position_error(model: PPO | None, episodes: int, seed: int) -> float:
    """model=None 表示随机策略；否则使用确定性 PPO。"""
    errors = []
    for episode in range(episodes):
        env = JointReachEnv()
        obs, _ = env.reset(seed=seed + episode)
        while True:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            errors.append(info["position_error"])
            if terminated or truncated:
                break
        env.close()
    return float(np.mean(errors))


def main():
    out = Path("/tmp/joint_ppo_output")
    best = out / "best"
    logs = out / "eval"
    out.mkdir(exist_ok=True)
    best.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)

    check_env(JointReachEnv(), warn=True, skip_render_check=True)
    random_error = mean_position_error(None, episodes=10, seed=10_000)
    print(f"random mean position error: {random_error:.4f} rad")

    train_env = Monitor(JointReachEnv())
    eval_env = Monitor(JointReachEnv())
    train_env.reset(seed=0)
    eval_env.reset(seed=1_000)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best),
        log_path=str(logs),
        eval_freq=5_000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        seed=0,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs={"net_arch": [64, 64]},
        tensorboard_log=str(out / "tensorboard"),
    )
    model.learn(total_timesteps=100_000, callback=eval_callback)
    model.save(out / "joint_reach_final")
    train_env.close()
    eval_env.close()

    # 若 EvalCallback 已保存 best_model.zip，优先评估 best；否则评估 final。
    best_path = best / "best_model.zip"
    load_path = best_path if best_path.exists() else out / "joint_reach_final.zip"
    loaded = PPO.load(load_path)

    trained_error = mean_position_error(loaded, episodes=20, seed=20_000)
    print(f"trained mean position error: {trained_error:.4f} rad")
    print(f"improvement ratio: {random_error / max(trained_error, 1e-9):.2f}x")

    # SB3 自带的 episode return 评估。
    eval_env = Monitor(JointReachEnv())
    mean_return, std_return = evaluate_policy(
        loaded, eval_env, n_eval_episodes=20, deterministic=True
    )
    eval_env.close()
    print(f"return: {mean_return:.2f} +/- {std_return:.2f}")


if __name__ == "__main__":
    main()
```

## 5. 运行

```bash
python /tmp/train_joint_ppo.py
```

可选查看 TensorBoard：

```bash
tensorboard --logdir /tmp/joint_ppo_output/tensorboard
```

低配置 CPU 可先把 `total_timesteps` 改为 `10_000`，只验证训练、保存、加载链路；**10k smoke run 不等于收敛**。

## 6. 如何判断真的学会了

训练日志中的 reward 上升只是线索。至少同时检查：

1. 固定评估 seed；
2. `deterministic=True`；
3. 训练策略的平均位置误差显著小于随机策略；
4. episode 中没有大量关节限位 termination；
5. action 和 torque 没有长期饱和。

一次合理结果可能类似：

```text
random mean position error: 0.35 rad
trained mean position error: 0.04 rad
improvement ratio: 8.75x
```

具体数值依赖 MuJoCo、SB3、PyTorch 版本和训练步数，不应硬套。

## 7. 参数与控制意义

| PPO 参数 | 本章值 | 直觉 |
|---|---:|---|
| `n_steps` | 1024 | 更新前收集多少环境步 |
| `batch_size` | 64 | 每次梯度计算用多少样本 |
| `gamma` | 0.99 | 对未来 reward 的重视程度 |
| `gae_lambda` | 0.95 | advantage 偏差/方差折中 |
| `clip_range` | 0.2 | 限制策略更新幅度 |

不要一次同时修改全部参数。优先检查环境和 reward，再考虑调 PPO。

## 8. 常见失败

### Reward 越训越差

先检查 reward 符号。若误差项写成正数，策略会主动增大误差。

### Action 一直顶在 ±1

常见原因：`action_scale` 太小导致策略只能打满，或 torque/Kp 太小无法到达目标。也可能 reward 缺少 action/torque 约束。

### 训练很慢

训练时不要开 viewer。MuJoCo headless 通常远快于实时；物理仿真 4 秒不必消耗现实 4 秒。

### 同一个模型每次结果差很多

训练有随机性。固定环境、SB3 和评估 seed；比较多组 seed，而不是只挑最好的一次。

### 模型加载后 shape 错误

训练和推理必须使用完全相同的 observation 顺序、尺度和 action mapping。

## 9. 安全提示

PPO 只优化你写下的 reward，并不自动理解“安全”。即使单关节误差很小，也要保留 action clipping、目标角限制和 torque clipping。进入 T800 后还要增加姿态、碰撞和跌倒终止。

## 10. 小练习

1. 用 zero action 增加第三个基线，并与 random/PPO 比较。
2. 把 `action_scale` 改为 0.1，重新训练并分析 action 饱和率。
3. 把速度惩罚从 0.02 改为 0，观察是否更容易振荡。
4. 用 3 个训练 seed，报告平均值而不是最好结果。

## 11. 验收清单

- [ ] 环境通过 `check_env`。
- [ ] 低步数 smoke train 能保存和加载模型。
- [ ] 加载后的模型能完成确定性推理。
- [ ] 使用独立于训练的评估 seed。
- [ ] 训练策略平均位置误差优于 random baseline。
- [ ] 没有把训练 reward 当作唯一指标。

## 12. 与 SDK 的关系

SDK 的 `rl_walking_example_runner` 是**推理/部署端**：读取 observation、运行 MNN、把 action 映射为目标角，然后发送 PD 命令。本章补上了它不包含的训练环节，但任务仍只是单关节。下一章会把相同模式扩展到 T800 的 12 个腿部关节。