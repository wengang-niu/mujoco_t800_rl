"""04 — T800 站立 Gymnasium 环境（手写，不用 stable-baselines3）。

把 T800 包装成标准 Gymnasium 环境 T800StandEnv，供 05 的手写 PPO 训练：

- 动作空间：Box(-1,1,(4,))  —— 仅双踝 pitch/roll 的目标角偏移（站立平衡的可行子空间）
- 观测空间：Box(-10,10,(39,)) —— 腿误差/速度 + 上一动作 + 基座角速度/重力/线速度 + 足底接触
- 控制方式：PD 内环（动作 -> 目标角偏移 -> 力矩），frame_skip=5（策略 100Hz）

为什么动作只留双踝 pitch/roll：实测 T800 是「重躯干(质心0.95m)+轻腿+11.4cm小脚」的
硬倒立摆，站立平衡关键只在踝；髋/膝/踝roll 等其余 10 个腿关节对平衡是噪声（且初始
大探索下会淹没踝控制梯度）。修复踝力矩限幅（≤35N·m，否则轻脚被 80N·m 发射离地）后，
一个 2 维踝 pitch 手调 PD 即可站满 10s，故动作空间收敛到该可验证子空间。
- 奖励：直立度 + 高度 + 足底接触 + 存活 - 速度/动作变化/力矩惩罚
- 终止：倒下（高度过低或倾角过大）、关节超限、NaN

注意：高频 PD 循环用 t800_common.build_addr 预计算的地址 + numpy 花式索引，
避免每个物理子步做 ~100 次 mj_name2id 字符串查找（否则训练慢到不可用）。

运行自检（含吞吐量实测）：python 04_stand_env.py
"""
from __future__ import annotations

import time

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import t800_common as tc


# 动作只作用于双踝 pitch/roll（相对 leg_idx 的 12 维下标：踝 pitch L/roll L/pitch R/roll R）。
# 这是站立平衡的可验证子空间：其余腿关节对平衡是噪声，锁定在名义姿态。
ACTION_LEG_IDX = np.array([4, 5, 10, 11], dtype=int)


class T800StandEnv(gym.Env):
    """浮动基座双足保持站立 + 抗扰动的强化学习环境。"""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.model, self.data = tc.load_model()
        self.addr = tc.build_addr(self.model)              # 预计算地址（高频用）
        self.leg_idx = self.addr.leg                       # 腿关节在 25 维里的下标
        self.torque_lim = self.addr.tau_lim
        self.jnt_range = self.model.jnt_range[self.addr.jid[self.leg_idx]]  # 腿关节行程 [12,2]

        self.action_space = spaces.Box(-1.0, 1.0, (len(ACTION_LEG_IDX),), dtype=np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (39,), dtype=np.float32)

        self.frame_skip = tc.FRAME_SKIP
        self.max_steps = tc.MAX_STEPS
        self._steps = 0
        self._prev_action = np.zeros(len(ACTION_LEG_IDX), dtype=np.float64)

    # ---- 核心接口 ----

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        q0 = tc.NOMINAL_Q.copy()
        q0[self.leg_idx] += self.np_random.uniform(-0.02, 0.02, size=12)  # 腿关节小噪声
        tc.reset_to_standing(self.model, self.data, q=q0)
        self.data.qvel[3:6] += self.np_random.uniform(-0.02, 0.02, size=3)  # 基座角速度扰动
        self._steps = 0
        self._prev_action = np.zeros(len(ACTION_LEG_IDX), dtype=np.float64)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        q_des = tc.NOMINAL_Q.copy()
        leg_target = tc.NOMINAL_Q[self.leg_idx].copy()
        # 动作只作用于双踝 pitch/roll，其余腿关节锁定在名义姿态
        leg_target[ACTION_LEG_IDX] += action * tc.ACTION_SCALE[ACTION_LEG_IDX]
        # 目标角限制在关节行程内（留 0.03 裕量）
        lo = self.jnt_range[:, 0] + 0.03
        hi = self.jnt_range[:, 1] - 0.03
        q_des[self.leg_idx] = np.clip(leg_target, lo, hi)

        # PD 内环：frame_skip 个物理步内重复施加同一目标（向量化）
        tau = None
        for _ in range(self.frame_skip):
            tau = tc.pd_torque_fast(self.data, self.addr, q_des)  # 已按力矩限幅
            self.data.ctrl[self.addr.act] = tau
            mujoco.mj_step(self.model, self.data)

        obs = self._obs()
        reward, terms = self._reward(action, tau)
        terminated = self._terminated()
        truncated = self._steps >= self.max_steps
        self._steps += 1
        self._prev_action = action
        return obs, float(reward), terminated, truncated, {"reward_terms": terms}

    # ---- 观测 / 奖励 / 终止 ----

    def _obs(self) -> np.ndarray:
        q = self.data.qpos[self.addr.qpos[self.leg_idx]]          # 12 腿关节位置
        qd = self.data.qvel[self.addr.dof[self.leg_idx]]          # 12 腿关节速度
        ang_vel = self.data.qvel[3:6]          # 基座角速度（世界系）
        lin_vel = self.data.qvel[:3]           # 基座线速度（世界系）
        gravity = tc.projected_gravity(self.data)
        contact = np.tanh(tc.foot_contacts_fast(self.data, self.addr) / 400.0)
        return np.concatenate([
            q - tc.NOMINAL_Q[self.leg_idx],    # 0:12  腿关节误差
            0.05 * qd,                          # 12:24 腿速度（缩放）
            self._prev_action,                  # 24:28 上一动作（踝4维）
            ang_vel,                            # 28:31 基座角速度
            gravity,                            # 31:34 投影重力
            lin_vel,                            # 34:37 基座线速度
            contact,                            # 37:39 足底接触
        ]).astype(np.float32)

    def _reward(self, action: np.ndarray, tau: np.ndarray) -> tuple[float, dict]:
        gravity = tc.projected_gravity(self.data)
        height = tc.base_height(self.data)
        q = self.data.qpos[self.addr.qpos]
        qd = self.data.qvel[self.addr.dof]
        ang_vel = self.data.qvel[3:6]
        lin_vel = self.data.qvel[:3]
        contact = np.tanh(tc.foot_contacts_fast(self.data, self.addr) / 400.0)
        terms = {
            "upright": 2.0 * np.exp(-6.0 * np.sum((gravity - np.array([0, 0, -1])) ** 2)),
            "height": 1.0 * np.exp(-20.0 * (height - tc.TARGET_HEIGHT) ** 2),
            "feet_contact": 0.2 * float(np.mean(contact)),
            "survival": 0.2,
            "joint_velocity": -0.002 * float(np.mean(qd ** 2)),
            "base_ang_vel": -0.3 * float(np.sum(ang_vel ** 2)),
            "base_lin_vel": -0.3 * float(np.sum(lin_vel ** 2)),
            "action_rate": -0.01 * float(np.mean((action - self._prev_action) ** 2)),
            "torque": -1e-5 * float(np.mean(tau ** 2)),
        }
        return float(sum(terms.values())), terms

    def _terminated(self) -> bool:
        d = self.data
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            return True
        if tc.base_height(d) < 0.6:                      # 倒下
            return True
        if np.arccos(np.clip(-tc.projected_gravity(d)[2], -1, 1)) > 0.8:  # 倾角过大
            return True
        q = self.data.qpos[self.addr.qpos[self.leg_idx]]
        lo = self.jnt_range[:, 0] + 0.03
        hi = self.jnt_range[:, 1] - 0.03
        return bool(np.any(q < lo) or np.any(q > hi))    # 关节超限


if __name__ == "__main__":
    # 自检 1：随机动作 rollout，验证形状、无 NaN、终止与 reset 正常
    env = T800StandEnv()
    obs, _ = env.reset()
    print(f"obs shape = {obs.shape}, action shape = {env.action_space.shape}")
    total_reward = 0.0
    for ep in range(3):
        obs, _ = env.reset()
        ep_reward = 0.0
        for t in range(env.max_steps):
            action = env.action_space.sample()  # 随机动作
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            assert obs.shape == (39,), obs.shape
            assert np.isfinite(obs).all(), "obs 出现 NaN"
            if terminated or truncated:
                break
        total_reward += ep_reward
        print(f"episode {ep}: length={t + 1}, reward={ep_reward:.1f}, "
              f"terminated={terminated}")
    print(f"mean episode reward (random policy) = {total_reward / 3:.1f}")

    # 自检 2：吞吐量实测（训练每秒能跑多少 env.step）
    env.reset()
    n_bench = 500
    action = np.zeros(len(ACTION_LEG_IDX))
    t0 = time.time()
    for _ in range(n_bench):
        env.step(action)
        if env._steps >= env.max_steps:
            env.reset()
    dt = time.time() - t0
    print(f"\n吞吐量: {n_bench / dt:.0f} env.step/s（每步含 {env.frame_skip} 个物理子步）")
    print(f"预计 2048 步/迭代 收集耗时 ≈ {2048 / (n_bench / dt):.1f}s")
    print("环境自检通过。")
