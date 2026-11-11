from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


JOINT_NAMES = [
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

# 来自 SDK pd_stand/default.yaml；顺序严格对应上面的 JOINT_NAMES。
DEFAULT_Q = np.array([
    -0.105,  0.105,  0.263, 0.145, -0.051, -0.074,
    -0.105, -0.105, -0.263, 0.145, -0.051,  0.074,
     0.0,
     0.0, 0.05, 0.0, -0.1, 0.0,
     0.0, -0.05, 0.0, -0.1, 0.0,
     0.0, 0.0,
], dtype=np.float64)

# 教学用保守增益，不直接使用 SDK pd_stand 的高增益。
KP = np.array([
    140, 100, 80, 160, 35, 25,
    140, 100, 80, 160, 35, 25,
    50,
    30, 25, 20, 25, 12,
    30, 25, 20, 25, 12,
    10, 10,
], dtype=np.float64)
KD = np.array([
    5, 4, 3, 5, 1.2, 1.0,
    5, 4, 3, 5, 1.2, 1.0,
    2,
    1, 1, 0.8, 1, 0.5,
    1, 1, 0.8, 1, 0.5,
    0.5, 0.5,
], dtype=np.float64)
ACTION_SCALE = np.array([
    0.18, 0.10, 0.10, 0.22, 0.14, 0.10,
    0.18, 0.10, 0.10, 0.22, 0.14, 0.10,
], dtype=np.float64)


class T800StandEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str = "models/t800/t800.xml",
        render_mode: str | None = None,
        curriculum_level: int = 1,
    ):
        super().__init__()
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.render_mode = render_mode
        self.curriculum_level = int(np.clip(curriculum_level, 0, 3))
        self.model = mujoco.MjModel.from_xml_path(str(Path(xml_path).resolve()))
        self.data = mujoco.MjData(self.model)
        if self.model.nu != 25:
            raise ValueError(f"expected 25 actuators, got {self.model.nu}")
        if not np.isclose(self.model.opt.timestep, 0.002):
            raise ValueError(f"expected dt=0.002, got {self.model.opt.timestep}")

        self.qadr, self.dadr, self.actuator_ids = [], [], []
        self.joint_low, self.joint_high = [], []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}"
            )
            if jid < 0 or aid < 0:
                raise ValueError(f"missing joint/actuator: {name}")
            self.qadr.append(int(self.model.jnt_qposadr[jid]))
            self.dadr.append(int(self.model.jnt_dofadr[jid]))
            self.actuator_ids.append(aid)
            self.joint_low.append(self.model.jnt_range[jid, 0])
            self.joint_high.append(self.model.jnt_range[jid, 1])

        self.qadr = np.asarray(self.qadr)
        self.dadr = np.asarray(self.dadr)
        self.actuator_ids = np.asarray(self.actuator_ids)
        self.joint_low = np.asarray(self.joint_low)
        self.joint_high = np.asarray(self.joint_high)
        self.ctrl_limit = np.min(
            np.abs(self.model.actuator_ctrlrange[self.actuator_ids]), axis=1
        )
        # 教学降额：不超过 MJCF 硬上限的 25%，且最多 80 Nm。
        self.torque_limit = np.minimum(0.25 * self.ctrl_limit, 80.0)

        # 第一个 joint 是 freejoint，使用模型地址表而不是假定业务 joint 偏移。
        free_jid = next(
            j for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        )
        self.base_qadr = int(self.model.jnt_qposadr[free_jid])
        self.base_dadr = int(self.model.jnt_dofadr[free_jid])

        self.leg_idx = np.arange(12)
        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
        self.observation_space = spaces.Box(-100.0, 100.0, (47,), np.float32)
        self.frame_skip = 5
        self.max_steps = 1000       # 10 s
        self.target_slew = 0.025    # rad per policy step
        self.target_height = 1.03
        self.step_count = 0
        self.previous_action = np.zeros(12)
        self.q_target = DEFAULT_Q.copy()
        self.last_torque = np.zeros(25)
        self._viewer = None

    def _q(self):
        return self.data.qpos[self.qadr].copy()

    def _qd(self):
        return self.data.qvel[self.dadr].copy()

    def _base_features(self):
        quat = self.data.qpos[self.base_qadr + 3:self.base_qadr + 7]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        rotation = rotation.reshape(3, 3)  # body -> world
        projected_gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        base_linear_velocity = self.data.qvel[
            self.base_dadr:self.base_dadr + 3
        ].copy()
        try:
            angular_velocity = self.data.sensor("imu_angular_velocity").data.copy()
        except KeyError:
            angular_velocity = self.data.qvel[
                self.base_dadr + 3:self.base_dadr + 6
            ].copy()
        return projected_gravity, base_linear_velocity, angular_velocity

    def _foot_contacts(self):
        # 归一化脚部力强度；无接触约为 0，有接触逐渐接近 1。
        left = np.linalg.norm(self.data.sensor("force_left_foot").data)
        right = np.linalg.norm(self.data.sensor("force_right_foot").data)
        return np.tanh(np.array([left, right]) / 300.0)

    def _obs(self):
        q, qd = self._q(), self._qd()
        gravity, base_vel, angular_vel = self._base_features()
        obs = np.concatenate([
            q[self.leg_idx] - DEFAULT_Q[self.leg_idx],
            0.05 * qd[self.leg_idx],
            self.previous_action,
            angular_vel,
            gravity,
            base_vel,
            self._foot_contacts(),
        ])
        assert obs.shape == (47,)
        return np.clip(obs, -100.0, 100.0).astype(np.float32)

    def _roll_pitch(self):
        quat = self.data.qpos[self.base_qadr + 3:self.base_qadr + 7]
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, quat)
        r = mat.reshape(3, 3)
        pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
        roll = np.arctan2(r[2, 1], r[2, 2])
        return float(roll), float(pitch)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = DEFAULT_Q

        # 课程 0: 准确站姿；1: 关节噪声；2: 再加 base 倾斜/速度；3: step 中加外力。
        if self.curriculum_level >= 1:
            self.data.qpos[self.qadr] += self.np_random.uniform(-0.015, 0.015, 25)
        if self.curriculum_level >= 2:
            roll, pitch = self.np_random.uniform(-0.025, 0.025, 2)
            quat = np.empty(4)
            # 小角度 rotation vector -> quaternion。
            axis_angle = np.array([roll, pitch, 0.0])
            mujoco.mju_axisAngle2Quat(
                quat,
                axis_angle / max(np.linalg.norm(axis_angle), 1e-12),
                np.linalg.norm(axis_angle),
            )
            self.data.qpos[self.base_qadr + 3:self.base_qadr + 7] = quat
            self.data.qvel[self.base_dadr:self.base_dadr + 6] = \
                self.np_random.uniform(-0.03, 0.03, 6)

        self.previous_action.fill(0.0)
        self.q_target = DEFAULT_Q.copy()
        self.last_torque.fill(0.0)
        self.step_count = 0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), self._info({})

    def _info(self, reward_terms):
        roll, pitch = self._roll_pitch()
        height = float(self.data.qpos[self.base_qadr + 2])
        return {
            "reward_terms": reward_terms,
            "base_height": height,
            "max_tilt": max(abs(roll), abs(pitch)),
            "mean_abs_torque": float(np.mean(np.abs(self.last_torque))),
            "torque_saturation": float(np.mean(
                np.abs(self.last_torque) >= 0.999 * self.torque_limit
            )),
        }

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1.0, 1.0).astype(float)
        desired = DEFAULT_Q.copy()
        desired[self.leg_idx] += action * ACTION_SCALE
        desired = np.clip(desired, self.joint_low + 0.03, self.joint_high - 0.03)
        delta = np.clip(desired - self.q_target, -self.target_slew, self.target_slew)
        self.q_target += delta

        # curriculum 3：偶尔施加一个很小的水平外力，仅持续一个 policy step。
        self.data.xfrc_applied[:] = 0.0
        if self.curriculum_level >= 3 and self.np_random.random() < 0.02:
            base_body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "LINK_BASE"
            )
            self.data.xfrc_applied[base_body, :2] = self.np_random.uniform(-20, 20, 2)

        for _ in range(self.frame_skip):
            q, qd = self._q(), self._qd()
            raw_torque = KP * (self.q_target - q) - KD * qd
            self.last_torque = np.clip(raw_torque, -self.torque_limit, self.torque_limit)
            self.data.ctrl[self.actuator_ids] = self.last_torque
            mujoco.mj_step(self.model, self.data)

        self.data.xfrc_applied[:] = 0.0
        self.step_count += 1
        q, qd = self._q(), self._qd()
        gravity, _, angular_vel = self._base_features()
        contacts = self._foot_contacts()
        height = float(self.data.qpos[self.base_qadr + 2])
        action_rate = action - self.previous_action

        terms = {
            "upright": 2.0 * float(np.exp(-6.0 * np.sum(
                (gravity - np.array([0.0, 0.0, -1.0])) ** 2
            ))),
            "height": 1.0 * float(np.exp(-20.0 * (height - self.target_height) ** 2)),
            "pose": 0.5 * float(np.exp(-2.0 * np.mean((q - DEFAULT_Q) ** 2))),
            "feet_contact": 0.2 * float(np.mean(contacts)),
            "survival": 0.2,
            "joint_velocity": -0.002 * float(np.mean(qd ** 2)),
            "base_angular_velocity": -0.02 * float(np.sum(angular_vel ** 2)),
            "action_rate": -0.01 * float(np.mean(action_rate ** 2)),
            "torque": -0.00001 * float(np.mean(self.last_torque ** 2)),
        }
        reward = float(sum(terms.values()))
        self.previous_action = action.copy()

        roll, pitch = self._roll_pitch()
        finite = bool(
            np.isfinite(self.data.qpos).all()
            and np.isfinite(self.data.qvel).all()
            and np.isfinite(reward)
        )
        low_height = height < 0.72
        excessive_tilt = abs(roll) > 0.8 or abs(pitch) > 0.8
        joint_violation = bool(np.any(q < self.joint_low - 1e-3)
                               or np.any(q > self.joint_high + 1e-3))
        terminated = (not finite) or low_height or excessive_tilt or joint_violation
        truncated = self.step_count >= self.max_steps
        info = self._info(terms)
        if terminated:
            if not finite:
                info["termination_reason"] = "non_finite"
            elif low_height:
                info["termination_reason"] = "low_height"
            elif excessive_tilt:
                info["termination_reason"] = "excessive_tilt"
            else:
                info["termination_reason"] = "joint_limit"

        if self.render_mode == "human":
            self.render()
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                import mujoco.viewer
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


def evaluate(model, episodes=10, seed=10_000, mode="ppo", render=False):
    records = []
    for ep in range(episodes):
        env = T800StandEnv(
            render_mode="human" if render and ep == 0 else None,
            curriculum_level=2,
        )
        obs, _ = env.reset(seed=seed + ep)
        rewards, heights, tilts, torques, saturations = [], [], [], [], []
        terminated = truncated = False
        while not (terminated or truncated):
            if mode == "random":
                action = env.action_space.sample()
            elif mode in ("zero", "pd"):
                action = np.zeros(12, dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            heights.append(info["base_height"])
            tilts.append(info["max_tilt"])
            torques.append(info["mean_abs_torque"])
            saturations.append(info["torque_saturation"])
        records.append({
            "success": truncated and not terminated,
            "duration": len(rewards) * 0.01,
            "return": sum(rewards),
            "min_height": min(heights),
            "max_tilt": max(tilts),
            "mean_torque": np.mean(torques),
            "saturation": np.mean(saturations),
        })
        env.close()

    print(f"\n[{mode}]")
    for key in records[0]:
        print(f"{key:>12}: {np.mean([r[key] for r in records]):.4f}")


def make_env(seed):
    def factory():
        env = Monitor(T800StandEnv(curriculum_level=1))
        env.reset(seed=seed)
        return env
    return factory


def train(total_timesteps):
    out = Path("/tmp/t800_stand_output")
    out.mkdir(exist_ok=True)
    train_env = DummyVecEnv([make_env(i) for i in range(4)])
    eval_env = Monitor(T800StandEnv(curriculum_level=2))
    eval_env.reset(seed=5_000)
    callback = EvalCallback(
        eval_env,
        best_model_save_path=str(out / "best"),
        log_path=str(out / "eval"),
        eval_freq=max(10_000 // 4, 1),  # callback 按 vector env call 计数
        n_eval_episodes=5,
        deterministic=True,
    )
    model = PPO(
        "MlpPolicy", train_env, seed=0, verbose=1,
        learning_rate=3e-4, n_steps=2048, batch_size=256,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        policy_kwargs={"net_arch": [256, 128]},
        tensorboard_log=str(out / "tensorboard"),
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(out / "latest")
    train_env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", type=int, default=0, metavar="STEPS")
    parser.add_argument("--eval", type=str, default="")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        env = T800StandEnv(curriculum_level=0)
        check_env(env, warn=True, skip_render_check=True)
        obs, _ = env.reset(seed=123)
        assert obs.shape == (47,)
        t0 = env.data.time
        _, reward, _, _, info = env.step(np.zeros(12, np.float32))
        assert np.isclose(env.data.time - t0, 5 * env.model.opt.timestep)
        assert np.isclose(reward, sum(info["reward_terms"].values()))
        env.close()
        print("smoke checks passed")

    if args.train > 0:
        train(args.train)

    if args.eval:
        model = PPO.load(args.eval)
        evaluate(None, mode="random")
        evaluate(None, mode="zero")
        # 本环境中 zero 与 pd 都表示默认 target + 底层 PD；保留两个名字强调基线含义。
        evaluate(None, mode="pd")
        evaluate(model, mode="ppo", render=args.render)


if __name__ == "__main__":
    main()