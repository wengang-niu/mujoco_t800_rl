"""06 — 评估训练好的站立策略。

加载 05 训练出的 actor 权重，评估机器人能否站满全程（默认 10s），
并统计存活时长、成功率、最小基座高度、最大倾角。可选：
  - viewer 展示站立
  - 周期性水平扰动力，演示抗扰动

用法：
  python 06_eval_stand.py                       # headless 评估最终模型
  python 06_eval_stand.py --ckpt checkpoints/ppo_stand_1000.pt --episodes 20
  python 06_eval_stand.py --viewer              # 打开 viewer 看站立
  python 06_eval_stand.py --viewer --disturb    # viewer + 周期性扰动
"""
from __future__ import annotations

import argparse
import importlib

import mujoco
import numpy as np

import t800_common as tc


def make_env():
    return importlib.import_module("04_stand_env").T800StandEnv()


def make_ppo():
    return importlib.import_module("05_ppo").PPO(39, 4)


def root_body_id(model: mujoco.MjModel) -> int:
    """浮动基座双足的根 body（施加扰动力的作用点）。"""
    return int(model.body_rootid[0])


def apply_disturbance(env, t: int, force: float, period: float) -> None:
    """给根 body 施加水平正弦扰动力（沿 x 轴，模拟前后推搡）。"""
    f = force * np.sin(2 * np.pi * t / period)
    env.data.xfrc_applied[root_body_id(env.model), :3] = [f, 0.0, 0.0]


def run_episode(env, ppo, disturb: bool, force: float, period: float):
    """跑一回合，返回 (总奖励, 存活步数, 是否站满, 最小高度, 最大倾角)。"""
    obs, _ = env.reset()
    min_h = tc.base_height(env.data)
    max_tilt = 0.0
    total = 0.0
    for t in range(env.max_steps):
        if disturb:
            apply_disturbance(env, t, force, period)
        action = ppo.act_deterministic(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        h = tc.base_height(env.data)
        g = tc.projected_gravity(env.data)
        tilt = float(np.arccos(np.clip(-g[2], -1.0, 1.0)))
        min_h = min(min_h, h)
        max_tilt = max(max_tilt, tilt)
        if terminated:
            return total, t + 1, False, min_h, max_tilt
    return total, env.max_steps, True, min_h, max_tilt


def eval_headless(env, ppo, episodes: int, disturb: bool, force: float, period: float):
    lens, success, min_hs, max_tilts, rets = [], 0, [], [], []
    for _ in range(episodes):
        ret, length, ok, min_h, max_tilt = run_episode(env, ppo, disturb, force, period)
        lens.append(length); min_hs.append(min_h); max_tilts.append(max_tilt); rets.append(ret)
        success += int(ok)

    dt = env.model.opt.timestep * env.frame_skip  # 每策略步的仿真时长
    print(f"episodes        : {episodes}")
    print(f"成功率(站满全程) : {success}/{episodes} = {100*success/episodes:.0f}%")
    print(f"平均存活时长     : {np.mean(lens)*dt:.2f}s (满分 {env.max_steps*dt:.1f}s)")
    print(f"最小基座高度     : {np.min(min_hs):.3f}m (倒下阈值 0.6m)")
    print(f"最大倾角         : {np.rad2deg(np.max(max_tilts)):.1f}° (终止阈值 45.8°)")
    print(f"平均回合奖励     : {np.mean(rets):.1f}")
    return success / episodes


def run_viewer(env, ppo, disturb: bool, force: float, period: float):
    env.reset()
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        step = 0
        while viewer.is_running() and env.data.time < env.max_steps * env.model.opt.timestep * env.frame_skip:
            if disturb:
                apply_disturbance(env, step, force, period)
            obs = env._obs()
            action = ppo.act_deterministic(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            viewer.sync()
            step += 1
            if terminated:
                env.reset()
                step = 0
    print(f"viewer 结束：模拟时长 {env.data.time:.2f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/ppo_stand_final.pt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--disturb", action="store_true", help="施加周期性水平扰动力")
    p.add_argument("--force", type=float, default=30.0, help="扰动力幅值 N")
    p.add_argument("--period", type=float, default=1.0, help="扰动周期 s")
    args = p.parse_args()

    env = make_env()
    ppo = make_ppo()
    ppo.load(args.ckpt)
    print(f"已加载策略：{args.ckpt}")

    if args.viewer:
        run_viewer(env, ppo, args.disturb, args.force, args.period)
    else:
        eval_headless(env, ppo, args.episodes, args.disturb, args.force, args.period)


if __name__ == "__main__":
    main()
