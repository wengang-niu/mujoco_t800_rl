"""03 — 手写 PD 站立（非 RL 基线）。

用 PD 控制器把全身 25 个关节锁在名义站立姿态，验证 PD 内环（力矩方向、
增益、按名称寻址）可用，并拿到后续奖励函数需要的信号。

注意：浮动基座双足是倒立摆系统，纯固定 PD 无法长期站稳（约 1~2s 会
向前倾倒，因为质心在踝关节前方、足底压力中心落在质心后方）。这并非
代码错误，而是这类机器人需要 RL 主动调节目标姿态来实时平衡的根本原因。

用法：
  python 03_pd_stand.py              # 打开 viewer 观看
  python 03_pd_stand.py --headless   # 无窗口跑 10s，打印诊断（验证用）
"""
from __future__ import annotations

import argparse

import mujoco
import numpy as np

import t800_common as tc


def main() -> None:
    args = parse_args()
    model, data = tc.load_model()
    tc.reset_to_standing(model, data)

    print(f"PD 站立：target_height={tc.TARGET_HEIGHT}m, 时长={args.seconds}s, 步长={model.opt.timestep}s")

    if args.headless:
        run_headless(model, data, args.seconds)
    else:
        run_viewer(model, data, args.seconds)


def run_headless(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> None:
    """无窗口跑仿真，周期性打印诊断（机器人应站稳不倒）。"""
    total_steps = int(seconds / model.opt.timestep)
    print(f"{'t(s)':>6} {'base_z':>8} {'roll':>7} {'pitch':>7} {'max|err|':>9} {'footL':>8} {'footR':>8}")
    for step in range(total_steps):
        tau = tc.pd_torque(model, data, tc.NOMINAL_Q)   # 目标 = 名义站立角
        tc.apply_torque(model, data, tau)
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("仿真状态出现 NaN 或 Inf")
        if step % int(0.5 / model.opt.timestep) == 0:
            t = step * model.opt.timestep
            roll, pitch = tc.base_roll_pitch(data)
            q = tc.joint_qpos(model, data, tc.JOINT_NAMES)
            err = np.max(np.abs(q - tc.NOMINAL_Q))
            fl, fr = tc.foot_contacts(data)
            print(f"{t:>6.2f} {data.qpos[2]:>8.4f} {roll:>7.3f} {pitch:>7.3f} {err:>9.4f} {fl:>8.1f} {fr:>8.1f}")

    print(f"\n最终 base_z={data.qpos[2]:.4f}")
    print("预期：机器人约 1~2s 后向前倾倒（倒立摆不稳），这正说明需要 RL 主动平衡。")
    print("若 PD 内环力矩方向正确、诊断量变化合理，即达到本步目的。")


def run_viewer(model: mujoco.MjModel, data: mujoco.MjData, seconds: float) -> None:
    """在 viewer 中观看 PD 站立。"""
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < seconds:
            tau = tc.pd_torque(model, data, tc.NOMINAL_Q)
            tc.apply_torque(model, data, tau)
            mujoco.mj_step(model, data)
            viewer.sync()
    print(f"simulated time : {data.time:.3f} s")
    print(f"base xyz       : {np.round(data.qpos[:3], 4)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--headless", action="store_true", help="无窗口跑仿真")
    p.add_argument("--seconds", type=float, default=10.0, help="仿真时长（秒）")
    return p.parse_args()


if __name__ == "__main__":
    main()
