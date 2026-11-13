"""02 — 掌握 T800 的状态与控制接口。

在 01 的基础上，演示如何按名称读写关节状态、传感器读数，以及
如何给力矩电机下发 data.ctrl。跑一个固定的控制循环并打印观察结果，
为后续 PD 站立（03）和 RL 环境（04）打基础。

运行：conda activate t800 && python 02_control_interface.py
"""
from __future__ import annotations

import mujoco
import numpy as np

import t800_common as tc


def main() -> None:
    model, data = tc.load_model()

    # 1. 模型尺寸
    print(f"nq / nv / nu = {model.nq} / {model.nv} / {model.nu}   (nu=25 个力矩电机)")
    print(f"nsensordata    = {model.nsensordata}")
    print(f"timestep       = {model.opt.timestep:.3f} s\n")

    # 2. 按名称定位一个关节/电机（自由关节在前，必须按名称寻址而非切片）
    name = "J03_KNEE_PITCH_L"
    print(f"关节 {name}: qpos_adr={tc.qpos_adr(model, name)}, "
          f"dof_adr={tc.dof_adr(model, name)}, "
          f"actuator_id={tc.actuator_id(model, name)}")
    print(f"其电机 ctrlrange = {model.actuator_ctrlrange[tc.actuator_id(model, name)]} N·m\n")

    # 3. 复位后的初始状态与传感器
    print("初始腿关节 qpos :", np.round(tc.joint_qpos(model, data, tc.LEG_NAMES[:6]), 4), "(左腿前 6 个)")
    print("初始基座 qpos   :", np.round(data.qpos[:7], 4), "(位置3 + 四元数4)")
    print("投影重力(初始)  :", np.round(tc.projected_gravity(data), 4), "(应≈[0,0,-1])")
    print("足底力(初始)    :", np.round(tc.foot_contacts(data), 2), "N\n")

    # 4. 给左膝一个恒定力矩，观察 qvel 变化（验证力矩下发链路）
    knee_act = tc.actuator_id(model, name)
    torque = 30.0  # N·m
    print(f"给 {name} 施加 {torque} N·m 恒力矩，跑 200 步，观察关节速度：")
    print(f"{'step':>6} {'knee_qpos':>12} {'knee_qvel':>12} {'base_z':>8}")
    for step in range(200):
        data.ctrl[knee_act] = torque
        mujoco.mj_step(model, data)  # 推进一个物理步
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError("仿真状态出现 NaN 或 Inf")
        if step % 40 == 0:
            q = data.qpos[tc.qpos_adr(model, name)]
            qd = data.qvel[tc.dof_adr(model, name)]
            print(f"{step:>6} {q:>12.4f} {qd:>12.4f} {data.qpos[2]:>8.4f}")

    # 5. 读取 IMU 与足底力（验证传感器链路）
    print("\nIMU 角速度(陀螺) :", np.round(data.sensor("imu_angular_velocity").data, 4), "rad/s")
    print("IMU 线加速度     :", np.round(data.sensor("imu_linear_acceleration").data, 4), "m/s^2")
    print("左右足底力       :", np.round(tc.foot_contacts(data), 2), "N")

    print("\nOK: 状态、传感器、力矩下发三条链路均已打通。")


if __name__ == "__main__":
    main()
