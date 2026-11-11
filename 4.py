import argparse
import math
from pathlib import Path
import time

import mujoco
import numpy as np


MODEL_PATH = Path("models/t800/t800.xml")

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

# 与 SDK pd_stand/default.yaml 顺序一致，单位 rad。
Q_GOAL = np.array([
    -0.105,  0.105,  0.263,  0.145, -0.051, -0.074,
    -0.105, -0.105, -0.263,  0.145, -0.051,  0.074,
     0.000,
     0.000,  0.050,  0.000, -0.100,  0.000,
     0.000, -0.050,  0.000, -0.100,  0.000,
     0.000,  0.000,
], dtype=np.float64)

# 教学用保守增益；不是 SDK 真机参数。
KP = np.array([
    120, 80, 80, 140, 40, 30,
    120, 80, 80, 140, 40, 30,
    40,
    30, 25, 20, 25, 10,
    30, 25, 20, 25, 10,
    8, 8,
], dtype=np.float64)
KD = np.array([
    6, 5, 5, 7, 2, 2,
    6, 5, 5, 7, 2, 2,
    2,
    1.5, 1.2, 1.0, 1.2, 0.5,
    1.5, 1.2, 1.0, 1.2, 0.5,
    0.5, 0.5,
], dtype=np.float64)


def object_id(model, obj_type, name):
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise KeyError(f"模型中找不到 {name}")
    return int(idx)


def build_joint_map(model):
    qpos_adr, dof_adr, actuator_ids = [], [], []
    for name in JOINT_NAMES:
        jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"{name} 不是 hinge joint")

        aid = object_id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}"
        )
        transmitted_joint = int(model.actuator_trnid[aid, 0])
        if transmitted_joint != jid:
            raise ValueError(f"motor_{name} 没有连接到 {name}")

        qpos_adr.append(int(model.jnt_qposadr[jid]))
        dof_adr.append(int(model.jnt_dofadr[jid]))
        actuator_ids.append(aid)

    if len(set(actuator_ids)) != 25:
        raise ValueError("25 个关节必须一一对应 25 个 actuator")
    return (
        np.asarray(qpos_adr),
        np.asarray(dof_adr),
        np.asarray(actuator_ids),
    )


def quintic(q_start, q_goal, elapsed, duration):
    s = np.clip(elapsed / duration, 0.0, 1.0)
    blend = 10*s**3 - 15*s**4 + 6*s**5
    blend_dot = (30*s**2 - 60*s**3 + 30*s**4) / duration
    q_des = q_start + blend * (q_goal - q_start)
    qd_des = blend_dot * (q_goal - q_start)
    return q_des, qd_des


def roll_pitch_from_quat(quat_wxyz):
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, quat_wxyz)
    r = rotation.reshape(3, 3)
    pitch = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    roll = math.atan2(float(r[2, 1]), float(r[2, 2]))
    return roll, pitch


def sensor_vector(model, data, name):
    sid = object_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    start = int(model.sensor_adr[sid])
    size = int(model.sensor_dim[sid])
    return data.sensordata[start:start + size].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"请从仓库根目录运行；找不到 {MODEL_PATH}")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH.resolve()))
    data = mujoco.MjData(model)
    if model.nu != 25:
        raise ValueError(f"预期 25 个 actuator，实际为 {model.nu}")
    dt = float(model.opt.timestep)
    if not np.isclose(dt, 0.002):
        raise ValueError(f"预期 timestep=0.002，实际为 {dt}")

    qadr, dadr, aids = build_joint_map(model)
    base_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "LINK_BASE")

    # 读取每个 actuator 的 MJCF 硬限制；教学上再降额到 25%。
    hard_low = model.actuator_ctrlrange[aids, 0].copy()
    hard_high = model.actuator_ctrlrange[aids, 1].copy()
    torque_limit = 0.25 * np.minimum(np.abs(hard_low), np.abs(hard_high))

    # 关节目标保留 0.02 rad 安全边界。
    joint_low = np.empty(25)
    joint_high = np.empty(25)
    for i, name in enumerate(JOINT_NAMES):
        jid = object_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_limited[jid]:
            joint_low[i] = model.jnt_range[jid, 0] + 0.02
            joint_high[i] = model.jnt_range[jid, 1] - 0.02
        else:
            joint_low[i], joint_high[i] = -np.inf, np.inf
    q_goal = np.clip(Q_GOAL, joint_low, joint_high)

    # 从接近站立但不完全相同的姿态开始。
    start_offset = np.zeros(25)
    start_offset[[0, 3, 6, 9]] = [0.03, -0.03, 0.03, -0.03]
    data.qpos[qadr] = np.clip(q_goal + start_offset, joint_low, joint_high)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    q_start = data.qpos[qadr].copy()

    transition_duration = 3.0
    max_target_rate = 1.0       # rad/s
    min_base_height = 0.65      # m
    max_tilt = math.radians(50)
    previous_target = q_start.copy()
    saturation_steps = 0
    total_steps = 0
    failed = False

    def control_step(elapsed):
        nonlocal previous_target, saturation_steps, total_steps, failed
        q = data.qpos[qadr].copy()
        qd = data.qvel[dadr].copy()
        q_des_raw, qd_des = quintic(
            q_start, q_goal, elapsed, transition_duration
        )
        q_des_raw = np.clip(q_des_raw, joint_low, joint_high)

        # 限制相邻控制周期的目标角变化。
        delta_limit = max_target_rate * dt
        q_des = previous_target + np.clip(
            q_des_raw - previous_target, -delta_limit, delta_limit
        )
        previous_target = q_des

        tau_raw = KP * (q_des - q) + KD * (qd_des - qd)
        tau = np.clip(tau_raw, -torque_limit, torque_limit)
        saturation_steps += int(np.any(np.abs(tau_raw) > torque_limit))
        total_steps += 1

        finite = np.isfinite(q).all() and np.isfinite(qd).all()
        base_height = float(data.xpos[base_id, 2])
        roll, pitch = roll_pitch_from_quat(data.xquat[base_id])
        posture_safe = (
            base_height >= min_base_height
            and abs(roll) <= max_tilt
            and abs(pitch) <= max_tilt
        )

        if not finite or not posture_safe:
            # 安全输出：有限的纯速度阻尼，而不是继续追踪站姿。
            safe_kd = np.minimum(KD, 2.0)
            tau = np.clip(-safe_kd * qd, -torque_limit, torque_limit)
            failed = True

        data.ctrl[aids] = tau
        return q, q_des, tau, base_height, roll, pitch

    viewer_context = None
    if args.viewer:
        # 用 importlib 避免在 main() 内重新绑定名称 mujoco。
        from importlib import import_module
        viewer_module = import_module("mujoco.viewer")
        viewer_context = viewer_module.launch_passive(model, data)

    try:
        while data.time < args.duration and not failed:
            q, q_des, tau, height, roll, pitch = control_step(data.time)
            mujoco.mj_step(model, data)

            if total_steps % round(0.5 / dt) == 0:
                left_force = np.linalg.norm(
                    sensor_vector(model, data, "force_left_foot")
                )
                right_force = np.linalg.norm(
                    sensor_vector(model, data, "force_right_foot")
                )
                print(
                    f"t={data.time:5.2f}s height={height:5.3f}m "
                    f"roll={math.degrees(roll):6.2f}deg "
                    f"pitch={math.degrees(pitch):6.2f}deg "
                    f"max_err={np.max(np.abs(q_des-q)):6.3f}rad "
                    f"feet=({left_force:6.1f},{right_force:6.1f})N"
                )

            if viewer_context is not None:
                if not viewer_context.is_running():
                    break
                viewer_context.sync()
                time.sleep(dt)
    finally:
        if viewer_context is not None:
            viewer_context.close()

    saturation_ratio = saturation_steps / max(total_steps, 1)
    print(f"结束时间: {data.time:.3f} s")
    print(f"发生安全终止: {failed}")
    print(f"至少一路力矩饱和的控制周期比例: {100*saturation_ratio:.2f}%")
    print(f"最大实际力矩: {np.max(np.abs(data.ctrl[aids])):.2f} N·m")


if __name__ == "__main__":
    main()