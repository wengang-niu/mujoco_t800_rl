"""T800 机器人 RL 站立实验的共享工具。

本模块集中存放模型加载、按名称寻址、名义站立姿态、PD 增益、
以及观测/奖励需要的物理量读取函数，供 02~06 各脚本复用。

注意：MuJoCo 4.0 用相对路径加载模型会非确定性失败（已实测），
所以 MODEL_PATH 一律使用绝对路径。
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

# 模型绝对路径（相对路径加载在 MuJoCo 4.0 下会偶发报错）
MODEL_PATH = (Path(__file__).resolve().parent / "models" / "t800" / "t800.xml").as_posix()

# 25 个关节（顺序即 SDK/模型顺序，前 12 为双腿）
JOINT_NAMES = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW",
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
    "J23_HEAD_PITCH", "J24_HEAD_YAW",
]
LEG_NAMES = JOINT_NAMES[:12]   # 双腿 12 个关节（RL 的动作空间）
ARM_NAMES = JOINT_NAMES[13:23]  # 双臂 10 个关节
HEAD_NAMES = JOINT_NAMES[23:25]  # 头 2 个关节

# 名义站立姿态：默认 qpos0 全 0（直腿站立），基座高度 1.03m
NOMINAL_Q = np.zeros(25, dtype=np.float64)

# 各关节 PD 增益（25 维，顺序同 JOINT_NAMES）。
# 腿部增益刻意调高：内环刚度足够后，策略用很小的目标角偏移就能获得
# 接近满额的力矩权限（踝 Kp=400 × 0.12rad ≈ 48N·m，远大于原先 50×0.12=6N·m），
# 这是 RL 能学会主动平衡的前提（浮动双足靠固定 PD 站不住，必须靠策略主动反馈）。
KP = np.array([
    500, 300, 150, 500, 400, 200,   # 左腿 hip_pitch/roll/yaw knee ankle_pitch/roll
    500, 300, 150, 500, 400, 200,   # 右腿
    50,                            # 腰 yaw
    30, 30, 20, 25, 15,            # 左臂
    30, 30, 20, 25, 15,            # 右臂
    15, 15,                        # 头
], dtype=np.float64)
KD = np.array([
    20, 12, 8, 20, 15, 8,
    20, 12, 8, 20, 15, 8,
    2,
    1.5, 1.5, 1.0, 1.2, 0.8,
    1.5, 1.5, 1.0, 1.2, 0.8,
    0.8, 0.8,
], dtype=np.float64)

# RL 动作（腿 12 维）对应的目标角最大偏移量（弧度）
ACTION_SCALE = np.array([
    0.15, 0.10, 0.10, 0.20, 0.12, 0.10,
    0.15, 0.10, 0.10, 0.20, 0.12, 0.10,
], dtype=np.float64)

TARGET_HEIGHT = 1.016   # 站立名义基座高度（m）：脚底刚好着地且完全承重（实测）
TIMESTEP = 0.002       # 物理步长（s）
FRAME_SKIP = 5         # 策略每 5 个物理步决策一次（策略 100Hz）
MAX_STEPS = 1000       # 每回合最大策略步数（约 10s）


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """加载模型并创建运行时数据（绝对路径）。"""
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


# ---- 按名称寻址工具（自由关节在最前，需按名称而非切片定位） ----

def joint_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def qpos_adr(model: mujoco.MjModel, name: str) -> int:
    """关节在 qpos 中的起始地址。"""
    return model.jnt_qposadr[joint_id(model, name)]


def dof_adr(model: mujoco.MjModel, name: str) -> int:
    """关节在 qvel/ctrl 自由度中的地址。"""
    return model.jnt_dofadr[joint_id(model, name)]


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    """按关节名得到对应电机索引（电机命名规则 motor_<关节名>）。"""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")


def joint_qpos(model: mujoco.MjModel, data: mujoco.MjData, names: list[str]) -> np.ndarray:
    """按名称读取一组关节的当前 qpos。"""
    return np.array([data.qpos[qpos_adr(model, n)] for n in names])


def joint_qvel(model: mujoco.MjModel, data: mujoco.MjData, names: list[str]) -> np.ndarray:
    """按名称读取一组关节的当前 qvel。"""
    return np.array([data.qvel[dof_adr(model, n)] for n in names])


def torque_limit(model: mujoco.MjModel) -> np.ndarray:
    """每关节力矩限幅（取 ctrlrange 的 50% 作为安全裕量）。

    注意：踝关节不能简单取 0.5×ctrlrange（=80N·m）。保持脚底不离地的
    有效踝力矩上限 ≈ 足半长 5.7cm × 体重 833N ≈ 47N·m，且实测超过 ~35N·m
    就会把轻脚「发射」离地（双足接触力归零、基座反而升高），导致直接失衡。
    所以踝 pitch/roll 额外收紧到 35N·m。
    """
    lim = np.zeros(model.nu)
    for i, n in enumerate(JOINT_NAMES):
        aid = actuator_id(model, n)
        lim[i] = 0.5 * model.actuator_ctrlrange[aid][1]
    for i in (4, 5, 10, 11):          # 左右踝 pitch/roll
        lim[i] = min(lim[i], 35.0)
    return lim


# ---- 物理量读取（观测/奖励用） ----

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """四元数 (w,x,y,z) -> 3x3 旋转矩阵（body->world）。"""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def projected_gravity(data: mujoco.MjData) -> np.ndarray:
    """世界重力 [0,0,-1] 投影到基座坐标系（直立时应约等于 [0,0,-1]）。"""
    R = quat_to_rot(data.qpos[3:7])       # body->world
    return R.T @ np.array([0.0, 0.0, -1.0])  # world->body


def base_roll_pitch(data: mujoco.MjData) -> tuple[float, float]:
    """由投影重力解出基座 roll/pitch（直立时均≈0）。"""
    g = projected_gravity(data)
    roll = np.arctan2(-g[1], -g[2])
    pitch = np.arctan2(g[0], -g[2])
    return float(roll), float(pitch)


def foot_contacts(data: mujoco.MjData) -> np.ndarray:
    """左右足底接触强度（力传感器模长，落在地上时明显>0）。"""
    fl = np.linalg.norm(data.sensor("force_left_foot").data)
    fr = np.linalg.norm(data.sensor("force_right_foot").data)
    return np.array([fl, fr])


def base_height(data: mujoco.MjData) -> float:
    """基座离地高度（qpos 前 3 个分量的 z）。"""
    return float(data.qpos[2])


# ---- 站立复位与 PD 控制 ----

def reset_to_standing(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_height: float = TARGET_HEIGHT,
    q: np.ndarray | None = None,
) -> None:
    """复位到站立姿态：基座放在给定高度，关节设为 q（默认全 0 直腿）。"""
    mujoco.mj_resetData(model, data)
    data.qpos[2] = base_height
    if q is not None:
        for i, n in enumerate(JOINT_NAMES):
            data.qpos[qpos_adr(model, n)] = q[i]
    mujoco.mj_forward(model, data)


def pd_torque(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q_des: np.ndarray,
    qd_des: float = 0.0,
) -> np.ndarray:
    """全身 PD：tau = Kp(q_des-q) + Kd(qd_des-qd)，按力矩限幅（25 维）。"""
    q = joint_qpos(model, data, JOINT_NAMES)
    qd = joint_qvel(model, data, JOINT_NAMES)
    tau = KP * (q_des - q) + KD * (qd_des - qd)
    return np.clip(tau, -torque_limit(model), torque_limit(model))


def apply_torque(model: mujoco.MjModel, data: mujoco.MjData, tau: np.ndarray) -> None:
    """把 25 维力矩写入对应电机 ctrl。"""
    for i, n in enumerate(JOINT_NAMES):
        data.ctrl[actuator_id(model, n)] = tau[i]


# ---- 高频循环的向量化接口（预计算地址，避免每步按名称寻址）----
# 训练内环每个物理子步都要算 PD，若用上面的按名称寻址会做 ~100 次
# mj_name2id 字符串查找，慢到不可用；这里预计算一次，之后全部 numpy 花式索引。

def build_addr(model: mujoco.MjModel) -> SimpleNamespace:
    """一次性预计算 25 个关节的 qpos/dof/actuator 地址、关节 id 与增益/限幅。"""
    jid = np.array([joint_id(model, n) for n in JOINT_NAMES], dtype=int)
    qpos = np.array([model.jnt_qposadr[j] for j in jid], dtype=int)
    dof = np.array([model.jnt_dofadr[j] for j in jid], dtype=int)
    act = np.array([actuator_id(model, n) for n in JOINT_NAMES], dtype=int)
    # 足底力传感器的数据地址（3 维力向量在 sensordata 中的起始位置）
    s_l = model.sensor("force_left_foot")
    s_r = model.sensor("force_right_foot")
    return SimpleNamespace(
        jid=jid, qpos=qpos, dof=dof, act=act,
        leg=np.arange(12, dtype=int),
        kp=KP.copy(), kd=KD.copy(), tau_lim=torque_limit(model),
        sfl_adr=int(s_l.adr.item()), sfl_dim=int(s_l.dim.item()),
        sfr_adr=int(s_r.adr.item()), sfr_dim=int(s_r.dim.item()),
    )


def pd_torque_fast(data: mujoco.MjData, addr: SimpleNamespace, q_des: np.ndarray,
                   qd_des: float = 0.0) -> np.ndarray:
    """向量化 PD：tau = kp*(q_des-q) + kd*(qd_des-qd)，按力矩限幅（25 维）。"""
    q = data.qpos[addr.qpos]
    qd = data.qvel[addr.dof]
    tau = addr.kp * (q_des - q) + addr.kd * (qd_des - qd)
    return np.clip(tau, -addr.tau_lim, addr.tau_lim)


def foot_contacts_fast(data: mujoco.MjData, addr: SimpleNamespace) -> np.ndarray:
    """左右足底接触强度（力传感器模长），用预计算地址避免按名称寻址。"""
    fl = np.linalg.norm(data.sensordata[addr.sfl_adr:addr.sfl_adr + addr.sfl_dim])
    fr = np.linalg.norm(data.sensordata[addr.sfr_adr:addr.sfr_adr + addr.sfr_dim])
    return np.array([fl, fr])
