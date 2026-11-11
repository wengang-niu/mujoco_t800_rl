from dataclasses import dataclass
from pathlib import Path
import math
import sys

import mujoco
import numpy as np


@dataclass
class PIDController:
    kp: float
    kd: float
    ki: float = 0.0
    integral_limit: float = 0.5
    torque_limit: float = 8.0
    integral: float = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def compute(
        self,
        q: float,
        qd: float,
        q_des: float,
        qd_des: float,
        dt: float,
        tau_ff: float = 0.0,
    ) -> float:
        if not np.isfinite([q, qd, q_des, qd_des, dt, tau_ff]).all():
            self.reset()
            return 0.0
        if dt <= 0.0:
            raise ValueError("dt 必须大于 0")

        error = q_des - q
        velocity_error = qd_des - qd

        # 先构造候选积分，再判断是否会加重输出饱和。
        candidate_integral = np.clip(
            self.integral + error * dt,
            -self.integral_limit,
            self.integral_limit,
        )
        candidate_tau = (
            self.kp * error
            + self.kd * velocity_error
            + self.ki * candidate_integral
            + tau_ff
        )

        saturated_high = candidate_tau > self.torque_limit
        saturated_low = candidate_tau < -self.torque_limit
        worsens_saturation = (
            (saturated_high and error > 0.0)
            or (saturated_low and error < 0.0)
        )
        if not worsens_saturation:
            self.integral = float(candidate_integral)

        tau = (
            self.kp * error
            + self.kd * velocity_error
            + self.ki * self.integral
            + tau_ff
        )
        return float(np.clip(tau, -self.torque_limit, self.torque_limit))


def load_model(xml_path: Path):
    model = mujoco.MjModel.from_xml_path(str(xml_path.resolve()))
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
    actuator_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hinge_motor"
    )
    if joint_id < 0 or actuator_id < 0:
        raise RuntimeError("模型中缺少 hinge 或 hinge_motor")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    dof_adr = int(model.jnt_dofadr[joint_id])
    return model, data, qpos_adr, dof_adr, actuator_id


def run_case(name, controller, tau_ff=0.0, duration=4.0):
    model, data, qadr, dadr, actuator_id = load_model(Path("single_hinge_pid.xml"))
    dt = float(model.opt.timestep)
    q_des = math.radians(35.0)
    controller.reset()

    errors = []
    torques = []
    for _ in range(round(duration / dt)):
        q = float(data.qpos[qadr])
        qd = float(data.qvel[dadr])
        tau = controller.compute(q, qd, q_des, 0.0, dt, tau_ff)
        data.ctrl[actuator_id] = tau
        mujoco.mj_step(model, data)

        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise FloatingPointError(f"{name}: 仿真出现 NaN/Inf")
        errors.append(q_des - q)
        torques.append(tau)

    errors = np.asarray(errors)
    torques = np.asarray(torques)
    saturated = np.mean(np.isclose(np.abs(torques), controller.torque_limit))
    print(
        f"{name:12s} | final_error={math.degrees(errors[-1]):7.3f} deg "
        f"| max_error={math.degrees(np.max(np.abs(errors))):7.3f} deg "
        f"| saturation={100*saturated:5.1f}% "
        f"| integral={controller.integral:7.4f}"
    )


def main():
    if not Path("./single_hinge_pid.xml").exists():
        sys.exit("请先创建 single_hinge_pid.xml")

    # 同一目标、同一模型，仅改变控制器。
    run_case("P", PIDController(kp=8.0, kd=0.0))
    run_case("PD", PIDController(kp=8.0, kd=1.2))
    run_case("PID", PIDController(kp=8.0, kd=1.2, ki=2.0))

    # 约 3 N·m 的前馈只是教学估值：实际重力补偿应由动力学模型计算。
    run_case(
        "PD+feedforward",
        PIDController(kp=8.0, kd=1.2),
        tau_ff=3.0,
    )


if __name__ == "__main__":
    main()