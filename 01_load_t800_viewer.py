from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = Path("models/t800/t800.xml").resolve()  # 模型 XML 的绝对路径


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))  # 加载模型（运动学/动力学参数）
    data = mujoco.MjData(model)  # 运行时状态数据（qpos/qvel/受力等）
    mujoco.mj_resetData(model, data)  # 将状态复位到初始位姿
    mujoco.mj_forward(model, data)  # 执行一次前向运动学/动力学计算

    print(f"MuJoCo version : {mujoco.__version__}")
    print(f"model path     : {MODEL_PATH}")
    print(f"nq / nv / nu   : {model.nq} / {model.nv} / {model.nu}")  # 广义位置/速度/控制量维度
    print(f"timestep       : {model.opt.timestep:.6f} s")  # 仿真步长

    with mujoco.viewer.launch_passive(model, data) as viewer:  # 被动模式 viewer（只同步显示）
        while viewer.is_running() and data.time < 10.0:  # 窗口未关闭且未满 10 秒
            mujoco.mj_step(model, data)  # 推进一个仿真步（无控制输入）

            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError("仿真状态出现 NaN 或 Inf")  # 数值发散保护

            viewer.sync()  # 刷新 viewer 画面


    print(f"simulated time : {data.time:.3f} s")
    print(f"base xyz       : {data.qpos[:3]}")  # 底座位置（前 3 个 qpos 分量）
    print("viewer simulation finished")


if __name__ == "__main__":
    main()
