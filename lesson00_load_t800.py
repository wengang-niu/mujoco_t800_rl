from pathlib import Path

import mujoco
import numpy as np


# 本程序要求从仓库根目录运行。          # 这样相对路径 models/t800/t800.xml 才能被正确解析。
MODEL_PATH = Path("models/t800/t800.xml").resolve()  # 解析为绝对路径，便于报错时直接定位模型文件。


def main() -> None:
    if not MODEL_PATH.is_file():  # 先检查模型文件是否存在，避免在加载 XML 时得到不直观的底层报错。
        raise FileNotFoundError(
            f"找不到模型：{MODEL_PATH}\n"  # 把绝对路径带出来，方便快速确认路径是否写错。
            "请确认当前目录是 mujoco_t800_rl 仓库根目录。"
        )

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))  # 读取 XML 并构建 MuJoCo 模型对象。
    data = mujoco.MjData(model)  # 为该模型分配一份可变仿真状态，包括 qpos/qvel/ctrl 等。

    print(f"MuJoCo version : {mujoco.__version__}")  # MuJoCo 版本，用于检查运行环境。
    print(f"model path     : {MODEL_PATH}")          # 加载的机器人 XML 模型绝对路径。
    print(f"nq             : {model.nq}")            # 广义位置数；自由关节占 7 维（位置 3 + 四元数 4）。
    print(f"nv             : {model.nv}")            # 广义速度数；自由关节占 6 维（线速度 3 + 角速度 3）。
    print(f"nu             : {model.nu}")            # 执行器控制输入数，即 ctrl 的长度。
    print(f"nsensor        : {model.nsensor}")       # XML 中定义的传感器数量。
    print(f"timestep       : {model.opt.timestep:.6f} s")  # 每次 mj_step 推进的时间；.6f 表示保留 6 位小数。
    print(f"qpos shape     : {data.qpos.shape}")     # 广义位置数组形状，长度通常等于 nq。
    print(f"qvel shape     : {data.qvel.shape}")     # 广义速度数组形状，长度通常等于 nv。
    print(f"ctrl shape     : {data.ctrl.shape}")     # 执行器控制数组形状，长度通常等于 nu。

    # XML 中 LINK_BASE 的初始位置和单位四元数已写入 qpos0。
    # mj_resetData 会恢复到该默认状态。
    mujoco.mj_resetData(model, data)  # 把 data 恢复到 XML 里定义的默认初始状态。
    mujoco.mj_forward(model, data)  # 在不积分时间的前提下刷新派生量，如站姿对应的几何与动力学量。

    simulated_seconds = 1.0  # 目标仿真时长，示例里先跑 1 秒用于观察模型是否能正常推进。
    steps = round(simulated_seconds / model.opt.timestep)  # 用“总时长 / 步长”估算需要调用 mj_step 的次数。

    # 暂时不施加任何执行器控制。自由基座机器人会受重力影响并跌倒；
    # 这是正确的物理结果，不是 MuJoCo 出错。
    data.ctrl[:] = 0.0
    for _ in range(steps):  # 逐步推进仿真，而不是一次性跑完，便于中途插入检查逻辑。
        mujoco.mj_step(model, data)  # 前进一步：积分状态、更新时间，并计算新的动力学结果。
        if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):  # 数值发散时尽早停止。
            raise FloatingPointError("仿真状态出现 NaN 或 Inf")

    print(f"simulated time : {data.time:.3f} s")  # 实际推进到的仿真时间，正常应接近 1.0 秒。
    print(f"base xyz       : {data.qpos[:3]}")  # 自由基座平移坐标，通常对应机器人底座的世界系位置。
    print("headless simulation finished")  # 无图形界面模式下完成一次最小仿真闭环。


if __name__ == "__main__":
    main()