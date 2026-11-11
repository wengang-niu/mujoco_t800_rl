from pathlib import Path  # 用面向对象的方式处理模型文件路径。
import time  # 用于计算仿真步的实际耗时，并控制可视化播放速度。

import mujoco  # MuJoCo 的模型加载、状态管理和物理仿真接口。
import mujoco.viewer  # MuJoCo 官方交互式三维查看器。


MODEL_PATH = Path("models/t800/t800.xml").resolve()  # 将模型相对路径转换为绝对路径。


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))  # 从 XML 加载固定不变的模型结构和参数。
    data = mujoco.MjData(model)  # 创建随仿真变化的数据，如时间、位置、速度和控制量。
    mujoco.mj_resetData(model, data)  # 将状态恢复到 XML 定义的默认初始状态。
    mujoco.mj_forward(model, data)  # 根据初始状态计算几何位置和动力学派生量，但不推进时间。

    # passive 模式下，程序负责推进仿真并调用 sync；退出 with 时 viewer 会自动关闭。
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and data.time < 10.0:  # 窗口未关闭且仿真时间未达到 10 秒时继续。
            wall_start = time.time()  # 记录本轮现实时间的起点，用于计算该轮耗时。
            data.ctrl[:] = 0.0  # 控制输入全部置零，机器人只受重力、接触力等物理作用。
            mujoco.mj_step(model, data)  # 将物理仿真向前推进一个 timestep。
            viewer.sync()  # 将最新状态同步到画面，并处理窗口的鼠标、键盘等事件。

            # 让一个仿真步占用约一个 timestep 的现实时间，从而接近实时播放；训练时不要 sleep。
            remaining = model.opt.timestep - (time.time() - wall_start)  # 等待时间 = 仿真步长 - 本轮耗时。
            if remaining > 0:  # 计算比目标步长快时才等待；如果已经超时，则直接进入下一轮。
                time.sleep(remaining)  # 暂停剩余时间，避免画面播放速度快于真实时间。


if __name__ == "__main__":  # 直接运行本文件时执行；被其他模块导入时不自动启动查看器。
    main()