"""
Unitree G1 人形机器人模块 (Unitree G1 Humanoid Robot Module)
====================================================

本模块实现了 Unitree G1 人形机器人的控制。

功能:
    - 通过 DDS 进行低级电机控制
    - 支持仿真和真机模式
    - IMU 状态读取
    - 遥控器支持
    - 逼近运动学 (IK) 支持
"""

import logging
import struct
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.envs.factory import make_env
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.unitree_g1.g1_utils import G1_29_JointArmIndex, G1_29_JointIndex
from lerobot.robots.unitree_g1.robot_kinematic_processor import G1_29_ArmIK

from ..robot import Robot
from .config_unitree_g1 import UnitreeG1Config

logger = logging.getLogger(__name__)

# DDS 主题名称遵循 Unitree SDK 命名规范
# ruff: noqa: N816
kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowState = "rt/lowstate"


@dataclass
class MotorState:
    q: float | None = None  # 位置
    dq: float | None = None  # 速度
    tau_est: float | None = None  # 估计力矩
    temperature: float | None = None  # 电机温度


@dataclass
class IMUState:
    quaternion: np.ndarray | None = None  # [w, x, y, z]
    gyroscope: np.ndarray | None = None  # [x, y, z] 角速度 (rad/s)
    accelerometer: np.ndarray | None = None  # [x, y, z] 线加速度 (m/s²)
    rpy: np.ndarray | None = None  # [滚转, 俱仰, 偏航] (rad)
    temperature: float | None = None  # IMU 温度


# g1 观测类
@dataclass
class G1_29_LowState:  # noqa: N801
    motor_state: list[MotorState] = field(default_factory=lambda: [MotorState() for _ in G1_29_JointIndex])
    imu_state: IMUState = field(default_factory=IMUState)
    wireless_remote: Any = None  # 原始无线遥控器数据
    mode_machine: int = 0  # 机器人模式


class UnitreeG1(Robot):
    config_class = UnitreeG1Config
    name = "unitree_g1"

    # unitree 遥控器
    class RemoteController:
        def __init__(self):
            self.lx = 0
            self.ly = 0
            self.rx = 0
            self.ry = 0
            self.button = [0] * 16

        def set(self, data):
            # wireless_remote
            keys = struct.unpack("H", data[2:4])[0]
            for i in range(16):
                self.button[i] = (keys & (1 << i)) >> i
            self.lx = struct.unpack("f", data[4:8])[0]
            self.rx = struct.unpack("f", data[8:12])[0]
            self.ry = struct.unpack("f", data[12:16])[0]
            self.ly = struct.unpack("f", data[20:24])[0]

    def __init__(self, config: UnitreeG1Config):
        super().__init__(config)

        logger.info("初始化 UnitreeG1...")

        self.config = config
        self.control_dt = config.control_dt

        # 初始化相机配置（基于 ZMQ）- 实际连接在 connect() 中
        self._cameras = make_cameras_from_configs(config.cameras)

        # 根据模式导入通道类
        if config.is_simulation:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
        else:
            from lerobot.robots.unitree_g1.unitree_sdk2_socket import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )

        # 存储以便在 connect() 中使用
        self._ChannelFactoryInitialize = ChannelFactoryInitialize
        self._ChannelPublisher = ChannelPublisher
        self._ChannelSubscriber = ChannelSubscriber

        # 初始化状态变量
        self.sim_env = None
        self._env_wrapper = None
        self._lowstate = None
        self._shutdown_event = threading.Event()
        self.subscribe_thread = None
        self.remote_controller = self.RemoteController()

        self.arm_ik = G1_29_ArmIK()

    def _subscribe_motor_state(self):  # 以 250Hz 轮询机器人状态
        while not self._shutdown_event.is_set():
            start_time = time.time()

            # 如果在仿真模式下则步进仿真
            if self.config.is_simulation and self.sim_env is not None:
                self.sim_env.step()

            msg = self.lowstate_subscriber.Read()
            if msg is not None:
                lowstate = G1_29_LowState()

                # 使用 jointindex 捕获电机状态
                for id in G1_29_JointIndex:
                    lowstate.motor_state[id].q = msg.motor_state[id].q
                    lowstate.motor_state[id].dq = msg.motor_state[id].dq
                    lowstate.motor_state[id].tau_est = msg.motor_state[id].tau_est
                    lowstate.motor_state[id].temperature = msg.motor_state[id].temperature

                # 捕获 IMU 状态
                lowstate.imu_state.quaternion = list(msg.imu_state.quaternion)
                lowstate.imu_state.gyroscope = list(msg.imu_state.gyroscope)
                lowstate.imu_state.accelerometer = list(msg.imu_state.accelerometer)
                lowstate.imu_state.rpy = list(msg.imu_state.rpy)
                lowstate.imu_state.temperature = msg.imu_state.temperature

                # 捕获无线遥控器数据
                lowstate.wireless_remote = msg.wireless_remote

                # 捕获 mode_machine
                lowstate.mode_machine = msg.mode_machine

                self._lowstate = lowstate

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))  # 保持恒定的控制 dt
            time.sleep(sleep_time)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

    def calibrate(self) -> None:  # 机器人已经校准过
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:  # 连接到 DDS
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
            LowCmd_ as hg_LowCmd,
            LowState_ as hg_LowState,
        )
        from unitree_sdk2py.utils.crc import CRC

        # 初始化 DDS 通道和仿真环境
        if self.config.is_simulation:
            self._ChannelFactoryInitialize(0, "lo")
            self._env_wrapper = make_env("lerobot/unitree-g1-mujoco", trust_remote_code=True)
            # 从 dict 结构中提取实际的 gym env
            self.sim_env = self._env_wrapper["hub_env"][0].envs[0]
        else:
            self._ChannelFactoryInitialize(0)

        # 初始化直接电机控制接口
        self.lowcmd_publisher = self._ChannelPublisher(kTopicLowCommand_Debug, hg_LowCmd)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = self._ChannelSubscriber(kTopicLowState, hg_LowState)
        self.lowstate_subscriber.Init()

        # 启动订阅线程以读取机器人状态
        self.subscribe_thread = threading.Thread(target=self._subscribe_motor_state)
        self.subscribe_thread.start()

        # 连接相机
        for cam in self._cameras.values():
            if not cam.is_connected:
                cam.connect()

        logger.info(f"已连接 {len(self._cameras)} 个相机。")

        # 初始化 lowcmd 消息
        self.crc = CRC()
        self.msg = unitree_hg_msg_dds__LowCmd_()
        self.msg.mode_pr = 0

        # 等待第一个状态消息到达
        lowstate = None
        while lowstate is None:
            lowstate = self._lowstate
            if lowstate is None:
                time.sleep(0.01)
            logger.warning("[UnitreeG1] 等待机器人状态...")
        logger.warning("[UnitreeG1] 已连接到机器人。"))
        self.msg.mode_machine = lowstate.mode_machine

        # 使用配置中的统一 kp/kd 初始化所有电机
        self.kp = np.array(self.config.kp, dtype=np.float32)
        self.kd = np.array(self.config.kd, dtype=np.float32)

        for id in G1_29_JointIndex:
            self.msg.motor_cmd[id].mode = 1
            self.msg.motor_cmd[id].kp = self.kp[id.value]
            self.msg.motor_cmd[id].kd = self.kd[id.value]
            self.msg.motor_cmd[id].q = lowstate.motor_state[id.value].q

    def disconnect(self):
        # 通知线程停止并解除任何等待
        self._shutdown_event.set()

        # 等待订阅线程完成
        if self.subscribe_thread is not None:
            self.subscribe_thread.join(timeout=2.0)
            if self.subscribe_thread.is_alive():
                logger.warning("订阅线程未干净地停止")

        # 关闭仿真环境
        if self.config.is_simulation and self.sim_env is not None:
            try:
            # 首先强制终止图像发布子进程以避免长时间等待
                if hasattr(self.sim_env, "simulator") and hasattr(self.sim_env.simulator, "sim_env"):
                    sim_env_inner = self.sim_env.simulator.sim_env
                    if hasattr(sim_env_inner, "image_publish_process"):
                        proc = sim_env_inner.image_publish_process
                        if proc.process and proc.process.is_alive():
                            logger.info("强制终止图像发布子进程..."))
                            proc.stop_event.set()
                            proc.process.terminate()
                            proc.process.join(timeout=1)
                            if proc.process.is_alive():
                                proc.process.kill()
                self.sim_env.close()
            except Exception as e:
                logger.warning(f"关闭 sim_env 时出错: {e}")
            self.sim_env = None
            self._env_wrapper = None

        # 断开相机
        for cam in self._cameras.values():
            cam.disconnect()

    def get_observation(self) -> RobotObservation:
        lowstate = self._lowstate
        if lowstate is None:
            return {}

        obs = {}

        # 电机 - 所有关节的 q, dq, tau
        for motor in G1_29_JointIndex:
            name = motor.name
            idx = motor.value
            obs[f"{name}.q"] = lowstate.motor_state[idx].q
            obs[f"{name}.dq"] = lowstate.motor_state[idx].dq
            obs[f"{name}.tau"] = lowstate.motor_state[idx].tau_est

        # IMU - 陀螺仪
        if lowstate.imu_state.gyroscope:
            obs["imu.gyro.x"] = lowstate.imu_state.gyroscope[0]
            obs["imu.gyro.y"] = lowstate.imu_state.gyroscope[1]
            obs["imu.gyro.z"] = lowstate.imu_state.gyroscope[2]

        # IMU - 加速度计
        if lowstate.imu_state.accelerometer:
            obs["imu.accel.x"] = lowstate.imu_state.accelerometer[0]
            obs["imu.accel.y"] = lowstate.imu_state.accelerometer[1]
            obs["imu.accel.z"] = lowstate.imu_state.accelerometer[2]

        # IMU - 四元数
        if lowstate.imu_state.quaternion:
            obs["imu.quat.w"] = lowstate.imu_state.quaternion[0]
            obs["imu.quat.x"] = lowstate.imu_state.quaternion[1]
            obs["imu.quat.y"] = lowstate.imu_state.quaternion[2]
            obs["imu.quat.z"] = lowstate.imu_state.quaternion[3]

        # IMU - 滚仰航
        if lowstate.imu_state.rpy:
            obs["imu.rpy.roll"] = lowstate.imu_state.rpy[0]
            obs["imu.rpy.pitch"] = lowstate.imu_state.rpy[1]
            obs["imu.rpy.yaw"] = lowstate.imu_state.rpy[2]

        # 控制器 - 解析 wireless_remote 并添加到 obs
        if lowstate.wireless_remote and len(lowstate.wireless_remote) >= 24:
            self.remote_controller.set(lowstate.wireless_remote)
        obs["remote.buttons"] = self.remote_controller.button.copy()
        obs["remote.lx"] = self.remote_controller.lx
        obs["remote.ly"] = self.remote_controller.ly
        obs["remote.rx"] = self.remote_controller.rx
        obs["remote.ry"] = self.remote_controller.ry

        # 相机 - 从 ZMQ 相机读取图像
        for cam_name, cam in self._cameras.items():
            obs[cam_name] = cam.read_latest()

        return obs

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        return self._lowstate is not None

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{G1_29_JointIndex(motor).name}.q": float for motor in G1_29_JointIndex}

    @property
    def cameras(self) -> dict:
        return self._cameras

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    def send_action(self, action: RobotAction) -> RobotAction:
        for motor in G1_29_JointIndex:
            key = f"{motor.name}.q"
            if key in action:
                self.msg.motor_cmd[motor.value].q = action[key]
                self.msg.motor_cmd[motor.value].qd = 0
                self.msg.motor_cmd[motor.value].kp = self.kp[motor.value]
                self.msg.motor_cmd[motor.value].kd = self.kd[motor.value]
                self.msg.motor_cmd[motor.value].tau = 0

        if self.config.gravity_compensation:
            # 从电机命令构建 action_np（机械臂关节索引 15-28，本地索引 0-13）
            action_np = np.zeros(14)
            arm_start_idx = G1_29_JointArmIndex.kLeftShoulderPitch.value  # 15
            for joint in G1_29_JointArmIndex:
                local_idx = joint.value - arm_start_idx
                action_np[local_idx] = self.msg.motor_cmd[joint.value].q
            tau = self.arm_ik.solve_tau(action_np)

            # 将 tau 应用回电机命令
            for joint in G1_29_JointArmIndex:
                local_idx = joint.value - arm_start_idx
                self.msg.motor_cmd[joint.value].tau = tau[local_idx]

        self.msg.crc = self.crc.Crc(self.msg)
        self.lowcmd_publisher.Write(self.msg)
        return action

    def get_gravity_orientation(self, quaternion):  # 从四元数获取重力方向
        """从四元数获取重力方向。"""
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)
        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
        return gravity_orientation

    def reset(
        self,
        control_dt: float | None = None,
        default_positions: list[float] | None = None,
    ) -> None:  # 将机器人移动到默认位置
        if control_dt is None:
            control_dt = self.config.control_dt
        if default_positions is None:
            default_positions = np.array(self.config.default_positions, dtype=np.float32)

        if self.config.is_simulation and self.sim_env is not None:
            self.sim_env.reset()

            for motor in G1_29_JointIndex:
                self.msg.motor_cmd[motor.value].q = default_positions[motor.value]
                self.msg.motor_cmd[motor.value].qd = 0
                self.msg.motor_cmd[motor.value].kp = self.kp[motor.value]
                self.msg.motor_cmd[motor.value].kd = self.kd[motor.value]
                self.msg.motor_cmd[motor.value].tau = 0
            self.msg.crc = self.crc.Crc(self.msg)
            self.lowcmd_publisher.Write(self.msg)
        else:
            total_time = 3.0
            num_steps = int(total_time / control_dt)

            # 获取当前状态
            obs = self.get_observation()

            # 记录当前位置
            init_dof_pos = np.zeros(29, dtype=np.float32)
            for motor in G1_29_JointIndex:
                init_dof_pos[motor.value] = obs[f"{motor.name}.q"]

            # 插值到默认位置
            for step in range(num_steps):
                start_time = time.time()

                alpha = step / num_steps
                action_dict = {}
                for motor in G1_29_JointIndex:
                    target_pos = default_positions[motor.value]
                    interp_pos = init_dof_pos[motor.value] * (1 - alpha) + target_pos * alpha
                    action_dict[f"{motor.name}.q"] = float(interp_pos)

                self.send_action(action_dict)

                # 保持恒定的控制率
                elapsed = time.time() - start_time
                sleep_time = max(0, control_dt - elapsed)
                time.sleep(sleep_time)

        logger.info("已到达默认位置")
