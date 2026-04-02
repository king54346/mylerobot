"""
电机驱动模块 (Motors Module)
==========================

提供与各种伺服电机通信的基础设施。

主要组件：
- Motor: 电机配置数据类
- MotorCalibration: 电机标定数据类
- MotorNormMode: 电机归一化模式枚举

支持的电机类型：
- Dynamixel (dynamixel/): ROBOTIS 的 Dynamixel 系列电机
- Feetech (feetech/): Feetech 系列电机
- Damiao (damiao/): 达妙电机
- Robstride (robstride/): Robstride 电机
"""

from .motors_bus import (
    Motor,
    MotorCalibration,
    MotorNormMode,
)
