"""
遥操作系统模块 (Teleoperator Module)
====================================

该模块提供了机器人遥操作系统的核心组件。
遥操作器用于人类操作员通过各种输入设备远程控制机器人。

支持的遥操作设备类型：

**机械臂主手 (Leader Arms):**
- KochLeader: Koch机械臂主手
- OmxLeader: OMX机械臂主手  
- SO100Leader/SO101Leader: SO100/SO101机械臂主手
- OpenArmLeader: OpenArm开源机械臂主手
- BiSOLeader: 双臂SO系列主手
- BiOpenArmLeader: 双臂OpenArm主手

**游戏控制器:**
- GamepadTeleop: 游戏手柄遥操作

**键盘控制:**
- KeyboardTeleop: 键盘遥操作
- KeyboardEndEffectorTeleop: 键盘末端执行器控制

**专用设备:**
- HomunculusGlove/HomunculusArm: Homunculus手套/手臂
- UnitreeG1Teleoperator: 宇树G1人形机器人遥操作
- Reachy2Teleoperator: Reachy2机器人遥操作
- PhoneTeleop: 手机遥操作（使用手机传感器）

核心组件：
- Teleoperator: 遥操作器抽象基类
- TeleoperatorConfig: 遥操作器配置基类
- TeleopEvents: 遥操作事件枚举
- make_teleoperator_from_config: 从配置创建遥操作器的工厂函数
"""

# 遥操作器配置基类
from .config import TeleoperatorConfig
# 遥操作器抽象基类
from .teleoperator import Teleoperator
# 遥操作事件枚举和工厂函数
from .utils import TeleopEvents, make_teleoperator_from_config
