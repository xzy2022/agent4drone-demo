# src2\schemas.py

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

# ==========================================
# 1. 基础参数模型 (Base Parameters)
# ==========================================

class DroneBaseParams(BaseModel):
    """所有针对特定无人机操作的基类"""
    drone_id: str = Field(
        ..., 
        description="The unique identifier (UUID) of the drone to control."
    )

class Waypoint(BaseModel):
    """三维坐标点定义"""
    x: float = Field(..., description="X coordinate in meters")
    y: float = Field(..., description="Y coordinate in meters")
    z: float = Field(..., description="Z coordinate (altitude) in meters")

# ==========================================
# 2. 动作控制契约 (Action Contracts)
# 对应 UAVAPIClient 的原子操作
# ==========================================

class TakeOffParams(DroneBaseParams):
    altitude: float = Field(
        10.0, 
        gt=0,          # 高度必须大于 0
        le=200,        # 目前先设 200
        description="Target altitude in meters. Must be positive."
    )

class LandParams(DroneBaseParams):
    pass

class ReturnHomeParams(DroneBaseParams):
    pass

class SetHomeParams(DroneBaseParams):
    pass

class MoveToParams(DroneBaseParams):
    x: float = Field(..., description="Target X coordinate")
    y: float = Field(..., description="Target Y coordinate")
    z: float = Field(..., description="Target Z coordinate")

class MoveTowardsParams(DroneBaseParams):
    distance: float = Field(..., description="Distance to move in meters。我当前猜测是让无人机按照自己的方向水平移动")
    heading: Optional[float] = Field(
        None, 
        description="Heading direction in degrees (0-360). If None, uses current heading."
    )
    dz: Optional[float] = Field(
        None, 
        description="Vertical component (altitude change) in meters."
    )

class MoveAlongPathParams(DroneBaseParams):
    waypoints: List[Waypoint] = Field(
        ..., 
        description="Ordered list of waypoints for the drone to follow."
    )

class ChangeAltitudeParams(DroneBaseParams):
    altitude: float = Field(..., description="New target absolute altitude")

class HoverParams(DroneBaseParams):
    duration: Optional[float] = Field(
        None, 
        description="Time to hover in seconds. If None, hovers indefinitely until next command."
    )

class RotateParams(DroneBaseParams):
    heading: float = Field(
        ..., 
        ge=0, le=360,
        description="Target absolute heading in degrees (0=North, 90=East)."
    )

class ChargeParams(DroneBaseParams):
    charge_amount: float = Field(
        ..., 
        ge=0, le=100,
        description="Target battery percentage to charge to (0-100)."
    )

class TakePhotoParams(DroneBaseParams):
    pass

class CalibrateParams(DroneBaseParams):
    pass

# --- 通信类 ---

class SendMessageParams(DroneBaseParams):
    target_drone_id: str = Field(..., description="ID of the recipient drone")
    message: str = Field(..., description="Content of the message")

class BroadcastParams(DroneBaseParams):
    message: str = Field(..., description="Content of the message to broadcast to all drones")

# --- 感知/查询类 ---
# 这边感知到底有哪些是有权限访问的还是未知
# ???

# class GetDroneStatusParams(DroneBaseParams):
#     pass

# class GetNearbyEntitiesParams(DroneBaseParams):
#     pass

# class CheckCollisionParams(BaseModel):
#     point: Waypoint
#     safety_margin: float = Field(0.0, description="Safety buffer distance in meters")

# ==========================================
# 3. LLM 输出结构 (LLM Output Contracts)
# 用于标准化 Agent 的结构化输出
# ==========================================

class MissionStep(BaseModel):
    """单步指令结构"""
    func: str = Field(
        ..., 
        description="The exact name of the tool function to call (e.g., 'take_off', 'move_to')."
    )
    # 使用 Dict 以保持灵活性，但运行时可以通过上面的 Params 类进行校验
    params: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Parameters for the function matching the specific tool schema."
    )

class MissionPlan(BaseModel):
    """
    任务规划输出结构
    用于 NLP Commander 将自然语言转换为指令序列
    """
    mission_steps: List[MissionStep] = Field(
        ..., 
        description="An ordered list of steps to execute the user's intent."
    )

class AgentThought(BaseModel):
    """
    包含思考过程的单一响应结构
    """
    thought: str = Field(..., description="The reasoning process behind the action.")
    action: Optional[str] = Field(None, description="The tool name to execute, if any.")
    action_input: Optional[Dict[str, Any]] = Field(None, description="Parameters for the tool.")
    final_answer: Optional[str] = Field(None, description="The final response to the user if no action is needed.")