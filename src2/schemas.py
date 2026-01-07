# src2/schemas.py
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

# ==========================================
# 1. 基础数据模型 (Basic Models)
# ==========================================

class Position(BaseModel):
    """三维坐标位置"""
    x: float = Field(..., description="X 坐标 (米)")
    y: float = Field(..., description="Y 坐标 (米)")
    z: float = Field(..., description="Z 坐标/高度 (米)")

class Vertex(BaseModel):
    """二维顶点 (用于多边形障碍物)"""
    x: float
    y: float

class DroneBaseParams(BaseModel):
    """所有针对特定无人机操作的基类"""
    drone_id: str = Field(
        ..., 
        description="The unique identifier (UUID) of the drone to control."
    )

# ==========================================
# 2. 实体状态模型 (Entity Status Models)
# 基于真实 Probe 数据构建
# ==========================================

class DroneStatus(BaseModel):
    """无人机详细状态 (对应 get_drone_status 返回值)"""
    id: str = Field(..., description="无人机唯一 ID")
    name: str
    model: str
    status: str = Field(..., description="当前状态 (idle, flying, etc.)")
    position: Position
    heading: float = Field(..., description="机头朝向 (0-360度)")
    battery_level: float = Field(..., description="剩余电量百分比 (0-100)")
    max_speed: Optional[float] = None
    max_altitude: Optional[float] = None
    home_position: Optional[Position] = None
    perceived_radius: Optional[float] = None

class ObstacleType(str, Enum):
    """障碍物类型枚举"""
    # ??? 这里各个障碍物的颜色对应没有问题，但是属性是AI猜测的
    POINT = "point"       # 点障碍物 (黑色)
    CYLINDER = "cylinder" # 圆柱障碍物 (棕色) - 使用 radius, height
    ELLIPSE = "ellipse"   # 椭圆柱障碍物 (灰蓝色) - 使用 width, length, height
    POLYGON = "polygon"   # 多面体障碍物 (灰色) - 使用 vertices, height

class Obstacle(BaseModel):
    """障碍物定义 (兼容 Point, Cylinder, Ellipse, Polygon)"""
    # ??? 关于不同的障碍物有什么字段，并没有详细检验，先这样写
    id: str
    name: str
    type: str = Field(..., description="障碍物类型") # 建议对应 ObstacleType
    position: Position
    height: Optional[float] = Field(None, description="障碍物高度")
    
    # --- 不同形状的特有字段 ---
    radius: Optional[float] = Field(None, description="半径 (用于 Cylinder)")
    width: Optional[float] = Field(None, description="宽度/长轴 (用于 Ellipse)")
    length: Optional[float] = Field(None, description="长度/短轴 (用于 Ellipse)")
    vertices: Optional[List[Vertex]] = Field(None, description="底面顶点列表 (用于 Polygon)")

class Target(BaseModel):
    """目标点定义"""
    id: str
    position: Position
    status: Optional[str] = "unvisited"
    class Config:
        extra = "allow"

class NearbyResponse(BaseModel):
    """感知接口 get_nearby_entities 返回的数据结构"""
    drones: List[Dict[str, Any]] = [] # 简化处理其他无人机
    targets: List[Target] = []
    obstacles: List[Obstacle] = []

# ==========================================
# 3. 动作控制契约 (Action Contracts)
# 采用 Version B 的校验逻辑 + Version A 的可用性确认
# ==========================================

class TakeOffParams(DroneBaseParams):
    altitude: float = Field(
        10.0, 
        gt=0, 
        description="Target altitude in meters. Must be positive."
    )

class LandParams(DroneBaseParams):
    pass

class ReturnHomeParams(DroneBaseParams):
    pass

class SetHomeParams(DroneBaseParams):
    pass

class MoveToParams(DroneBaseParams):
    x: float = Field(..., description="目标 X 坐标")
    y: float = Field(..., description="目标 Y 坐标")
    z: float = Field(..., description="目标 Z 坐标")

class MoveTowardsParams(DroneBaseParams):
    distance: float = Field(..., description="Distance to move in meters (forward relative to heading).")
    heading: Optional[float] = Field(
        None, 
        description="Heading direction in degrees (0-360). If None, uses current heading."
    )
    dz: Optional[float] = Field(
        None, 
        description="Vertical component (altitude change) in meters."
    )

class ChangeAltitudeParams(DroneBaseParams):
    altitude: float = Field(..., description="New target absolute altitude")

class HoverParams(DroneBaseParams):
    duration: Optional[float] = Field(
        None, 
        description="Time to hover in seconds. If None, hovers indefinitely."
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

class GetNearbyEntitiesParams(DroneBaseParams):
    """获取周边感知信息"""
    pass

# ==========================================
# 4. LLM 输出结构 (LLM Output Contracts)
# 融合两种模式：既支持指令流，也支持 ReAct 思考流
# ==========================================

class AgentAction(BaseModel):
    """单步动作定义"""
    func: str = Field(..., description="The tool function name (e.g., 'take_off')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the function")
    thought: Optional[str] = Field(None, description="Reasoning behind this action")

class MissionPlan(BaseModel):
    """
    [模式 A: 规划器]
    适用于 NLP Commander，一次性生成多步计划
    """
    mission_steps: List[AgentAction] = Field(..., description="Ordered list of steps to execute.")

class AgentThought(BaseModel):
    """
    [模式 B: ReAct Agent]
    适用于 LangChain Agent，包含思考、行动或最终回复
    """
    thought: str = Field(..., description="The reasoning process.")
    action: Optional[str] = Field(None, description="Tool name to execute.")
    action_input: Optional[Dict[str, Any]] = Field(None, description="Tool parameters.")
    final_answer: Optional[str] = Field(None, description="Final response to user if task is done.")