# src2/tools_registry.py
import json
from typing import List, Callable, Optional
from langchain_core.tools import StructuredTool, BaseTool, Tool

from uav_api_client import UAVAPIClient
from src2.schemas import (
    # Navigation Params
    TakeOffParams,
    LandParams,
    MoveToParams,
    MoveTowardsParams,
    ChangeAltitudeParams,
    RotateParams,
    HoverParams,
    ReturnHomeParams,
    # Perception/Info Params
    GetNearbyEntitiesParams, # 其实就是 DroneBaseParams
    DroneBaseParams,         # 用于只需要 drone_id 的简单查询
    # System/Mission Params
    SetHomeParams,
    CalibrateParams,
    ChargeParams,
    TakePhotoParams
)

class UAVToolRegistry:
    """
    UAV 工具注册表 - 显式参数版
    优点：
    1. 完美适配 LangChain 的参数解包调用机制。
    2. IDE 可以提供完整的代码补全和类型提示。
    3. AI 通过 args_schema 获取元数据，不受函数签名影响。
    """

    def __init__(self, client: UAVAPIClient):
        self.client = client

    def _safe_exec(self, func: Callable, **kwargs) -> str:
        """Helper to execute client methods safely and return JSON string."""
        try:
            result = func(**kwargs)
            return json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    # ==========================================
    # 1. Navigation Tools (Control Logic)
    # ==========================================

    def get_navigation_tools(self) -> List[BaseTool]:
        
        # 显式定义参数，IDE 友好
        def take_off(drone_id: str, altitude: float = 10.0) -> str:
            return self._safe_exec(self.client.take_off, drone_id=drone_id, altitude=altitude)

        def land(drone_id: str) -> str:
            return self._safe_exec(self.client.land, drone_id=drone_id)

        def move_to(drone_id: str, x: float, y: float, z: float) -> str:
            return self._safe_exec(self.client.move_to, drone_id=drone_id, x=x, y=y, z=z)

        def move_towards(drone_id: str, distance: float, heading: Optional[float] = None, dz: Optional[float] = None) -> str:
            return self._safe_exec(self.client.move_towards, drone_id=drone_id, distance=distance, heading=heading, dz=dz)

        def change_altitude(drone_id: str, altitude: float) -> str:
            return self._safe_exec(self.client.change_altitude, drone_id=drone_id, altitude=altitude)

        def rotate(drone_id: str, heading: float) -> str:
            return self._safe_exec(self.client.rotate, drone_id=drone_id, heading=heading)

        def hover(drone_id: str, duration: Optional[float] = None) -> str:
            return self._safe_exec(self.client.hover, drone_id=drone_id, duration=duration)

        def return_home(drone_id: str) -> str:
            return self._safe_exec(self.client.return_home, drone_id=drone_id)

        return [
            StructuredTool.from_function(
                func=take_off,
                name="take_off",
                description="Command a drone to take off to a specific altitude.",
                args_schema=TakeOffParams
            ),
            StructuredTool.from_function(
                func=land,
                name="land",
                description="Command a drone to land at its current position.",
                args_schema=LandParams
            ),
            StructuredTool.from_function(
                func=move_to,
                name="move_to",
                description="Move a drone to specific absolute coordinates (x, y, z).",
                args_schema=MoveToParams
            ),
            StructuredTool.from_function(
                func=move_towards,
                name="move_towards",
                description="Move a drone a specific distance relative to its position.",
                args_schema=MoveTowardsParams
            ),
            StructuredTool.from_function(
                func=change_altitude,
                name="change_altitude",
                description="Change a drone's absolute altitude.",
                args_schema=ChangeAltitudeParams
            ),
            StructuredTool.from_function(
                func=rotate,
                name="rotate",
                description="Rotate a drone to a specific heading (0-360).",
                args_schema=RotateParams
            ),
            StructuredTool.from_function(
                func=hover,
                name="hover",
                description="Command a drone to hover in place.",
                args_schema=HoverParams
            ),
            StructuredTool.from_function(
                func=return_home,
                name="return_home",
                description="Command a drone to return to its home position.",
                args_schema=ReturnHomeParams
            ),
        ]

    # ==========================================
    # 2. Perception Tools (Read-Only)
    # ==========================================

    def get_perception_tools(self) -> List[BaseTool]:

        def get_drone_status(drone_id: str) -> str:
            return self._safe_exec(self.client.get_drone_status, drone_id=drone_id)

        def get_nearby_entities(drone_id: str) -> str:
            return self._safe_exec(self.client.get_nearby_entities, drone_id=drone_id)

        def list_drones() -> str:
            return self._safe_exec(self.client.list_drones)

        def get_weather() -> str:
            return self._safe_exec(self.client.get_weather)

        return [
            StructuredTool.from_function(
                func=get_drone_status,
                name="get_drone_status",
                description="Get detailed status (position, battery, state) of a specific drone.",
                args_schema=DroneBaseParams
            ),
            StructuredTool.from_function(
                func=get_nearby_entities,
                name="get_nearby_entities",
                description="Get entities (drones, targets, obstacles) within the drone's perception radius.",
                args_schema=GetNearbyEntitiesParams
            ),
            Tool(
                name="list_drones",
                # 注意：对于没有参数的 Tool，LangChain 有时会传一个空字符串作为 tool_input
                # 我们用 lambda 忽略它
                func=lambda tool_input: list_drones(),
                description="List all available drones in the session. No input required."
            ),
            Tool(
                name="get_weather",
                func=lambda tool_input: get_weather(),
                description="Get current weather conditions. No input required."
            ),
        ]

    # ==========================================
    # 3. System & Mission Tools
    # ==========================================

    def get_system_tools(self) -> List[BaseTool]:

        def set_home(drone_id: str) -> str:
            return self._safe_exec(self.client.set_home, drone_id=drone_id)

        def calibrate(drone_id: str) -> str:
            return self._safe_exec(self.client.calibrate, drone_id=drone_id)

        def charge(drone_id: str, charge_amount: float) -> str:
            return self._safe_exec(self.client.charge, drone_id=drone_id, charge_amount=charge_amount)
        
        def take_photo(drone_id: str) -> str:
            return self._safe_exec(self.client.take_photo, drone_id=drone_id)
        
        def get_task_progress() -> str:
            return self._safe_exec(self.client.get_task_progress)

        return [
            StructuredTool.from_function(
                func=set_home,
                name="set_home",
                description="Set the drone's current position as its new home.",
                args_schema=SetHomeParams
            ),
            StructuredTool.from_function(
                func=calibrate,
                name="calibrate",
                description="Calibrate the drone's sensors.",
                args_schema=CalibrateParams
            ),
            StructuredTool.from_function(
                func=charge,
                name="charge",
                description="Charge the drone's battery (must be at station).",
                args_schema=ChargeParams
            ),
            StructuredTool.from_function(
                func=take_photo,
                name="take_photo",
                description="Take a photo with the drone's camera.",
                args_schema=TakePhotoParams
            ),
            Tool(
                name="get_task_progress",
                func=lambda tool_input: get_task_progress(),
                description="Check the current mission task progress. No input required."
            )
        ]

    # ==========================================
    # 4. Aggregation
    # ==========================================

    def get_all_tools(self) -> List[BaseTool]:
        return (
            self.get_navigation_tools() + 
            self.get_perception_tools() + 
            self.get_system_tools()
        )