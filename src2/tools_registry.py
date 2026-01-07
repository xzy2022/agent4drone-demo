# src2/tools_registry.py
import json
from typing import List, Any, Callable, Dict, Optional
from functools import wraps

from langchain.tools import StructuredTool, Tool
from langchain_core.tools import BaseTool

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
    UAV 工具注册表
    负责将 UAVAPIClient 的原生方法包装为 LangChain 可用的 StructuredTool。
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
        """Returns tools for moving and controlling the drone."""
        
        def take_off(params: TakeOffParams) -> str:
            return self._safe_exec(self.client.take_off, 
                                   drone_id=params.drone_id, 
                                   altitude=params.altitude)

        def land(params: LandParams) -> str:
            return self._safe_exec(self.client.land, drone_id=params.drone_id)

        def move_to(params: MoveToParams) -> str:
            return self._safe_exec(self.client.move_to, 
                                   drone_id=params.drone_id, 
                                   x=params.x, y=params.y, z=params.z)

        def move_towards(params: MoveTowardsParams) -> str:
            return self._safe_exec(self.client.move_towards,
                                   drone_id=params.drone_id,
                                   distance=params.distance,
                                   heading=params.heading,
                                   dz=params.dz)

        def change_altitude(params: ChangeAltitudeParams) -> str:
            return self._safe_exec(self.client.change_altitude,
                                   drone_id=params.drone_id,
                                   altitude=params.altitude)

        def rotate(params: RotateParams) -> str:
            return self._safe_exec(self.client.rotate,
                                   drone_id=params.drone_id,
                                   heading=params.heading)

        def hover(params: HoverParams) -> str:
            return self._safe_exec(self.client.hover,
                                   drone_id=params.drone_id,
                                   duration=params.duration)

        def return_home(params: ReturnHomeParams) -> str:
            return self._safe_exec(self.client.return_home, drone_id=params.drone_id)

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
        """Returns tools for gathering information about environment and drones."""

        def get_drone_status(params: DroneBaseParams) -> str:
            return self._safe_exec(self.client.get_drone_status, drone_id=params.drone_id)

        def get_nearby_entities(params: GetNearbyEntitiesParams) -> str:
            return self._safe_exec(self.client.get_nearby_entities, drone_id=params.drone_id)

        def list_drones() -> str:
            """No params needed"""
            return self._safe_exec(self.client.list_drones)

        def get_weather() -> str:
            """No params needed"""
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
                func=lambda _: list_drones(),
                description="List all available drones in the session. No input required."
            ),
            Tool(
                name="get_weather",
                func=lambda _: get_weather(),
                description="Get current weather conditions. No input required."
            ),
        ]

    # ==========================================
    # 3. System & Mission Tools
    # ==========================================

    def get_system_tools(self) -> List[BaseTool]:
        """Returns tools for system maintenance and mission specific actions."""

        def set_home(params: SetHomeParams) -> str:
            return self._safe_exec(self.client.set_home, drone_id=params.drone_id)

        def calibrate(params: CalibrateParams) -> str:
            return self._safe_exec(self.client.calibrate, drone_id=params.drone_id)

        def charge(params: ChargeParams) -> str:
            return self._safe_exec(self.client.charge, 
                                   drone_id=params.drone_id, 
                                   charge_amount=params.charge_amount)
        
        def take_photo(params: TakePhotoParams) -> str:
            return self._safe_exec(self.client.take_photo, drone_id=params.drone_id)
        
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
                func=lambda _: get_task_progress(),
                description="Check the current mission task progress. No input required."
            )
        ]

    # ==========================================
    # 4. Aggregation
    # ==========================================

    def get_all_tools(self) -> List[BaseTool]:
        """Returns a combined list of all available tools."""
        return (
            self.get_navigation_tools() + 
            self.get_perception_tools() + 
            self.get_system_tools()
        )