# src2/executor.py
import json
import time
from typing import Dict, Any

from langchain_core.tools import BaseTool
from src.uav_api_client import UAVAPIClient
from src2.tools_registry import UAVToolRegistry
from src2.schemas import MissionPlan, AgentAction

class MissionExecutor:
    """
    执行器：负责将静态的 MissionPlan 转化为实际的工具调用。
    核心思想：复用 UAVToolRegistry 中的 Tool 定义，确保行为一致性。
    """

    def __init__(self, client: UAVAPIClient):
        self.client = client
        self.registry = UAVToolRegistry(client)
        # 获取所有工具并建立索引： "take_off" -> Tool Object
        self.tools_map: Dict[str, BaseTool] = {
            t.name: t for t in self.registry.get_all_tools()
        }

    def execute_plan(self, plan: MissionPlan) -> None:
        """
        顺序执行 MissionPlan 中的所有步骤
        """
        print(f"🚀 Starting Mission Execution: {len(plan.mission_steps)} steps identified.")
        
        for i, step in enumerate(plan.mission_steps, 1):
            self._execute_single_step(i, step)
            # 可选：步骤间暂停，防止指令发送过快
            time.sleep(1.0) 
            
        print("✅ Mission Execution Logic Completed.")

    def _execute_single_step(self, index: int, action: AgentAction):
        tool_name = action.func
        tool_args = action.params
        thought = action.thought

        print(f"\n[Step {index}] {tool_name}")
        if thought:
            print(f"  💭 Thought: {thought}")
        
        # 1. 查找工具
        tool = self.tools_map.get(tool_name)
        if not tool:
            print(f"  ❌ Error: Tool '{tool_name}' not found in registry.")
            return

        # 2. 执行工具 (复用 Registry 中的 _safe_exec 逻辑)
        # LangChain Tool 的 run 方法会自动验证参数是否符合 Schema
        try:
            # 注意：tool.run 接收字典或字符串，这里传入 params 字典
            result = tool.run(tool_args)
            print(f"  ✅ Result: {result}")
        except Exception as e:
            print(f"  ❌ Execution Failed: {str(e)}")

