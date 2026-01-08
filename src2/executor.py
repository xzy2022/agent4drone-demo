# src2/executor.py
import json
import time
import logging
from typing import Dict, Any, Optional

from langchain_core.tools import BaseTool

from src.uav_api_client import UAVAPIClient
from src2.tools_registry import UAVToolRegistry
from src2.schemas import MissionPlan, AgentAction

# 配置基础日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UAVExecutor")

class MissionExecutor:
    """
    任务执行器 (Action Executor)
    
    职责：
    1. 将静态的 MissionPlan (Pydantic 对象) 转化为实际的函数调用。
    2. 充当调度层，连接 '意图' (Schema) 与 '能力' (Tool Registry)。
    3. 负责任务执行的生命周期管理 (顺序执行、错误中断、延时控制)。
    """

    def __init__(self, client: UAVAPIClient):
        """
        初始化执行器
        
        Args:
            client: 初始化的 UAVAPIClient 实例，用于连接物理/仿真无人机。
        """
        self.client = client
        self.registry = UAVToolRegistry(client)
        
        # 策略：默认加载所有可用工具。
        # Executor 应当具备执行系统所有合法指令的能力。
        # 对能力的限制(如禁飞区)应在 Planning 阶段通过 Prompt 或 Tool 过滤处理。
        self.tools_map: Dict[str, BaseTool] = {
            t.name: t for t in self.registry.get_all_tools()
        }
        
        logger.info(f"Executor initialized with {len(self.tools_map)} tools.")

    def execute_plan(self, plan: MissionPlan, step_delay: float = 1.0) -> bool:
        """
        执行完整的任务计划
        
        Args:
            plan: 由 LLM 生成并校验过的 MissionPlan 对象。
            step_delay: 步骤之间的安全延时 (秒)。
            
        Returns:
            bool: 任务是否全部成功完成。
        """
        total_steps = len(plan.mission_steps)
        logger.info(f"🚀 Starting Mission: {total_steps} steps in queue.")
        
        for i, step in enumerate(plan.mission_steps, 1):
            logger.info(f"--- Executing Step {i}/{total_steps} ---")
            
            success = self._execute_single_step(step)
            
            if not success:
                logger.error(f"❌ Mission Aborted at step {i} due to failure.")
                return False
            
            # 步骤间暂停，防止指令发送过快导致硬件阻塞
            if i < total_steps:
                time.sleep(step_delay)
                
        logger.info("✅ Mission Completed Successfully.")
        return True

    def _execute_single_step(self, action: AgentAction) -> bool:
        """
        执行单个动作单元
        """
        tool_name = action.func
        tool_params = action.params
        thought = action.thought

        # 1. 打印思考过程 (如果有)
        if thought:
            logger.info(f"💭 Thought: {thought}")
        
        logger.info(f"🔧 Action: {tool_name} | Params: {tool_params}")

        # 2. 查找工具
        tool = self.tools_map.get(tool_name)
        if not tool:
            logger.error(f"Tool '{tool_name}' not found in registry. Is the plan using a valid tool name?")
            return False

        # 3. 执行工具
        # LangChain 的 Tool.run() 方法会自动处理:
        # - 参数校验 (基于 args_schema)
        # - 异常捕获 (如果在 Tool 定义中配置了 handle_tool_error，或者我们复用 registry 的 _safe_exec)
        try:
            # 注意：tools_registry.py 中的方法已经返回了 JSON 字符串
            # 这里我们获取结果并记录
            result_str = tool.run(tool_params)
            
            # 简单的结果检查逻辑
            # 由于 _safe_exec 即使报错也会返回字符串，我们需要根据内容判断是否真的成功
            # 这里的判断逻辑比较简单，实际生产中可能需要解析 JSON 里的 status 字段
            if "Error executing tool" in result_str:
                logger.error(f"Execution Error: {result_str}")
                return False
            
            logger.info(f"✅ Result: {result_str}")
            return True

        except Exception as e:
            # 这一层是最后的防线，防止 Tool 内部抛出未捕获异常导致程序崩溃
            logger.critical(f"Unhandled Exception during execution: {str(e)}")
            return False

