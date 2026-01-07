# src/mission_controller.py
import time
import json
from typing import Dict, Any
from uav_executor import UAVExecutor
from llm_service import LLMService
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class MissionController:
    def __init__(self, drone_id: str = "487bc0b6"):
        self.drone_id = drone_id
        
        # 1. 初始化手 (Executor)
        self.executor = UAVExecutor()
        
        # 2. 初始化大脑 (LLM) - 用于探索决策
        # 注意：这里你可以根据配置切换 "Ollama" 或 "OpenAI"
        llm_svc = LLMService()
        self.llm = llm_svc.create_llm("Ollama", override_temperature=0.1) 
        
        self.mission_completed = False

    def run(self):
        """
        对应 Mermaid 图中的主流程
        """
        print(f"🚀 任务开始: {self.drone_id}")

        # --- Init 阶段 ---
        if not self._check_ready():
            print("❌ 无人机未就绪，中止任务")
            return

        # TakeOff
        self.executor.execute("take_off", {"drone_id": self.drone_id, "altitude": 10})
        time.sleep(2)

        # --- 循环感知阶段 (While Loop) ---
        while True:
            # 1. Observe (获取感知数据)
            status = self._get_status()
            if not status:
                break
            
            # 2. CheckBattery (优先级 1: 生存)
            if status.get("battery", 100) < 20:
                print("🪫 电量不足 (<20%)，触发返航...")
                self._return_home()
                break

            # 3. CheckTask (优先级 2: 任务完成)
            if self.mission_completed: # 这里可以根据 status 判断
                print("✅ 任务已完成，返航...")
                self._return_home()
                break

            # 4. CheckObstacle (优先级 3: 避障 - 必须优先于移动)
            # 假设 status 里有 obstacle_detected 字段
            if status.get("obstacle_detected", False):
                print("🚧 检测到障碍物，执行避障...")
                self.executor.execute("avoid_obstacle", {"drone_id": self.drone_id, "direction": "right"})
                time.sleep(1)
                continue

            # 5. CheckTarget (优先级 4: 发现目标)
            # 模拟：假设 status 里有 visual_targets
            targets = status.get("visual_targets", [])
            if targets:
                print(f"🎯 发现目标: {targets}，执行接近...")
                self.executor.execute("move_to", {"drone_id": self.drone_id, "position": targets[0]['pos']})
                self.executor.execute("record_data", {"target_id": targets[0]['id']})
                self.mission_completed = True # 假设发现即完成
                continue

            # 6. ExploreStrategy (优先级 5: 默认探索)
            # === 这里我们引入 LLM 做决策 ===
            print("🧭 无特定事件，请求 LLM 生成探索策略...")
            next_move = self._ask_llm_for_strategy(status)
            
            if next_move:
                print(f"🤖 LLM 建议移动至: {next_move}")
                move_params = {
                    "drone_id": self.drone_id,
                    "x": next_move.get("x"),
                    "y": next_move.get("y"),
                    "z": next_move.get("z")
                }
                self.executor.execute("move_to", move_params)
            
            time.sleep(1) # 模拟循环间隔

    def _check_ready(self) -> bool:
        """检查无人机是否就绪"""
        res = self.executor.execute("get_drone_status", {"drone_id": self.drone_id})
        return res["success"] and res["data"].get("state") != "error"

    def _get_status(self) -> Dict[str, Any]:
        """获取当前综合状态"""
        res = self.executor.execute("get_drone_status", {"drone_id": self.drone_id})
        if res["success"]:
            return res["data"]
        return {}

    def _return_home(self):
        self.executor.execute("return_home", {"drone_id": self.drone_id})

    def _ask_llm_for_strategy(self, current_status: Dict) -> Dict:
        """
        利用 LLMService 决定下一步去哪。
        这是 LLM 发挥作用的地方：处理非结构化环境信息。
        """
        prompt = ChatPromptTemplate.from_template(
            """
            你是一个无人机任务规划助手。
            当前无人机状态: {status}
            当前位置: {position}
            
            请分析当前情况，给出一个下一步探索的坐标 (x, y, z)。
            只返回 JSON 格式，例如: {{"x": 10, "y": 20, "z": 5}}
            不要包含其他废话。
            """
        )
        
        chain = prompt | self.llm | JsonOutputParser()
        
        try:
            # 假设 status 里包含位置信息
            current_pos = current_status.get("position", {"x":0, "y":0, "z":0})
            result = chain.invoke({"status": str(current_status), "position": str(current_pos)})
            return result
        except Exception as e:
            print(f"⚠️ LLM 思考失败: {e}，执行随机探索")
            return {"x": current_pos["x"]+1, "y": current_pos["y"], "z": 5} # 降级策略

# ============================
if __name__ == "__main__":
    # 模拟运行
    controller = MissionController()
    controller.run()