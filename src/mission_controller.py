# src/mission_controller.py
import time
import json
import os
from datetime import datetime
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
        
        # 2. 初始化大脑 (LLM)
        llm_svc = LLMService()
        self.llm = llm_svc.create_llm("Ollama", override_temperature=0.1) 
        
        self.mission_completed = False

        # --- 新增：日志系统初始化 ---
        self.llm_conversation_count = 0  # 对话计数器
        
        # 创建基础目录 llm_logs/YYYYMMDD_HHMMSS
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join("llm_logs", current_time_str)
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"📁 LLM 日志目录已创建: {self.log_dir}")

    def run(self):
        print(f"🚀 任务开始: {self.drone_id}")

        if not self._check_ready():
            print("❌ 无人机未就绪，中止任务")
            return

        self.executor.execute("take_off", {"drone_id": self.drone_id, "altitude": 10})
        time.sleep(2)

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
        增加日志记录功能的 LLM 请求方法
        """
        # 1. 计数器自增
        self.llm_conversation_count += 1
        
        # 2. 准备输入数据
        prompt_template = """
            你是一个无人机任务规划助手。
            当前无人机状态: {status}
            当前位置: {position}
            
            请分析当前情况，给出一个下一步探索的坐标 (x, y, z)。
            只返回 JSON 格式，例如: {{"x": 10, "y": 20, "z": 5}}
            不要包含其他废话。
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = prompt | self.llm | JsonOutputParser()
        
        current_pos = current_status.get("position", {"x": 0, "y": 0, "z": 0})
        
        # 构造输入变量字典
        input_vars = {
            "status": str(current_status), 
            "position": str(current_pos)
        }

        # 初始化日志结构
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "dialogue_id": self.llm_conversation_count,
            "prompt_template": prompt_template,
            "inputs": input_vars,
            "output": None,
            "success": False,
            "error_message": None,
            "latency_seconds": 0.0
        }

        result = None
        start_time = time.time()

        try:
            # 3. 执行 LLM 调用
            result = chain.invoke(input_vars)
            
            # 记录成功结果
            log_entry["output"] = result
            log_entry["success"] = True

        except Exception as e:
            # 记录失败结果
            error_msg = str(e)
            print(f"⚠️ LLM 思考失败: {error_msg}，执行随机探索")
            log_entry["error_message"] = error_msg
            log_entry["success"] = False
            
            # 降级策略
            result = {"x": current_pos["x"] + 1, "y": current_pos["y"], "z": 5}

        finally:
            # 4. 计算耗时并保存日志
            end_time = time.time()
            log_entry["latency_seconds"] = round(end_time - start_time, 4)
            self._save_llm_log(log_entry)

        return result

    def _save_llm_log(self, log_data: Dict):
        """
        将单次对话保存为 JSON 文件
        """
        # 文件名示例: 001_dialogue.json, 002_dialogue.json
        filename = f"{self.llm_conversation_count:03d}_dialogue.json"
        file_path = os.path.join(self.log_dir, filename)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"❌ 写入 LLM 日志失败: {e}")

# ============================
if __name__ == "__main__":
    controller = MissionController()
    controller.run()