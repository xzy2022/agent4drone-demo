# src/nlp_commander.py
import time
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from llm_service import LLMService
from context_manager import DroneContextManager 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import BaseMessage

class NaturalLanguageCommander:
    def __init__(self, context_manager):
        """
        初始化自然语言指挥官
        :param context_manager: 必须传入已初始化好的 DroneContextManager 实例
        """
        # === 1. 绑定上下文管理器 (依赖注入) ===
        self.context_manager = context_manager
        
        # === 2. 初始化大脑 (LLM) ===
        # 使用较低的 temperature (0.1) 以保证指令解析的稳定性
        llm_svc = LLMService()
        self.llm = llm_svc.create_llm("Ollama", override_temperature=0.0) 
        
        # === 3. 日志系统初始化 ===
        self.llm_conversation_count = 0
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join("llm_logs", "nlp_commands", current_time_str)
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"📁 NLP 指令日志目录已创建: {self.log_dir}")

    def parse_instruction(self, text_command: str) -> Dict[str, Any]:
        """
        将自然语言文本解析为标准化的控制命令序列
        """
        self.llm_conversation_count += 1
        
        # === Step A: 获取最新的环境上下文 ===
        # 在每次解析前，获取当前最新的 "名称 -> ID" 映射字符串
        # 这样即使无人机是刚上线的，LLM 也能获得正确的 ID
        current_context_str = self.context_manager.get_system_prompt_context()

        # 如果环境为空，打印警告（但不阻断，可能是在测试无无人机场景）
        if "没有检测到" in current_context_str:
            print("⚠️ 警告: 当前环境中没有检测到在线无人机，生成的指令可能缺乏有效 ID。")

        # === Step B: 核心 Prompt 设计 ===
        prompt_template = """
        你是一个无人机编队控制中枢。将用户的自然语言指令转换为标准 JSON 控制序列。

        ### 关键规则
        1. **ID 匹配**: 必须根据下面的 [环境上下文] 将自然语言名称（如 "Drone 1"）转换为真实的 UUID。
        2. **格式限制**: 仅输出 JSON 对象，**不要**包含 Markdown (```json) 标记或额外解释。
        
        ### 环境上下文 (Name -> UUID 映射)
        {drone_context_str}

        ### API 接口定义
        支持的函数 (func) 及参数 (params):
        1. 动作类:
        - take_off(drone_id: str, altitude: float)
        - move_to(drone_id: str, x: float, y: float, z: float)
        - land(drone_id: str)
        - return_home(drone_id: str)
        - take_photo(drone_id: str)
        2. 查询类:
        - list_drones()
        - get_drone_status(drone_id: str)

        ### 用户指令
        "{user_input}"

        ### 输出 JSON 结构示例
        {{
            "mission_steps": [
                {{ "func": "take_off", "params": {{ "drone_id": "uav_uuid_here", "altitude": 10 }} }},
                {{ "func": "move_to", "params": {{ "drone_id": "uav_uuid_here", "x": 10, "y": 20, "z": 10 }} }}
            ]
        }}
        """
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        parser = JsonOutputParser()
        
        # === Step C: 注入变量 (包含上下文和用户输入) ===
        input_vars = {
            "drone_context_str": current_context_str, # <--- 关键：注入上下文
            "user_input": text_command
        }

        # 初始化日志结构
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "dialogue_id": self.llm_conversation_count,
            "input_text": text_command,
            "context_used": current_context_str, # 记录当时使用了什么上下文
            "raw_response": None,
            "parsed_output": None,
            "success": False,
            "error_message": None,
            "latency_seconds": 0.0
        }

        result = {"mission_steps": []}
        start_time = time.time()

        print(f"🔄 正在解析指令: \"{text_command}\" ...")

        try:
            # Step D: 调用 LLM
            messages = prompt.invoke(input_vars)
            response = self.llm.invoke(messages)
            
            # 提取原始文本
            raw_content = response.content if isinstance(response, BaseMessage) else str(response)
            log_entry["raw_response"] = raw_content

            # Step E: 解析 JSON
            parsed_result = parser.parse(raw_content)
            
            # 简单的格式标准化
            if isinstance(parsed_result, list):
                # 如果 LLM 没按示例返回 {"mission_steps": []} 而是直接返回了列表
                result["mission_steps"] = parsed_result
            elif isinstance(parsed_result, dict):
                if "mission_steps" in parsed_result:
                    result = parsed_result
                else:
                    # 如果返回了单个 dict 动作，包裹进列表
                    result["mission_steps"] = [parsed_result]

            log_entry["parsed_output"] = result
            log_entry["success"] = True

        except Exception as e:
            error_msg = str(e)
            print(f"❌ 解析失败: {error_msg}")
            log_entry["error_message"] = error_msg
            log_entry["success"] = False

        finally:
            end_time = time.time()
            log_entry["latency_seconds"] = round(end_time - start_time, 4)
            self._save_llm_log(log_entry)

        return result

    def _save_llm_log(self, log_data: Dict):
        filename = f"{self.llm_conversation_count:03d}_nlp_parse.json"
        file_path = os.path.join(self.log_dir, filename)
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"❌ 写入日志失败: {e}")

    def execute_parsed_mission(self, parsed_data: Dict):
        """
        模拟执行解析后的命令
        """
        steps = parsed_data.get("mission_steps", [])
        
        if not steps:
            print("⚠️ 未解析到有效步骤，跳过执行。")
            return

        print(f"\n🚀 开始执行任务序列 ({len(steps)} 步):")
        print("="*50)
        
        for i, step in enumerate(steps, 1):
            func_name = step.get("func")
            params = step.get("params", {})
            drone_id = params.get("drone_id", "UNKNOWN_ID")
            
            # 模拟执行延迟
            print(f"Step {i}: 无人机[{drone_id}] -> 执行 [{func_name}] 参数: {params}")
            time.sleep(0.5) 
            
        print("="*50)
        print("✅ 序列执行完毕。\n")


# ==========================================
# 独立运行测试块 (集成测试)
# ==========================================
if __name__ == "__main__":
    # 1. 环境路径设置 (确保能导入同级模块)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
    sys.path.append(str(project_root)) # 将 src 的上一级加入 path，或者将 src 加入 path
    # 这里为了方便直接 import src 下的模块，建议把 src 目录加进去
    sys.path.append(str(current_dir)) 

    from uav_api_client import UAVAPIClient
    from context_manager import DroneContextManager
    
    # 2. 初始化底层连接
    print("\n" + "="*60)
    base_url = "http://localhost:8000"
    print(f"🔌 正在连接 UAV Server ({base_url})...")
    
    try:
        real_client = UAVAPIClient(base_url)
        # 测试连接
        drones = real_client.list_drones()
        print(f"✅ 连接成功! 当前在线无人机数量: {len(drones)}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("⚠️ 将使用空上下文进行测试 (解析结果中的 ID 将无法匹配)...")
        real_client = UAVAPIClient(base_url) # 继续实例化以便程序跑下去

    # 3. 初始化上下文管理器
    context_manager = DroneContextManager(real_client)
    # 强制刷新一次，获取最新列表
    context_manager.refresh()

    # 4. 初始化指挥官 (注入上下文管理器)
    commander = NaturalLanguageCommander(context_manager)

    # 5. 测试用例
    test_commands = [
        "Drone 2 take off to 35 meters", 
        "Make Drone 1 fly to (100, 200, 50) and then take a photo",
    ]

    for cmd in test_commands:
        print(f"\n🗣️  指令: {cmd}")
        
        # 解析
        parsed_mission = commander.parse_instruction(cmd)
        
        # 打印原始 JSON (Debug)
        # print(json.dumps(parsed_mission, indent=2, ensure_ascii=False))
        
        # 执行
        commander.execute_parsed_mission(parsed_mission)
        time.sleep(1)