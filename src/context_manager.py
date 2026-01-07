# src/context_manager.py
import json

class DroneContextManager:
    def __init__(self, client):
        self.client = client
        self.drone_map = {}  # 存储 { "Drone 1": "id_123", "Drone 2": "id_456" }
        self.drone_info_summary = "" # 存储给 LLM 看的精简文本

    def refresh(self):
        """调用 list_drones 并构建精简映射表"""
        print("🔄 正在从服务端同步无人机列表...")
        
        # 1. 调用 API (假设 execute_command 已经封装好或直接用 client)
        # 这里模拟直接调用 client 的方法，你需要根据你的 execute_command 调整
        drones_list = self.client.list_drones() 
        
        # 2. 清空旧数据
        self.drone_map = {}
        summary_lines = []

        # 3. 提取关键信息 (过滤掉 useless 的字段)
        for drone in drones_list:
            d_name = drone.get('name', 'Unknown') # 例如 "Drone 1"
            d_id = drone.get('id')                # 例如 "487bc0b6"
            d_status = drone.get('status')        # 例如 "idle"
            
            # 建立映射
            self.drone_map[d_name] = d_id
            
            # 构建给 LLM 看的单行简介
            # 格式: - Drone 1 (ID: 487bc0b6): [idle]
            summary_lines.append(f"- {d_name} (ID: {d_id}): [{d_status}]")

        self.drone_info_summary = "\n".join(summary_lines)
        print(f"✅ 无人机列表已更新，共发现 {len(self.drone_map)} 架无人机。")

    def get_id_by_name(self, name_query):
        """辅助函数：尝试根据名字找ID (也可以让LLM自己找，这个函数给后端逻辑兜底)"""
        for name, pid in self.drone_map.items():
            if name.lower() in name_query.lower():
                return pid
        return None

    def get_system_prompt_context(self):
        """返回注入到 System Prompt 中的文本"""
        return f"""
            当前可用无人机列表 (Name -> ID 映射):
            {self.drone_info_summary}
            ----------------------------------
            """