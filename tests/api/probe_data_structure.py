import sys
import json
import time
from pathlib import Path

# 环境配置
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.uav_api_client import UAVAPIClient

def probe_structure():
    print("🔬 Probing Data Structures & Missing Commands...")
    client = UAVAPIClient("http://localhost:8000")
    
    # 1. 获取一个可用的无人机 ID
    drones = client.list_drones()
    if not drones:
        print("❌ No drones found.")
        return
    
    drone_id = drones[0]['id']
    print(f"👉 Target Drone: {drone_id}")

    # ==========================================
    # 任务 A: 补测关键移动与交互指令
    # ==========================================
    print("\n[Testing Missing Commands]")
    
    # 测试 move_to (关键!)
    try:
        # 我们尝试原地移动或者微小移动，主要看是否报 403
        print("   ⏳ Testing: move_to...", end="\r")
        # 获取当前位置，尝试往上飞 0.1 米 (安全测试)
        status = client.get_drone_status(drone_id)
        current_z = status.get('position', {}).get('z', 0)
        
        # 构造移动指令
        client.move_to(drone_id, x=0, y=0, z=current_z + 0.5)
        print("✅ move_to            : AVAILABLE")
    except Exception as e:
        if "403" in str(e):
            print("❌ move_to            : DENIED (403)")
        else:
            # 如果是其他错误（如参数错误），通常说明权限是有的
            print(f"⚠️ move_to            : ERROR ({str(e)}) - Likely AVAILABLE but failed execution")

    # 测试 charge (充电)
    try:
        print("   ⏳ Testing: charge...", end="\r")
        client.charge(drone_id, charge_amount=10)
        print("✅ charge             : AVAILABLE")
    except Exception as e:
        msg = str(e)
        if "403" in msg:
            print("❌ charge             : DENIED (403)")
        elif "landed" in msg.lower() or "charger" in msg.lower():
            # 这种错误说明API调用通了，只是物理条件不满足
            print("✅ charge             : AVAILABLE (Logic Constraint)")
        else:
            print(f"⚠️ charge             : {msg}")

    # ==========================================
    # 任务 B: 抓取数据结构 (Schema 依据)
    # ==========================================
    print("\n[Capturing Data Payloads]")
    
    # 1. 抓取 Drone Status
    try:
        status_data = client.get_drone_status(drone_id)
        print("\n📄 [Payload] get_drone_status:")
        print(json.dumps(status_data, indent=2))
    except Exception as e:
        print(f"❌ Failed to get status: {e}")

    # 2. 抓取 Nearby Entities (核心感知数据)
    try:
        nearby_data = client.get_nearby_entities(drone_id)
        print("\n📄 [Payload] get_nearby_entities:")
        print(json.dumps(nearby_data, indent=2))
    except Exception as e:
        print(f"❌ Failed to get nearby: {e}")

if __name__ == "__main__":
    probe_structure()