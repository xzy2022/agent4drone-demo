import sys
import os
import time
import json
from pathlib import Path

# --- 环境路径设置: 确保能导入 src 模块 ---
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))


from src.uav_api_client import UAVAPIClient

def test_api_permission(name, func, **kwargs):
    """
    执行单个 API 调用并根据返回结果判断权限状态
    Returns: (status_icon, status_text, execution_result)
    """
    try:
        print(f"   ⏳ Testing: {name}...", end="\r")
        result = func(**kwargs)
        
        # 检查是否包含由 wrapper 捕获的权限错误字符串
        result_str = str(result)
        if "Permission denied" in result_str or "Access denied" in result_str:
             return "❌", "DENIED (403)", result
        if "Authentication failed" in result_str:
             return "🚫", "AUTH FAIL (401)", result
        
        # 成功
        return "✅", "AVAILABLE", result

    except Exception as e:
        err_msg = str(e)
        if "403" in err_msg or "Permission denied" in err_msg:
            return "❌", "DENIED (403)", err_msg
        elif "401" in err_msg:
            return "🚫", "AUTH FAIL (401)", err_msg
        elif "404" in err_msg:
            return "❓", "NOT FOUND (404)", err_msg
        else:
            return "⚠️", f"ERROR ({type(e).__name__})", err_msg

def run_capability_scan():
    print("="*60)
    print("🕵️  UAV API Capability Scan (Student Agent Permission Check)")
    print("="*60)
    
    base_url = "http://localhost:8000"
    client = UAVAPIClient(base_url)
    
    results = []

    # ==========================================
    # 1. 全局信息/环境类接口 (Session & Environment)
    # ==========================================
    print("\n[Global Information Interfaces]")
    global_checks = [
        ("get_current_session", client.get_current_session, {}),
        ("get_weather", client.get_weather, {}),
        ("get_targets", client.get_targets, {}),          # 重点关注：Student 角色通常看不到全局目标
        ("get_obstacles", client.get_obstacles, {}),      # 重点关注：Student 角色通常看不到全局障碍
        ("get_waypoints", client.get_waypoints, {}),
        ("get_task_progress", client.get_task_progress, {}),
    ]

    for name, func, params in global_checks:
        icon, status, res = test_api_permission(name, func, **params)
        print(f"{icon} {name:<25} : {status}")
        results.append({"name": name, "status": status, "type": "Global"})

    # ==========================================
    # 2. 无人机特定接口 (Drone Specific)
    # ==========================================
    print("\n[Drone Specific Interfaces]")
    
    # 先获取无人机列表
    try:
        drones = client.list_drones()
        icon, status, _ = test_api_permission("list_drones", client.list_drones)
        print(f"{icon} {'list_drones':<25} : {status}")
        
        if not drones:
            print("⚠️  No drones available via list_drones(). Skipping drone-specific tests.")
            return
        
        # 选取第一架无人机进行测试
        test_drone_id = drones[0].get('id')
        print(f"👉 Using Drone ID: {test_drone_id} for testing commands")

    except Exception as e:
        print(f"❌ Failed to list drones: {e}")
        return

    # 这里的测试按"安全性"排序，破坏性小的在前
    drone_checks = [
        # --- 只读/感知类 (通常允许) ---
        ("get_drone_status", client.get_drone_status, {"drone_id": test_drone_id}),
        ("get_nearby_entities", client.get_nearby_entities, {"drone_id": test_drone_id}), # 重点：Student 应该用这个代替 get_targets
        
        # --- 动作类 (可能需要状态配合，只要不是 403 就算 Pass) ---
        ("take_photo", client.take_photo, {"drone_id": test_drone_id}),
        ("calibrate", client.calibrate, {"drone_id": test_drone_id}),
        ("set_home", client.set_home, {"drone_id": test_drone_id}),
        
        # --- 运动控制类 (警告：这些会真的让无人机动起来) ---
        # 我们主要看是否有权限调用，调用后立即捕获结果
        ("take_off", client.take_off, {"drone_id": test_drone_id, "altitude": 2.0}), 
        ("hover", client.hover, {"drone_id": test_drone_id, "duration": 1}),
        ("rotate", client.rotate, {"drone_id": test_drone_id, "heading": 90}),
        ("land", client.land, {"drone_id": test_drone_id}), # 最后测试降落
    ]

    for name, func, params in drone_checks:
        icon, status, res = test_api_permission(name, func, **params)
        
        # 如果是因为状态不对（例如已经在地上还调用land）导致的Error，不算权限问题
        if "ERROR" in status and ("state" in str(res).lower() or "landed" in str(res).lower()):
            status = "AVAILABLE (Logic Error)"
            icon = "✅"
            
        print(f"{icon} {name:<25} : {status}")
        if "DENIED" in status:
            print(f"    └── Reason: {str(res)}")
        
        results.append({"name": name, "status": status, "type": "Drone"})
        time.sleep(0.5) # 稍微暂停，避免请求过快

    # ==========================================
    # 3. 总结建议
    # ==========================================
    print("\n" + "="*60)
    print("📋 API Schema Refactoring Recommendations")
    print("="*60)
    
    available = [r['name'] for r in results if "AVAILABLE" in r['status'] or "✅" in r['status']]
    denied = [r['name'] for r in results if "DENIED" in r['status']]

    print(f"✅ KEEP ({len(available)}): These should be defined in src/schemas.py")
    print(f"   {', '.join(available)}")
    
    print(f"\n❌ REMOVE/IGNORE ({len(denied)}): Do not include these in Student Agent schemas")
    for d in denied:
        print(f"   - {d}")

    # 特别提示逻辑
    if "get_targets" in denied and "get_nearby_entities" in available:
        print("\n💡 Insight: You cannot access global 'get_targets'. You MUST use 'get_nearby_entities' for perception.")
    
    if "get_obstacles" in denied:
        print("💡 Insight: You cannot see all obstacles. You must rely on 'get_nearby_entities' or collision warnings.")

if __name__ == "__main__":
    run_capability_scan()