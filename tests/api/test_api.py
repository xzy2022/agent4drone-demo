# tests\api\test_api.py
import time
import json
from uav_api_client import UAVAPIClient

# --- 核心封装：通用指令执行器 ---
def execute_command(client: UAVAPIClient, func_name: str, params: dict = None):
    """
    通用执行函数：根据函数名动态调用 client 中的方法
    
    Args:
        client: UAVAPIClient 实例
        func_name: 要调用的函数名 (字符串)
        params: 参数字典 (例如 {'altitude': 10})
    """
    if params is None:
        params = {}

    # 1. 检查 Client 里有没有这个函数
    if not hasattr(client, func_name):
        return f"❌ Error: Function '{func_name}' not found."

    # 2. 获取函数对象
    func = getattr(client, func_name)

    # 3. 执行函数
    try:
        print(f"⚡ Calling: {func_name}({params})")
        # **kwargs 解包：把字典自动对应到函数的参数上
        result = func(**params) 
        return result
    except Exception as e:
        return f"❌ Execution Error: {str(e)}"

# --- 测试代码 ---
if __name__ == "__main__":
    # 配置
    BASE_URL = "http://localhost:8000"
    DRONE_ID = "487bc0b6"  # 替换你的 ID
    
    client = UAVAPIClient(BASE_URL)
    
    # === 模拟智能体生成的指令流 ===
    # 以后你的 LLM 只需要生成下面这种 JSON 格式的数据即可
    agent_commands = [
        # 1. 测试无参调用
        {"func": "get_drone_status", "params": {"drone_id": DRONE_ID}},
        
        # 2. 测试简单参数
        {"func": "take_off", "params": {"drone_id": DRONE_ID, "altitude": 5}},
        
        # 3. 测试多参数
        {"func": "move_to", "params": {"drone_id": DRONE_ID, "x": 50, "y": 50, "z": 5}},
        
        # 4. 测试感知接口
        {"func": "get_nearby_entities", "params": {"drone_id": DRONE_ID}},
        
        # 5. 测试一个不存在的接口 (测试健壮性)
        {"func": "dance_in_the_air", "params": {"drone_id": DRONE_ID}},
        
        # 6. 返航
        {"func": "return_home", "params": {"drone_id": DRONE_ID}},

        # 7. 任务
        {"func": "get_task_progress", "params": {}},

        # 8. 天气。环境相关的其它内容都无法获取，受限于agent权限。
        {"func": "get_weather", "params": {}},

        # 9. 拍照。没看出来什么用，似乎走到一个目标点附近就直接完成了探测。
        {"func": "move_to", "params": {"drone_id": DRONE_ID, "x": 750, "y": 300, "z": 5}},
        {"func": "take_photo", "params": {"drone_id": DRONE_ID}},

        # 10. 旋转
        {"func": "rotate", "params": {"drone_id": DRONE_ID, "heading": 108}},

        # 11. 降落
        {"func": "land", "params": {"drone_id": DRONE_ID}},

        # 12. 充电。必须在充电桩处降落才能充电。
        {"func": "charge", "params": {"drone_id": DRONE_ID, "charge_amount": 30}},

        # 13. 起飞
        {"func": "take_off", "params": {"drone_id": DRONE_ID, "altitude": 20}},

        # 14. 校准。似乎某些状态才能校准，至少悬停是不可以的。
        {"func": "calibrate", "params": {"drone_id": DRONE_ID}},
        
    ]

    print(f"🤖 开始测试通用执行器...\n")

    for cmd in agent_commands:
        f_name = cmd["func"]
        f_params = cmd["params"]
        
        # 调用通用接口
        result = execute_command(client, f_name, f_params)
        
        # 打印结果
        print(f"   -> Result: {result}\n")
        
        # 模拟思考间隔
        time.sleep(1)