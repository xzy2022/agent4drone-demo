# src\uav_executor.py
import logging
import inspect
from typing import Dict, Any, List, Optional, Union
from uav_api_client import UAVAPIClient


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("UAVExecutor")

class UAVExecutor:
    """
    UAV 执行器层 (The "Hand" of the system)。
    
    职责：
    1. 接收结构化的指令 (函数名 + 参数)。
    2. 利用反射机制动态调用 UAVAPIClient 的方法。
    3. 捕获底层 API 的异常，返回统一格式的执行结果。
    4. 完全不包含任何 LLM/AI 逻辑。
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化执行器。
        
        Args:
            base_url: 无人机控制服务器地址。
            client_instance: (可选) 允许注入已有的 client 实例，方便测试。
        """
        self.client = UAVAPIClient(base_url)

    def execute(self, func_name: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        核心方法：执行具体的动作。

        Args:
            func_name (str): UAVAPIClient 中的方法名 (例如 "take_off")
            params (dict): 传递给方法的参数字典 (例如 {"altitude": 10})

        Returns:
            Dict: 统一格式的结果字典
            {
                "success": bool,
                "action": str,
                "result": Any,   # API 返回的数据
                "error": str     # 如果失败，错误信息
            }
        """
        if params is None:
            params = {}

        # 1. 安全检查：禁止调用私有方法 (以 _ 开头)
        if func_name.startswith("_"):
            return self._format_result(False, func_name, error="Access denied to private methods.")

        # 2. 检查方法是否存在
        if not hasattr(self.client, func_name):
            return self._format_result(False, func_name, error=f"Function '{func_name}' not supported by UAV Client.")

        # 3. 获取方法对象
        func = getattr(self.client, func_name)

        if not callable(func):
            return self._format_result(False, func_name, error=f"'{func_name}' is a property, not a function.")

        # 4. 执行调用
        try:
            logger.info(f"⚡ Executing: {func_name} with params {params}")
            
            # 动态解包参数调用
            api_result = func(**params)
            
            logger.info(f"✅ Success: {func_name}")
            return self._format_result(True, func_name, result=api_result)

        except TypeError as e:
            # 捕获参数不匹配错误 (例如少传了参数)
            err_msg = f"Argument mismatch: {str(e)}"
            logger.error(f"❌ Failed: {err_msg}")
            return self._format_result(False, func_name, error=err_msg)

        except Exception as e:
            # 捕获 API 通信错误或其他运行时错误
            err_msg = str(e)
            logger.error(f"❌ API Error: {err_msg}")
            return self._format_result(False, func_name, error=err_msg)

    def get_available_actions(self) -> List[str]:
        """
        获取当前 Client 支持的所有公开方法名称。
        这对于后续让 LLM 知道有哪些工具可用非常重要。
        """
        methods = []
        for name, method in inspect.getmembers(self.client, predicate=inspect.ismethod):
            if not name.startswith("_"):
                methods.append(name)
        return methods

    def _format_result(self, success: bool, action: str, result: Any = None, error: str = None) -> Dict[str, Any]:
        """标准化返回格式"""
        return {
            "success": success,
            "action": action,
            "data": result,  # 统一叫 data，方便后续解析
            "error": error
        }

# ==========================================
# ==========================================
if __name__ == "__main__":
    import time
    
    BASE_URL = "http://localhost:8000"
    executor = UAVExecutor(base_url=BASE_URL)

    # 打印支持的工具
    print(f"📋 可用工具: {executor.get_available_actions()}")

    # 测试指令集
    test_commands = [
        {"func": "get_drone_status", "params": {"drone_id": "487bc0b6"}},
        {"func": "take_off", "params": {"drone_id": "487bc0b6", "altitude": 50}},
        {"func": "take_off", "params": {"drone_id": "487bc0b6"}}, # 故意少传参数测试，但是没报错。
        
        {"func": "destroy_world", "params": {}}, # 故意调用不存在的函数
    ]

    for i, cmd in enumerate(test_commands):
        print(f"\n--- Test {i+1} ---")
        result = executor.execute(cmd["func"], cmd.get("params"))
        print(f"执行结果: {result}")
        time.sleep(0.5)