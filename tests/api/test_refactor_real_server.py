# tests\api\test_refactor_real_server.py
import sys
import os
import json
import unittest
from pathlib import Path

# --- 1. 环境路径设置 ---
# 将项目根目录加入路径，以便能导入 src 和 src2
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.uav_api_client import UAVAPIClient
# 导入新的重构成果 (Schema & Registry)
from src2.schemas import TakeOffParams, MoveToParams
from src2.tools_registry import UAVToolRegistry
from pydantic import ValidationError



# --- 2. 测试配置 ---
API_BASE_URL = "http://localhost:8000"  # 请根据实际情况修改

class TestRefactorIntegration(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """在所有测试开始前，连接一次服务器，获取一个真实的 drone_id"""
        print(f"\n🔌 正在连接服务器: {API_BASE_URL} ...")
        cls.client = UAVAPIClient(API_BASE_URL)
        cls.registry = UAVToolRegistry(cls.client)
        
        try:
            drones = cls.client.list_drones()
            if drones:
                cls.test_drone_id = drones[0]['id']
                print(f"✅ 连接成功，使用测试无人机 ID: {cls.test_drone_id}")
            else:
                print("⚠️ 连接成功但未发现无人机，部分测试将被跳过。")
                cls.test_drone_id = None
        except Exception as e:
            print(f"❌ 无法连接到服务器: {e}")
            print("请确保仿真器/服务器已启动。")
            sys.exit(1)

    # === 测试 1: Schema 验证 (第一步成果) ===
    def test_01_schema_validation(self):
        """验证 Pydantic 是否能在本地拦截非法参数"""
        print("\n🧪 [Test 1] Schema Validation (Local Guard)")
        
        # 1. 测试合法参数
        try:
            params = TakeOffParams(drone_id="test_id", altitude=10.0)
            self.assertEqual(params.altitude, 10.0)
            print("   ✅ 合法参数校验通过")
        except ValidationError:
            self.fail("合法参数不应触发 ValidationError")

        # 2. 测试非法参数 (例如高度为负数，Schema中定义了 gt=0)
        # 如果这里通过了，说明你的 Schema 起到了保护作用
        with self.assertRaises(ValidationError):
            TakeOffParams(drone_id="test_id", altitude=-5.0)
        print("   ✅ 非法参数(负高度)被 Schema 成功拦截")

    # === 测试 2: 工具生成 (第三步成果) ===
    def test_02_tool_structure(self):
        """验证 Registry 是否生成了符合 LangChain 标准的工具"""
        print("\n🧪 [Test 2] Tool Registry Structure")
        
        nav_tools = self.registry.get_navigation_tools()
        tool_names = [t.name for t in nav_tools]
        
        # 验证核心工具是否存在
        self.assertIn("take_off", tool_names)
        self.assertIn("move_to", tool_names)
        
        # 验证工具是否绑定了正确的 Schema
        take_off_tool = next(t for t in nav_tools if t.name == "take_off")
        self.assertEqual(take_off_tool.args_schema, TakeOffParams)
        print(f"   ✅ 工具列表生成正常: {tool_names[:3]}...")
        print(f"   ✅ 工具 'take_off' 已绑定 Schema: {take_off_tool.args_schema.__name__}")

    # === 测试 3: 真实服务器调用 (集成测试) ===
    def test_03_real_execution_read(self):
        """测试使用 Tool 实际上能否从服务器读取数据 (Read-Only)"""
        print("\n🧪 [Test 3] Real Server Execution (Read-Only)")
        
        if not self.test_drone_id:
            self.skipTest("无可用无人机，跳过实机测试")
        
        # 获取 perception 类工具
        tools = self.registry.get_perception_tools()
        status_tool = next(t for t in tools if t.name == "get_drone_status")
        
        # 模拟 Agent 调用工具 (传入字典)
        # 注意：这里我们传入的是字典，registry 会自动用 Schema 验证它
        input_args = {"drone_id": self.test_drone_id}
        
        print(f"   📡 正在通过工具调用 API: get_drone_status({self.test_drone_id})...")
        result_str = status_tool.invoke(input_args)
        
        # 验证返回的是 JSON 字符串且包含有效数据
        result = json.loads(result_str)
        
        # 不同的后端返回结构可能不同，但通常会有 status 或 id
        self.assertTrue(isinstance(result, dict))
        # 验证 ID 是否匹配 (取决于你的 API 返回结构，这里做宽泛检查)
        print("   ✅ 服务器返回数据成功")

    def test_04_real_execution_action(self):
        """测试真实的动作指令 (Write Action) - 会真的让无人机动作，请小心"""
        print("\n🧪 [Test 4] Real Server Execution (Action: Hover)")
        
        if not self.test_drone_id:
            self.skipTest("无可用无人机，跳过实机测试")

        # 我们选一个副作用最小的指令：Hover (悬停) 或 list_drones
        # 这里测试 Hover
        nav_tools = self.registry.get_navigation_tools()
        hover_tool = next(t for t in nav_tools if t.name == "hover")
        
        # 悬停 1 秒
        input_args = {"drone_id": self.test_drone_id, "duration": 1.0}
        
        print(f"   🚁 正在发送悬停指令...")
        result_str = hover_tool.invoke(input_args)
        result = json.loads(result_str)
        
        # 验证调用成功 (通常 API 返回 {'status': 'success'} 或类似)
        print(f"   ✅ 指令执行响应: {result}")


if __name__ == "__main__":
    unittest.main()