# tests\llm\test_context.py
import sys
import os
import json
import logging
import unittest
from pathlib import Path

# --- 环境路径设置 ---
# 确保能找到 src 目录和 uav_api_client
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
from src.uav_api_client import UAVAPIClient
from src.context_manager import DroneContextManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ContextTest")

class TestSystemContext(unittest.TestCase):
    
    def setUp(self):
        """
        初始化真实环境连接
        """
        print("\n" + "="*60)
        self.base_url = "http://localhost:8000"
        
        # 1. 实例化真实的 API 客户端
        try:
            self.real_client = UAVAPIClient(self.base_url)
            # 测试一下连接，获取列表看服务是否活着
            self.real_client.list_drones()
            logger.info(f"✅ 成功连接到 UAV Server: {self.base_url}")
        except Exception as e:
            self.skipTest(f"❌ 无法连接到 UAV Server ({self.base_url})，跳过测试。\n错误: {e}")

        # 2. 初始化上下文管理器
        self.context_manager = DroneContextManager(self.real_client)

    def test_context_generation(self):
        """
        测试仅获取上下文：
        1. 刷新环境感知 (获取真实无人机列表)
        2. 打印映射表 (Name -> ID)
        3. 生成并展示 System Prompt
        """
        
        # === Step 1: 刷新环境上下文 ===
        logger.info("STEP 1: 正在刷新环境上下文 (Refresh Context)...")
        self.context_manager.refresh()
        
        # 获取当前的映射表
        drone_map = self.context_manager.drone_map
        
        print("\n--- [当前感知到的无人机映射] ---")
        print(json.dumps(drone_map, indent=2, ensure_ascii=False))

        # 简单的断言，如果是空环境也没关系，只是打印出来
        if not drone_map:
            logger.warning("⚠️ 注意: 环境中当前没有发现任何无人机 (列表为空)。")
        else:
            logger.info(f"✅ 发现 {len(drone_map)} 架无人机。")

        # === Step 2: 验证 System Prompt ===
        logger.info("STEP 2: 生成 System Prompt (上下文注入内容)")
        system_prompt = self.context_manager.get_system_prompt_context()
        
        print("\n" + "="*20 + " SYSTEM PROMPT PREVIEW " + "="*20)
        print(system_prompt)
        print("="*20 + " END OF PROMPT " + "="*20 + "\n")

        # 验证返回类型
        self.assertIsInstance(system_prompt, str, "System Prompt 应该是一个字符串")
        self.assertTrue(len(system_prompt) > 0, "System Prompt 不应为空")

        # 如果有无人机，验证 prompt 里是否包含无人机的名字
        if drone_map:
            first_drone_name = list(drone_map.keys())[0]
            self.assertIn(first_drone_name, system_prompt, f"Prompt 中应包含无人机名称 '{first_drone_name}'")
            logger.info("✅ 验证通过: 无人机名称已正确注入到 Prompt 中。")

if __name__ == "__main__":
    unittest.main()