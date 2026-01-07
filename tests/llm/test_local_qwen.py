import sys
import os
from pathlib import Path

# --- 环境路径设置 ---
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.llm_service import LLMService
from langchain_core.messages import HumanMessage, SystemMessage

def test_qwen_connection():
    config_path = project_root / "config" / "llm_config.json"
    print(f"📂 读取配置文件: {config_path}")
    
    try:
        # 1. 初始化服务
        llm_service = LLMService(config_path=str(config_path))
        
        # 2. 显式指定要使用的 Provider
        target_provider = "Ollama" 
        # 如果你想测 DeepSeek，只需改为: target_provider = "DeepSeek"
        
        print(f"👉 请求创建 Provider: {target_provider}")
        llm = llm_service.create_llm(target_provider)
        
        # 3. 构造测试消息
        messages = [
            SystemMessage(content="你是一个专业的无人机控制助手。请简短回答。"),
            HumanMessage(content="你好，请介绍一下你自己。")
        ]
        
        print(f"\n🚀 发送请求给 {target_provider}...")
        print("-" * 50)
        
        # 4. 调用模型
        response = llm.invoke(messages)
        
        print(response.content)
        print("-" * 50)
        print("✅ 测试成功！LLM 通信正常。")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_qwen_connection()