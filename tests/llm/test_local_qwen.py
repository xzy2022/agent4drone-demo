import sys
import os
from pathlib import Path

# --- 环境路径设置 ---
# 这一步是为了让 python 能找到 src 目录下的模块
# 获取当前脚本所在目录 (tests/llm)
current_dir = Path(__file__).parent
# 获取项目根目录 (假设 tests 同级目录 src 存在)
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

from src.llm_service import LLMService
from langchain_core.messages import HumanMessage, SystemMessage

def test_qwen_connection():
    # 1. 指定配置文件路径 (假设你在项目根目录运行，或者是绝对路径)
    config_path = project_root / "config" / "llm_config.json"
    
    print(f"📂 读取配置文件: {config_path}")
    
    try:
        # 2. 初始化服务
        # 注意：确保你的 config/llm_config.json 中 "selected_provider" 是 "Ollama"
        llm_service = LLMService(config_path=str(config_path))
        
        # 3. 创建 LLM
        llm = llm_service.create_llm()
        
        # 4. 构造测试消息
        messages = [
            SystemMessage(content="你是一个专业的无人机控制助手。请简短回答。"),
            HumanMessage(content="你好，请介绍一下你自己，并告诉我你能做什么？")
        ]
        
        print("\n🚀 发送请求给 Ollama (Qwen3:8b)...")
        print("-" * 50)
        
        # 5. 调用模型 (使用 invoke)
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