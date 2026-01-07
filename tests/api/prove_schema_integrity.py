# tests\api\prove_schema_integrity.py
import sys
import json
from pathlib import Path

# 环境设置
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from src.uav_api_client import UAVAPIClient
from src2.tools_registry import UAVToolRegistry

def prove_it():
    # 1. 初始化
    client = UAVAPIClient() # 模拟客户端
    registry = UAVToolRegistry(client)
    
    # 2. 获取 take_off 工具
    nav_tools = registry.get_navigation_tools()
    take_off_tool = next(t for t in nav_tools if t.name == "take_off")

    print(f"🛠️  当前检查工具: [{take_off_tool.name}]")
    print(f"🐍 内部函数签名: {take_off_tool.func}") 
    print("   (你看到这里是 **kwargs，没关系，因为 LLM 不看这里)\n")

    # 3. 【核心证明】查看 LangChain 生成的 JSON Schema
    # 这就是 LLM 真正看到的“API 文档”
    schema = take_off_tool.args
    
    print("📜 LLM 看到的参数定义 (由 args_schema 生成):")
    print("=" * 40)
    print(json.dumps(schema, indent=2, ensure_ascii=False))
    print("=" * 40)

    # 4. 验证关键字段是否存在
    if "drone_id" in schema and "altitude" in schema:
        print("\n✅ 验证通过！尽管用了 **kwargs，参数 drone_id 和 altitude 依然清晰可见。")
        print("   LLM 会根据这个 Schema 生成正确的调用参数。")
    else:
        print("\n❌ 验证失败！参数丢失。")

if __name__ == "__main__":
    prove_it()