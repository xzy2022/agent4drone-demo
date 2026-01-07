import os
import json
import copy
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


class LLMService:
    def __init__(self, config_path: str = "config/llm_config.json"):
        """
        初始化 LLM 服务
        :param config_path: 配置文件路径
        """
        load_dotenv() # 加载 .env
        self.config_path = Path(config_path)
        self.full_config = self._load_config()
        self.active_config = self._get_active_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {self.config_path.absolute()}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_active_config(self) -> Dict[str, Any]:
        """获取并解析当前选中 provider 的配置"""
        selected_name = self.full_config.get("selected_provider")
        if not selected_name:
            raise ValueError("配置文件中缺少 'selected_provider' 字段")
        
        provider_config = self.full_config.get("providers", {}).get(selected_name)
        if not provider_config:
            raise ValueError(f"未找到 provider: {selected_name} 的配置")

        # 处理配置（深拷贝以防修改原字典）
        config = copy.deepcopy(provider_config)
        
        # 核心：替换环境变量占位符
        api_key = config.get("api_key")
        if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            real_key = os.getenv(env_var)
            if not real_key and config.get("type") == "openai":
                print(f"⚠️ 警告: 环境变量 {env_var} 未设置")
            config["api_key"] = real_key

        return config

    def create_llm(self):
        """
        创建并返回 LangChain 的 ChatModel 实例
        """
        conf = self.active_config
        llm_type = conf.get("type", "").lower()
        model_name = conf.get("model")
        temperature = conf.get("temperature", 0.1)

        print(f"🔄 初始化 LLM: [{llm_type}] {model_name}")

        if llm_type == "ollama":
            return ChatOllama(
                base_url=conf.get("base_url", "http://localhost:11434"),
                model=model_name,
                temperature=temperature
            )
        
        elif llm_type == "openai":
            return ChatOpenAI(
                base_url=conf.get("base_url"),
                api_key=conf.get("api_key"),
                model=model_name,
                temperature=temperature
            )
        
        else:
            raise ValueError(f"不支持的 LLM 类型: {llm_type}")

# 方便外部直接调用的单例模式（可选）
if __name__ == "__main__":
    # 简单的自测
    service = LLMService()
    llm = service.create_llm()
    print("LLM 对象创建成功:", llm)