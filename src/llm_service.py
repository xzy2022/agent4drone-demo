# src\llm_service.py
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
        load_dotenv()  # 加载 .env
        self.config_path = Path(config_path)
        self.raw_config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件未找到: {self.config_path.absolute()}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _process_config(self, provider_name: str) -> Dict[str, Any]:
        """提取指定 Provider 的配置并处理环境变量"""
        providers = self.raw_config.get("providers", {})
        provider_config = providers.get(provider_name)
        
        if not provider_config:
            valid_keys = list(providers.keys())
            raise ValueError(f"未找到 Provider '{provider_name}' 的配置。可用选项: {valid_keys}")

        # 深拷贝以防修改原字典
        config = copy.deepcopy(provider_config)
        
        # 替换环境变量占位符
        api_key = config.get("api_key")
        if isinstance(api_key, str) and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            real_key = os.getenv(env_var)
            if not real_key and config.get("type") == "openai":
                print(f"⚠️ 警告: 环境变量 {env_var} 未设置，OpenAI 兼容接口可能调用失败")
            config["api_key"] = real_key

        return config

    def create_llm(self, provider_name: str, override_temperature: Optional[float] = None):
        """
        根据 provider_name 创建 LangChain 实例
        :param provider_name: 对应配置文件中 providers 下的 key (如 "Ollama", "DeepSeek")
        :param override_temperature: 可选，覆盖配置文件中的温度
        """
        conf = self._process_config(provider_name)
        
        llm_type = conf.get("type", "").lower()
        model_name = conf.get("model")
        temperature = override_temperature if override_temperature is not None else conf.get("temperature", 0.1)

        print(f"🔄 初始化 LLM: Provider=[{provider_name}] Type=[{llm_type}] Model=[{model_name}]")

        if llm_type == "ollama":
            from langchain_ollama import ChatOllama
            return ChatOllama(
                base_url=conf.get("base_url", "http://localhost:11434"),
                model=model_name,
                temperature=temperature
            )
        
        elif llm_type == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOllama(
                base_url=conf.get("base_url"),
                model=model_name,
                temperature=temperature,
                # format="json",  # 强制 Ollama 输出 JSON (需模型支持，如 Llama3, Mistral)
                num_predict=200, # 限制最大 token 数，防止 100s 的生成
            )


        else:
            raise ValueError(f"不支持的 LLM 类型: {llm_type}")

if __name__ == "__main__":
    # 自测
    try:
        service = LLMService()
        # 可以在这里随意切换 "Ollama" 或 "DeepSeek"
        llm = service.create_llm("Ollama")
        print("✅ LLM 对象创建成功:", llm)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")