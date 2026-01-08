# src2/infrastructure.py
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from src2.configuration import SystemConfig

class LLMInfrastructure:
    def __init__(self, config: SystemConfig):
        self.config = config

    def create_llm(self, 
                   provider_name: str, 
                   model_name: Optional[str] = None, 
                   temperature: Optional[float] = None) -> BaseChatModel:
        """
        创建 LLM 实例
        :param provider_name: 对应配置文件中的 key (如 "Ollama")
        :param model_name: 强制指定模型名称 (如 "qwen2.5:1.5b")，如果不传则使用配置文件的默认值
        :param temperature: 覆盖温度设置
        """
        # 获取基础连接配置 (Base URL, API Key, Type)
        # 注意：这里用 copy 防止修改缓存的配置
        llm_conf = self.config.get_llm_config(provider_name).copy()
        
        # === 核心逻辑：参数优先级 ===
        # 运行时参数 > 配置文件默认值
        final_model = model_name if model_name else llm_conf.get("model")
        final_temp = temperature if temperature is not None else llm_conf.get("temperature", 0.1)

        llm_type = llm_conf.get("type", "").lower()
        
        print(f"🏭 Init LLM: [{provider_name}] Model=[{final_model}] Temp=[{final_temp}]")

        if llm_type == "ollama":
            return ChatOllama(
                base_url=llm_conf.get("base_url"),
                model=final_model,  # 使用最终决定的模型名
                temperature=final_temp
            )
        elif llm_type == "openai":
            return ChatOpenAI(
                base_url=llm_conf.get("base_url"),
                api_key=llm_conf.get("api_key"),
                model=final_model,  # 使用最终决定的模型名
                temperature=final_temp,
                max_retries=2
            )
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")