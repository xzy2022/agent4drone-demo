# src2/commander.py
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from src2.schemas import MissionPlan
from src2.configuration import SystemConfig
from src2.infrastructure import LLMInfrastructure

class NLPCommander:
    def __init__(self, 
                 role: str = "commander", 
                 provider_override: Optional[str] = None,
                 model_override: Optional[str] = None):
        """
        :param role: 角色名称，用于加载 Prompt 和默认配置
        :param provider_override: 强制覆盖 Provider (如从 GUI 传入)
        :param model_override: 强制覆盖 Model (如从 GUI 传入 "qwen2.5:1.5b")
        """
        self.config = SystemConfig()
        infra = LLMInfrastructure(self.config)
        
        # 1. 获取角色默认配置
        agent_settings = self.config.get_agent_settings(role)
        
        # 2. 决定 Provider (参数覆盖 > 角色配置 > 默认 Ollama)
        target_provider = provider_override or agent_settings.get("preferred_provider", "Ollama")
        
        # 3. 决定 Model (参数覆盖 > 角色配置 > Provider 默认(由infra处理))
        # 允许在 agent 配置里指定 "model": "qwen2.5:0.5b"
        target_model = model_override or agent_settings.get("model")
        
        # 4. 决定 Temperature
        target_temp = agent_settings.get("temperature_override")

        # 5. 创建实例
        self.llm = infra.create_llm(
            provider_name=target_provider,
            model_name=target_model,  # <--- 关键：透传模型名
            temperature=target_temp
        )
        
        # ...加载 Prompt 和 Parser (保持不变)...
        self.system_prompt = self.config.get_agent_prompt(role)
        self.parser = PydanticOutputParser(pydantic_object=MissionPlan)

    def generate_plan(self, user_command: str) -> MissionPlan:
        """
        核心方法：自然语言 -> Pydantic 对象
        """
        print(f"🧠 Planner receiving: '{user_command}'")
        try:
            # invoke 会直接返回一个 MissionPlan 实例
            plan = self.chain.invoke({"input": user_command})
            return plan
        except Exception as e:
            print(f"❌ Planning failed: {e}")
            # 返回空计划或抛出异常
            return MissionPlan(mission_steps=[])