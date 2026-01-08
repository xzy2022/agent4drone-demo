# src2/configuration.py
import os
import json
import re
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class SystemConfig:
    """全局配置中心：管理 LLM 连接、Agent 角色设定和 Prompt"""

    def __init__(self, config_path: str = "config/llm_config.json"):
        self.root_path = Path(__file__).resolve().parent.parent
        self.config_path = self.root_path / config_path
        self._raw_config = self._load_json()

    def _load_json(self) -> Dict[str, Any]:
        """加载 JSON 并注入环境变量"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用正则替换 ${VAR} 为环境变量
        def replace_env(match):
            var_name = match.group(1)
            return os.getenv(var_name, "")
        
        content = re.sub(r'\$\{(.+?)\}', replace_env, content)
        return json.loads(content)

    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """获取特定 LLM 提供商的连接参数"""
        providers = self._raw_config.get("providers", {})
        if provider not in providers:
            raise ValueError(f"Provider {provider} not found.")
        return providers[provider]

    # --- 关键：Prompt 与角色管理 ---
    def get_agent_prompt(self, role: str) -> str:
        """
        根据角色获取 System Prompt。
        支持从 JSON 读取，或者未来扩展为读取 src2/prompts/ 目录下的 .txt 文件
        """
        # 方案 1：直接在 llm_config.json 里定义 (适合短 Prompt)
        agents_config = self._raw_config.get("agents", {})
        if role in agents_config:
            return agents_config[role].get("system_prompt", "")
            
        # 方案 2 (推荐)：如果 Prompt 很长，从文件读取
        # prompt_path = self.root_path / f"src2/prompts/{role}.txt"
        # if prompt_path.exists():
        #     return prompt_path.read_text(encoding='utf-8')
            
        return "You are a helpful AI assistant."

    def get_agent_settings(self, role: str) -> Dict[str, Any]:
        """获取特定 Agent 的额外配置 (如温度、最大重试次数等)"""
        return self._raw_config.get("agents", {}).get(role, {})