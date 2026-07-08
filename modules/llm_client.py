"""
LLM 客户端封装 - DeepSeek API（OpenAI 兼容接口）
统一管理模型初始化、参数配置，供所有模块复用
"""
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(find_dotenv())

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请在 .env 中配置 DEEPSEEK_API_KEY")

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")


def get_llm(model: str = None, temperature: float = 0.0, max_tokens: int = 4096):
    """
    获取 DeepSeek LLM 实例
    Args:
        model: 模型名称，默认 DEEPSEEK_MODEL 环境变量或 deepseek-chat
        temperature: 温度参数，默认 0.0（消除幻觉）
        max_tokens: 最大输出 token
    """
    if model is None:
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    return ChatOpenAI(
        model=model,
        openai_api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# 预定义三层模型分工
def get_haiku():
    """轻量模型：检索、关键词提取"""
    return get_llm(model=os.getenv("MODEL_LIGHT", "deepseek-chat"), temperature=0.0)


def get_sonnet():
    """中量模型：单维度结构化分析"""
    return get_llm(model=os.getenv("MODEL_MEDIUM", "deepseek-chat"), temperature=0.0)


def get_opus():
    """重量模型：整合多维度简报生成"""
    return get_llm(model=os.getenv("MODEL_HEAVY", "deepseek-chat"), temperature=0.0)
