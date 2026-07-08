"""
Agent 工具集
封装竞品分析所需的独立工具：价格提取、新品提取、舆情提取、简报生成
"""
from langchain.tools import tool
from modules.llm_client import get_llm
from modules.prompts import (
    PRICE_EXTRACTION_PROMPT,
    NEW_PRODUCT_EXTRACTION_PROMPT,
    SENTIMENT_EXTRACTION_PROMPT,
    BRIEF_REPORT_PROMPT,
    SWOT_ANALYSIS_PROMPT,
)

_llm = get_llm()


# ========== 工具1：提取价格信息 ==========
@tool
def extract_price_info(news_text: str) -> str:
    """
    从舆情文本提取竞品价格、促销、降价涨价等价格相关信息
    Args:
        news_text: 行业新闻、论坛、电商舆情原文文本
    """
    prompt = PRICE_EXTRACTION_PROMPT.format(news_text=news_text)
    res = _llm.invoke(prompt)
    return res.content.strip()


# ========== 工具2：提取新品信息 ==========
@tool
def extract_new_product(news_text: str) -> str:
    """
    从舆情文本提取竞品新品、新机型、新增功能、新品上市相关信息
    Args:
        news_text: 行业新闻、论坛、电商舆情原文文本
    """
    prompt = NEW_PRODUCT_EXTRACTION_PROMPT.format(news_text=news_text)
    res = _llm.invoke(prompt)
    return res.content.strip()


# ========== 工具3：提取舆情信息 ==========
@tool
def extract_sentiment(news_text: str) -> str:
    """
    从舆情文本提取用户评价、投诉、口碑、舆情风险相关信息
    Args:
        news_text: 行业新闻、论坛、电商舆情原文文本
    """
    prompt = SENTIMENT_EXTRACTION_PROMPT.format(news_text=news_text)
    res = _llm.invoke(prompt)
    return res.content.strip()


# ========== 工具4：生成竞争简报 ==========
@tool
def generate_brief_report(price_info: str, product_info: str, sentiment_info: str) -> str:
    """
    整合价格、新品、舆情三维度信息，生成竞品竞争态势简报
    Args:
        price_info: 价格维度提取结果
        product_info: 新品维度提取结果
        sentiment_info: 舆情维度提取结果
    """
    prompt = BRIEF_REPORT_PROMPT.format(
        price_info=price_info,
        product_info=product_info,
        sentiment_info=sentiment_info,
    )
    res = _llm.invoke(prompt)
    return res.content.strip()


# ========== 工具5：SWOT 分析 ==========
@tool
def swot_analysis(context: str) -> str:
    """
    基于竞品情报进行 SWOT 分析
    Args:
        context: 整合的竞品情报文本
    """
    prompt = SWOT_ANALYSIS_PROMPT.format(context=context)
    res = _llm.invoke(prompt)
    return res.content.strip()


# 工具集合
tools = [
    extract_price_info,
    extract_new_product,
    extract_sentiment,
    generate_brief_report,
    swot_analysis,
]

__all__ = [
    "extract_price_info",
    "extract_new_product",
    "extract_sentiment",
    "generate_brief_report",
    "swot_analysis",
    "tools",
]
