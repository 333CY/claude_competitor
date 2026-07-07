from langchain.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

# 读取密钥并校验
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("未读取到 DEEPSEEK_API_KEY，请检查.env文件配置")

# 初始化DeepSeek LLM
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    openai_api_key=api_key,
    base_url="https://api.deepseek.com/",
    temperature=0.0,
)

# 工具1：提取价格信息
@tool
def extract_price_info(news_text: str) -> str:
    """
    从舆情文本提取竞品价格、促销、降价涨价等价格相关信息
    Args:
        news_text: 行业新闻、论坛、电商舆情原文文本
    """
    prompt = f"""仅提取文本中所有价格、折扣、促销、调价相关信息，只返回提取结果，不要多余描述：
文本内容: {news_text}
"""
    res = llm.invoke(prompt)
    return res.content.strip()

# 工具2：提取新品信息
@tool
def extract_new_product(news_text: str) -> str:
    """
    从舆情文本提取竞品新品、新机型、新增功能、新品上市相关信息
    Args:
        news_text: 行业新闻、论坛、电商舆情原文文本
    """
    prompt = f"""仅提取文本中新品、新款、新功能、新品上市相关信息，只返回提取结果，不要多余描述：
文本内容: {news_text}
"""
    res = llm.invoke(prompt)
    return res.content.strip()

# 工具集合，供Agent绑定使用
tools = [extract_price_info, extract_new_product]

# 本地测试入口
if __name__ == "__main__":
    test_text = "竞品A全线降价10%，同时推出新款产品搭载智能温控功能，新品首发价299元"
    print("=== 价格提取结果 ===")
    print(extract_price_info.invoke(test_text))
    print("\n=== 新品提取结果 ===")
    print(extract_new_product.invoke(test_text))