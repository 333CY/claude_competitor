"""
Day 1: API 连通性测试 & 基础模块验证
测试 DeepSeek API 连接 + LLM 客户端封装
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from modules.llm_client import get_llm
from modules.prompts import PRICE_EXTRACTION_PROMPT, AGENT_SYSTEM_PROMPT
from modules.data_loader import load_text_files

# ========== 测试 1: LLM 连接 ==========
def test_llm_connection():
    print("=" * 50)
    print("测试 1: DeepSeek API 连接")
    print("=" * 50)
    llm = get_llm()
    try:
        res = llm.invoke("简单介绍竞品情报分析的作用（一句话）")
        print("✅ API 连接成功！")
        print(f"回复: {res.content}\n")
    except Exception as e:
        print(f"❌ API 连接失败: {e}\n")
        return False
    return True


# ========== 测试 2: Prompt 模板 ==========
def test_prompt_template():
    print("=" * 50)
    print("测试 2: 价格提取 Prompt")
    print("=" * 50)
    llm = get_llm()
    test_text = "竞品A全线降价10%，标准版从999元降至899元"
    prompt = PRICE_EXTRACTION_PROMPT.format(news_text=test_text)
    try:
        res = llm.invoke(prompt)
        print("✅ 价格提取成功！")
        print(f"提取结果: {res.content}\n")
    except Exception as e:
        print(f"❌ 提取失败: {e}\n")


# ========== 测试 3: 数据加载 ==========
def test_data_loader():
    print("=" * 50)
    print("测试 3: 数据加载模块")
    print("=" * 50)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
    try:
        docs = load_text_files(data_dir)
        print(f"✅ 数据加载成功！加载 {len(docs)} 篇文档")
        for doc in docs:
            print(f"  - {doc.metadata.get('filename')}: {doc.page_content[:50]}...")
    except FileNotFoundError:
        print(f"⚠️  data/raw 目录为空，跳过（正常，待后续灌入数据）")
    print()


if __name__ == "__main__":
    print("\n🚀 Day 1 模块验证开始\n")
    test_llm_connection()
    test_prompt_template()
    test_data_loader()
    print("=" * 50)
    print("Day 1 验证完成 ✅")
    print("=" * 50)
