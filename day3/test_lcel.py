"""
Day 3 - 任务 A: LCEL 链式处理演示
PromptTemplate → LLMChain → RunnableSequence 进阶
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from modules.llm_client import get_llm

llm = get_llm(temperature=0.0)


# ========== 1. PromptTemplate + StrOutputParser ==========
def demo_basic_chain():
    print("=" * 50)
    print("1. 基础 PromptTemplate → LLM → StrOutputParser")
    print("=" * 50)

    template = """
你是一个竞品情报分析师。请根据以下信息，生成一份简洁的竞品分析摘要。
产品名称：{product}
竞争对手：{competitor}
关键差异点：{key_points}
"""
    prompt = PromptTemplate(
        input_variables=["product", "competitor", "key_points"],
        template=template,
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "product": "DeepSeek大模型",
        "competitor": "主流开源大模型",
        "key_points": "推理延迟低、128K长上下文、API价格仅为竞品1/10",
    })
    print(result[:400])
    print("✅ 基础链通过\n")


# ========== 2. RunnableParallel 并行提取 ==========
def demo_parallel_chain():
    print("=" * 50)
    print("2. RunnableParallel 并行三维度分析")
    print("=" * 50)

    from modules.prompts import (
        PRICE_EXTRACTION_PROMPT,
        NEW_PRODUCT_EXTRACTION_PROMPT,
        SENTIMENT_EXTRACTION_PROMPT,
    )

    price_prompt = PromptTemplate.from_template(PRICE_EXTRACTION_PROMPT)
    product_prompt = PromptTemplate.from_template(NEW_PRODUCT_EXTRACTION_PROMPT)
    sentiment_prompt = PromptTemplate.from_template(SENTIMENT_EXTRACTION_PROMPT)

    parallel_chain = RunnableParallel(
        price=price_prompt | llm | StrOutputParser(),
        product=product_prompt | llm | StrOutputParser(),
        sentiment=sentiment_prompt | llm | StrOutputParser(),
    )

    news = "竞品X发布2026旗舰，售价4999元较上代降500元，新增AI功能。电商差评率升至15%，用户投诉续航缩水。"
    result = parallel_chain.invoke({"news_text": news})
    for k, v in result.items():
        print(f"--- {k} ---\n{v[:200]}...\n")
    print("✅ 并行链通过\n")


# ========== 3. RunnableLambda 预处理 + RunnableSequence ==========
def demo_lambda_chain():
    print("=" * 50)
    print("3. RunnableLambda 预处理 + RunnableSequence")
    print("=" * 50)

    # 关键词提取函数
    def extract_keywords(text: str) -> str:
        return text[:80] + ("..." if len(text) > 80 else "")

    # JSON 分析模板
    json_parser = JsonOutputParser()
    analysis_template = """
基于以下竞品情报关键词，进行结构化分析：
关键词：{keywords}

{format_instructions}
"""
    analysis_prompt = PromptTemplate(
        template=analysis_template,
        input_variables=["keywords"],
        partial_variables={"format_instructions": json_parser.get_format_instructions()},
    )

    # 复合链：预处理 → 填充提示词 → LLM → JSON
    chain = (
        RunnableLambda(lambda x: {"keywords": extract_keywords(str(x))})
        | analysis_prompt
        | llm
        | json_parser
    )

    raw_info = "竞品Y在Q2财报中宣布全面降价20%，同时推出新一代智能家居产品线，搭载自研芯片，价格较上代降低30%，但用户反馈产品质量下滑，售后投诉激增。"
    result = chain.invoke(raw_info)
    print(f"结构化输出: {result}")
    print("✅ Lambda 链通过\n")


# ========== 4. RAG 链（检索 + 生成） ==========
def demo_rag_chain():
    print("=" * 50)
    print("4. RAG 检索生成链")
    print("=" * 50)

    # 模拟文档
    from langchain_core.documents import Document
    docs = [
        Document(page_content="竞品A在2026年7月宣布降价15%，旗舰从1299元降至1104元"),
        Document(page_content="竞品A推出新款入门产品，定价299元，目标抢占下沉市场"),
        Document(page_content="行业分析师认为竞品A的降价是应对市场份额下滑的防御性策略"),
    ]

    # 用 | 拼接检索结果
    retriever = lambda q: docs  # 模拟检索

    summary_prompt = PromptTemplate.from_template(
        "根据以下情报，生成一段竞品动态摘要（50字以内）：\n{context}"
    )

    chain = (
        {"context": RunnableLambda(lambda x: "\n".join(d.page_content for d in retriever(x["query"])))}
        | summary_prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke({"query": "竞品A"})
    print(f"摘要: {result}")
    print("✅ RAG 链通过\n")


if __name__ == "__main__":
    print("\n🚀 Day 3 LCEL 链式处理演示\n")
    demo_basic_chain()
    demo_parallel_chain()
    demo_lambda_chain()
    demo_rag_chain()
    print("=" * 50)
    print("Day 3 LCEL 演示全部通过 ✅")
    print("=" * 50)
