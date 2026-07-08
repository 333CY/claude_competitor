"""
Day 3 - 任务 C: Memory + 网络爬取集成
使用项目 ConversationMemory + requests/BeautifulSoup 网页抓取
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from modules.llm_client import get_llm
from modules.agent_core import ConversationMemory

llm = get_llm(temperature=0.0)


# ========== 1. ConversationBufferMemory 多轮对话 ==========
def demo_conversation_memory():
    print("=" * 50)
    print("1. ConversationMemory 多轮对话")
    print("=" * 50)

    memory = ConversationMemory()

    prompt = PromptTemplate.from_template(
        "你是一个竞品分析助手。\n对话历史：{history}\n用户问题：{input}\n请简洁回答："
    )

    def chat(user_input: str) -> str:
        history_str = "\n".join(
            f"{m['role']}: {m['content'][:100]}"
            for m in memory.get_all_messages()
        ) if memory.get_all_messages() else "（首次对话）"

        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"history": history_str, "input": user_input})
        memory.add_user_message(user_input)
        memory.add_ai_message(result)
        return result

    q1 = "竞品A今天宣布降价15%"
    a1 = chat(q1)
    print(f"用户: {q1}")
    print(f"助手: {a1[:200]}...\n")

    q2 = "这对我们有什么威胁？"
    a2 = chat(q2)
    print(f"用户: {q2}")
    print(f"助手: {a2[:200]}...\n")

    print(f"记忆条数: {len(memory.get_all_messages())}")
    print("✅ 对话记忆通过\n")


# ========== 2. 网页抓取 + 摘要生成 ==========
def demo_web_crawl():
    print("=" * 50)
    print("2. 网页抓取 + LLM 摘要生成")
    print("=" * 50)

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        print("⚠️ 需要安装: pip install requests beautifulsoup4")
        return

    # 测试 URL（使用一个稳定页面）
    url = "https://www.python.org/downloads/"
    print(f"◆ 抓取: {url}")

    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # 提取标题和正文
        title = soup.title.string if soup.title else "无标题"
        body = soup.get_text(separator="\n", strip=True)[:2000]

        print(f"  标题: {title}")
        print(f"  正文长度: {len(body)} 字符")

        # LLM 摘要
        summary_prompt = PromptTemplate.from_template(
            "请用一句话总结以下网页的核心内容：\n标题：{title}\n内容片段：{body}"
        )
        chain = summary_prompt | llm | StrOutputParser()
        summary = chain.invoke({"title": title, "body": body})
        print(f"  AI 摘要: {summary}")
        print("✅ 爬虫+摘要通过\n")

    except Exception as e:
        print(f"  ⚠️ 网络请求失败: {e}")
        print("  （使用模拟数据演示流程）\n")

        # 降级：模拟数据
        mock_html = "<html><title>竞品B发布2026新品</title><body>竞品B发布2026款旗舰，售价4999元起，新增AI功能，用户投诉续航问题</body></html>"
        soup = BeautifulSoup(mock_html, "html.parser")
        title = soup.title.string
        body = soup.get_text()[:500]
        print(f"  [模拟] 标题: {title}")

        summary_prompt = PromptTemplate.from_template(
            "用一句话总结：\n标题：{title}\n内容：{body}"
        )
        chain = summary_prompt | llm | StrOutputParser()
        summary = chain.invoke({"title": title, "body": body})
        print(f"  [模拟] AI 摘要: {summary}")
        print("✅ 降级方案通过\n")


# ========== 3. Memory + 爬虫联动 ==========
def demo_memory_crawl():
    print("=" * 50)
    print("3. Memory + 爬虫联动（多轮情报追踪）")
    print("=" * 50)

    memory = ConversationMemory()

    prompt = PromptTemplate.from_template(
        "竞品分析助手。历史：{history}\n当前：{input}\n请分析："
    )

    def analyze(user_input: str) -> str:
        history_str = "\n".join(
            f"{m['role']}: {m['content'][:100]}"
            for m in memory.get_all_messages()[-4:]
        ) if memory.get_all_messages() else "（首次）"
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"history": history_str, "input": user_input})
        memory.add_user_message(user_input)
        memory.add_ai_message(result)
        return result

    print("◆ 第一轮：输入竞品降价新闻")
    a1 = analyze("竞品A旗舰降价20%至999元，同时推出订阅制每月29元")
    print(f"  分析: {a1[:200]}...\n")

    print("◆ 第二轮：无需重复，直接追问")
    a2 = analyze("我们该怎么应对？")
    print(f"  建议: {a2[:200]}...\n")

    print(f"记忆条数: {len(memory.get_all_messages())}")
    print("✅ 记忆联动通过\n")


if __name__ == "__main__":
    print("\n🚀 Day 3 Memory + 爬虫演示\n")

    # 先确保依赖
    try:
        import requests, bs4
    except ImportError:
        print("⚠️ 安装依赖: pip install requests beautifulsoup4\n")

    demo_conversation_memory()
    demo_web_crawl()
    demo_memory_crawl()

    print("=" * 50)
    print("Day 3 Memory + 爬虫演示完成 ✅")
    print("=" * 50)
