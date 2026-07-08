"""
Day 2: RAG + Agent + Tools + Memory 集成验证
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# ========== 测试 1: 工具集 ==========
def test_tools():
    print("=" * 50)
    print("测试 1: Agent 工具集")
    print("=" * 50)
    from modules.tools import extract_price_info, extract_new_product, extract_sentiment

    test_text = "竞品A全线降价10%，标准版从999元降至899元，同时推出新款智能温控产品，但用户投诉售后响应慢"

    print("◆ 价格提取:")
    print(extract_price_info.invoke(test_text)[:200])
    print("\n◆ 新品提取:")
    print(extract_new_product.invoke(test_text)[:200])
    print("\n◆ 舆情提取:")
    print(extract_sentiment.invoke(test_text)[:200])
    print("✅ 工具集测试通过\n")


# ========== 测试 2: Agent 自主调度 ==========
def test_agent():
    print("=" * 50)
    print("测试 2: Agent 自主决策")
    print("=" * 50)
    from modules.agent_core import CompetitorAgent

    agent = CompetitorAgent(with_memory=False)
    query = "分析这条资讯里的价格和新品信息：竞品A全线降价10%，发布新款智能机型，首发价299元"
    result = agent.chat(query)
    print(f"Agent 回复: {result[:300]}...")
    print("✅ Agent 测试通过\n")


# ========== 测试 3: Agent + 记忆多轮对话 ==========
def test_agent_memory():
    print("=" * 50)
    print("测试 3: Agent 多轮对话记忆")
    print("=" * 50)
    from modules.agent_core import CompetitorAgent

    agent = CompetitorAgent(with_memory=True)

    q1 = "竞品A今天宣布全线降价15%，旗舰产品从1299元降至1104元"
    print(f"◆ 用户: {q1}")
    a1 = agent.chat_with_memory(q1)
    print(f"  助手: {a1[:200]}...\n")

    q2 = "结合刚才的信息，这对我们有什么威胁？"
    print(f"◆ 用户: {q2}")
    a2 = agent.chat_with_memory(q2)
    print(f"  助手: {a2[:200]}...\n")

    print(f"◆ 记忆中共 {len(agent.memory.get_all_messages())} 条消息")
    print("✅ 记忆测试通过\n")


# ========== 测试 4: 三维度分析 ==========
def test_full_analysis():
    print("=" * 50)
    print("测试 4: 三维度完整分析")
    print("=" * 50)
    from modules.agent_core import CompetitorAgent

    agent = CompetitorAgent()
    news = "竞品B发布2026款旗舰产品，售价4999元起，较上代降价500元，新增AI助手功能。但大量用户反映续航缩水、发热严重，电商平台差评率升至15%。"
    result = agent.analyze_news(news)
    for k, v in result.items():
        print(f"--- {k} ---\n{v[:200]}...\n")
    print("✅ 三维度分析通过\n")


if __name__ == "__main__":
    print("\n🚀 Day 2 模块验证开始\n")
    test_tools()
    test_agent()
    test_agent_memory()
    test_full_analysis()
    print("=" * 50)
    print("Day 2 验证完成 ✅")
    print("=" * 50)
