import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from query_tools import tools

# 自动加载 .env
load_dotenv(find_dotenv())
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_key:
    raise ValueError("请在项目根目录配置 .env 文件中的 DEEPSEEK_API_KEY")

# DeepSeek LLM 初始化
llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    openai_api_key=deepseek_key,
    base_url="https://api.deepseek.com/v1",
    temperature=0.0
)

# 创建Agent（新版原生，无需执行器包装）
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="你是专业竞品分析助手，根据用户资讯需求，调用对应工具提取价格、新品情报，整合清晰结果返回用户"
)

# 测试入口：直接调用 agent.invoke, 输入字段改为 messages
if __name__ == "__main__":
    query = "分析这条资讯里的价格和新品信息：新鲜水果刚到货！现摘现运无存放，西瓜脆甜、葡萄爆汁、芒果软糯，全部产地直销没有中间商！今日特价超划算，多买多优惠，好吃不贵，错过再等一周！新鲜本地西瓜 1 块 5 一斤！阳光葡萄 3 块 8 一斤，黄桃 4 块一斤，全部现摘现卖，汁水饱满，全场买满 20 再减 3 块，多买多划算！时令鲜果特惠开抢！芒果 5 块 2 一斤，脆梨 2 块一斤，精品水蜜桃 6 块一斤，产地直供无中间商，便宜新鲜随便挑！"
    res = agent.invoke({"messages": [("user", query)]})
    print("\n==== 最终分析结果 =====")
    print(res["messages"][-1].content)