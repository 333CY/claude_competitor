import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from query_tools import tools

load_dotenv(dotenv_path="../.env")

llm = ChatOpenAI(
    model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.0,
)

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "你是一位资深的竞品分析专家。你的任务是根据用户提供的舆情资讯，"
        "调用工具提取价格和新品信息，然后基于这些信息进行综合分析。"
        "请以清晰、有条理的方式回答。"
    ),
)


class ConversationMemory:
    """手动管理对话历史，以消息列表形式存储，供 Agent 上下文使用"""

    def __init__(self):
        self.messages = []

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_ai_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def get_all_messages(self):
        """返回完整的历史消息列表，可直接传入 agent.invoke"""
        return self.messages


# 实例化记忆
memory = ConversationMemory()


def chat_with_agent(user_input: str) -> str:
    """接收用户输入，调用 Agent，并将交互存入记忆"""
    # 1) 记录用户输入
    memory.add_user_message(user_input)

    # 2) 调用 Agent（传入全部历史消息）
    response = agent.invoke({"messages": memory.get_all_messages()})

    # 3) 提取 AI 回复内容
    ai_msg = response["messages"][-1]
    ai_content = ai_msg.content if hasattr(ai_msg, "content") else ai_msg.get("content", "")

    # 4) 记录 AI 回复
    memory.add_ai_message(ai_content)

    return ai_content


if __name__ == "__main__":
    print("=" * 60)
    print("竞品分析助手（带记忆）已启动...")
    print("=" * 60 + "\n")

    user_input_1 = "分析资讯：新鲜水果刚到货！现摘现运无存放，西瓜脆甜、葡萄爆汁、芒果软糯，全部产地直销没有中间商！今日特价超划算，多买多优惠，好吃不贵，错过再等一周！新鲜本地西瓜 1 块 5 一斤！阳光葡萄 3 块 8 一斤，黄桃 4 块一斤，全部现摘现卖，汁水饱满，全场买满 20 再减 3 块，多买多划算！时令鲜果特惠开抢！芒果 5 块 2 一斤，脆梨 2 块一斤，精品水蜜桃 6 块一斤，产地直供无中间商，便宜新鲜随便挑！"
    print(f"用户: {user_input_1}")
    answer_1 = chat_with_agent(user_input_1)
    print(f"助手: {answer_1}\n")
    print("-" * 60 + "\n")

    user_input_2 = "结合刚才的资讯，总结竞品带来的市场威胁"
    print(f"用户: {user_input_2}")
    answer_2 = chat_with_agent(user_input_2)
    print(f"助手: {answer_2}\n")
    print("-" * 60 + "\n")

    print("== 完整对话历史 ==")
    for msg in memory.get_all_messages():
        print(f"{msg['role'].upper()}: {msg['content'][:150]}...")
    print("=" * 60)