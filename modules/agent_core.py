"""
Agent 智能调度核心 + 对话记忆管理
支持：自主工具调用、上下文记忆、历史分析缓存
"""
import os
import json
from datetime import datetime
from typing import List, Dict
from langchain.agents import create_agent
from modules.llm_client import get_llm
from modules.tools import tools
from modules.prompts import AGENT_SYSTEM_PROMPT


# ========== 对话记忆 ==========
class ConversationMemory:
    """手动管理对话历史，支持持久化到文件"""

    def __init__(self, persist_path: str = None):
        self.messages: List[Dict] = []
        self.persist_path = persist_path or "./memory/chat_history.json"

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_ai_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def get_all_messages(self) -> List[Dict]:
        return self.messages

    def clear(self):
        self.messages = []

    def save(self):
        """持久化到文件"""
        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "w", encoding="utf-8") as f:
            json.dump({
                "updated_at": datetime.now().isoformat(),
                "messages": self.messages,
            }, f, ensure_ascii=False, indent=2)

    def load(self):
        """从文件恢复"""
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.messages = data.get("messages", [])


# ========== Agent 核心 ==========
class CompetitorAgent:
    """竞品分析智能体，封装 Agent 创建与调用"""

    def __init__(self, model: str = None, with_memory: bool = True):
        self.llm = get_llm(model=model)
        self.memory = ConversationMemory() if with_memory else None
        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
        )

    def chat(self, user_input: str) -> str:
        """单轮对话（无记忆）"""
        res = self.agent.invoke({"messages": [("user", user_input)]})
        return res["messages"][-1].content

    def chat_with_memory(self, user_input: str) -> str:
        """带记忆的多轮对话"""
        if self.memory is None:
            return self.chat(user_input)

        self.memory.add_user_message(user_input)
        res = self.agent.invoke({"messages": self.memory.get_all_messages()})
        ai_content = res["messages"][-1].content
        self.memory.add_ai_message(ai_content)
        return ai_content

    def analyze_news(self, news_text: str) -> Dict[str, str]:
        """
        对一条舆情资讯执行完整的三维度分析
        返回 {price, product, sentiment} 字典
        """
        from modules.tools import extract_price_info, extract_new_product, extract_sentiment

        return {
            "price": extract_price_info.invoke(news_text),
            "product": extract_new_product.invoke(news_text),
            "sentiment": extract_sentiment.invoke(news_text),
        }

    def generate_full_report(self, price: str, product: str, sentiment: str) -> str:
        """生成完整竞争简报"""
        from modules.tools import generate_brief_report
        return generate_brief_report.invoke({
            "price_info": price,
            "product_info": product,
            "sentiment_info": sentiment,
        })


__all__ = ["ConversationMemory", "CompetitorAgent"]
