import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

#加载密钥
load_dotenv()
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    api_key=os.getenv("ANTHROPIC_API_KEY")
    #中转平台追加 base_url=os.getenv("ANTHROPIC_BASE_URL")
)

#简单测试
res = llm.invoke("简单介绍竞品情报分析的作用")
print(res.content)