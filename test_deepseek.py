import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# 加载环境变量（需在 .env 中定义 DEEPSEEK_API_KEY 和可选的 DEEPSEEK_BASE_URL）
load_dotenv()

# 配置 DeepSeek API 参数（兼容 OpenAI 格式）
llm = ChatAnthropic(
    model="deepseek-v4-pro",                     # 模型名称，根据实际情况调整（例如 deepseek-chat）
    api_key=os.getenv("ANTHROPIC_API_KEY"),         # 必填，从环境变量获取
    base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic"),  # 默认官方地址
    temperature=0.7,                               # 可选，测试用
)

# 测试 API 可用性
try:
    print("正在测试 DeepSeek API 连接...")
    response = llm.invoke("不要让树漂浮是什么梗")
    print("✅ API 连接成功！")
    print("模型回复：", response.content)
except Exception as e:
    print("❌ API 连接失败，请检查以下配置：")
    print(f"   - API Key 是否正确: {os.getenv("ANTHROPIC_API_KEY") is not None}")
    print(f"   - Base URL 是否可访问: {os.getenv("ANTHROPIC_BASE_URL",  "https://api.deepseek.com/anthropic")}")
    print(f"   - 模型名称是否有效: deepseek-v4-pro")
    print(f"错误详情：{e}")