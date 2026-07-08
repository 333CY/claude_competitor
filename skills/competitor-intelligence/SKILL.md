---
name: competitor-intelligence
description: >
  竞品情报分析 Skill。输入竞品名称和舆情文本，自动执行三维度分析（价格/新品/舆情），
  生成 SWOT 对标报告和竞争态势简报。支持多源数据（CSV/网页/RSS）加载。
trigger_keywords: ["竞品", "竞争对手", "分析", "对标", "情报", "SWOT", "降价", "新品", "舆情"]
inputs:
  - competitor_name: 竞品名称
  - news_text: 舆情文本或数据源路径
outputs:
  - price_analysis: 价格维度提取结果
  - product_analysis: 新品维度提取结果
  - sentiment_analysis: 舆情维度提取结果
  - swot_report: SWOT 分析报告
  - brief_report: 竞争态势简报（Markdown）
dependencies:
  - langchain
  - langchain-openai
  - langchain-chroma
  - langchain-huggingface
  - python-dotenv
---

# competitor-intelligence

## 概述

基于大模型的竞品情报自动分析 Skill。接收竞品名称和舆情文本，调用工具链完成价格、新品、舆情三维度信息提取，最终生成 SWOT 对标分析报告和完整的竞争态势简报。

## 触发条件

当用户提及以下关键词时自动激活：
- 竞品、竞争对手、分析、对标、情报
- SWOT、降价、新品、舆情

## 工作流程

```
舆情文本 → 工具提取（价格/新品/舆情） → Agent 整合 → SWOT + 简报
```

## 使用示例

### Python API

```python
from modules.agent_core import CompetitorAgent

agent = CompetitorAgent(with_memory=True)

# 三维度分析
result = agent.analyze_news("竞品A降价10%，发布新款产品，用户投诉售后慢")

# 生成完整简报
report = agent.generate_full_report(
    price=result["price"],
    product=result["product"],
    sentiment=result["sentiment"],
)
print(report)
```

### 命令行

```bash
python day2/test_day2.py
```

## 架构

```
modules/
├── llm_client.py    # LLM 客户端（DeepSeek OpenAI 兼容）
├── prompts.py       # 提示词统一管理
├── data_loader.py   # 多源数据加载
├── rag_chain.py     # RAG 检索增强链
├── tools.py         # Agent 工具集
├── agent_core.py    # Agent 核心 + 记忆
└── api_server.py    # FastAPI 接口（待实现）
```
