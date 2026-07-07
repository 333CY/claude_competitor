# ===== 必须放在所有导入之前 =====
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 模型缓存全部下载到E盘
os.environ["HF_HOME"] = r"E:\学校\2025-2026-3\实习实训\claude_competitor\huggingface_cache"

from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. 初始化嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name= "BAAI/bge-m3",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# 模拟竞品数据
mock_texts = [
    "竞品A在2026年7月5日宣布全线产品降价15%，标准版年费从999元降至849元。",
    "竞品A推出了新的入门级套餐，定价为每月49元，直接对标本企业的轻量版。",
    "根据行业论坛爆料，竞品A在618大促期间，高级版实际成交价低至599元/年。",
    "竞品A的CEO在采访中表示，下半年将采取激进的价格策略抢占市场份额。",
]

docs = [Document(page_content=text, metadata={"source": "mock"}) for text in mock_texts]

# 2. 文本分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
split_docs = text_splitter.split_documents(docs)

# 3. 写入向量数据库并持久化
vector_db = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

print(f"✅ 灌库完成！共存入 {len(split_docs)} 条向量数据。")