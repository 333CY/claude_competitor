"""
RAG 检索增强生成模块
包含：向量库存储 + 检索 + 结构化提取链
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.llm_client import get_llm
from modules.prompts import PRICE_ANALYSIS_PROMPT

# ========== 嵌入模型（延迟加载） ==========
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ========== 向量库存储 ==========
def build_vector_store(
    documents: List[Document],
    persist_dir: str = "./chroma_db",
    chunk_size: int = 200,
    chunk_overlap: int = 20,
) -> Chroma:
    """
    将文档分块后写入 Chroma 向量数据库
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
    )
    return vector_db


def load_vector_store(persist_dir: str = "./chroma_db") -> Chroma:
    """加载已有向量数据库"""
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
    )


# ========== RAG 检索链 ==========
def build_price_rag_chain(persist_dir: str = "./chroma_db"):
    """
    构建价格分析 RAG 链：检索 → 提示词 → LLM → JSON
    """
    vector_db = load_vector_store(persist_dir)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()
    parser = JsonOutputParser()

    prompt = PromptTemplate(
        template=PRICE_ANALYSIS_PROMPT + "\n{format_instructions}",
        input_variables=["context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n".join(d.page_content for d in docs))
        }
        | prompt
        | llm
        | parser
    )
    return rag_chain


def build_rag_chain(
    prompt_template: str,
    persist_dir: str = "./chroma_db",
    parse_json: bool = True,
):
    """
    通用 RAG 链工厂
    Args:
        prompt_template: 提示词模板，需含 {context} 占位符
        persist_dir: 向量库路径
        parse_json: 是否 JSON 解析输出
    """
    vector_db = load_vector_store(persist_dir)
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    llm = get_llm()

    parser = JsonOutputParser() if parse_json else None

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context"],
        partial_variables=(
            {"format_instructions": parser.get_format_instructions()} if parse_json else {}
        ),
    )

    chain = (
        {
            "context": retriever | (lambda docs: "\n".join(d.page_content for d in docs))
        }
        | prompt
        | llm
    )
    if parse_json:
        chain = chain | parser
    return chain


__all__ = [
    "get_embeddings",
    "build_vector_store",
    "load_vector_store",
    "build_price_rag_chain",
    "build_rag_chain",
]
