import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
print("读取到的密钥：", os.environ.get("ANTHROPIC_API_KEY")) 
# 初始化 LLM（这不涉及下载，可以放外面）
llm = ChatOpenAI(
    model="deepseek-v4-pro",
    openai_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.0,
    timeout=30,                    # 增加超时，避免永久卡住
)

# 提示词 + 解析器（不涉及网络，放外面）
parser = JsonOutputParser()
prompt = PromptTemplate(
    template="根据上下文提取竞品价格信息，仅输出JSON：\n{context}\n{format_instructions}",
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

def build_rag_chain():
    """在 main 中调用，延迟加载嵌入模型和向量库"""
    print("◆ 开始加载嵌入模型...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("✅ 嵌入模型加载完成")

    print("◆ 加载向量数据库...")
    vector_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    print("✅ 数据库加载完成")

    # 组装 LCEL 链
    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n".join(d.page_content for d in docs))
        }
        | prompt
        | llm
        | parser
    )
    return rag_chain

if __name__ == "__main__":
    # 执行构建（此时才会真正开始下载/加载）
    rag_chain = build_rag_chain()

    print("◆ 开始执行 RAG 查询...")
    result = rag_chain.invoke("竞品A最近有什么价格变动？")
    print("✅ 结构化价格情报：", result)