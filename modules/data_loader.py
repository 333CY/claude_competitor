"""
数据加载模块
支持：网页抓取、RSS 订阅、CSV 导入、文本文件加载
"""
import os
import csv
from typing import List
from langchain_core.documents import Document


def load_csv(file_path: str, text_column: str = "content") -> List[Document]:
    """
    从 CSV 文件加载舆情数据
    Args:
        file_path: CSV 文件路径
        text_column: 文本内容所在列名
    Returns:
        LangChain Document 列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if text_column not in row:
                raise KeyError(f"CSV 中未找到列 '{text_column}'，可用列: {list(row.keys())}")
            text = row[text_column].strip()
            if not text:
                continue
            metadata = {k: v for k, v in row.items() if k != text_column}
            metadata["source"] = file_path
            docs.append(Document(page_content=text, metadata=metadata))
    return docs


def load_text_files(directory: str, encoding: str = "utf-8") -> List[Document]:
    """
    加载目录下所有 .txt 和 .md 文件
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"目录不存在: {directory}")

    docs = []
    for fname in os.listdir(directory):
        if not fname.endswith((".txt", ".md")):
            continue
        fpath = os.path.join(directory, fname)
        with open(fpath, "r", encoding=encoding) as f:
            text = f.read().strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": fpath, "filename": fname}
            ))
    return docs


def load_web_page(url: str) -> Document:
    """
    加载单个网页内容（需安装 beautifulsoup4, requests）
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("网页加载需要安装: pip install beautifulsoup4 requests")

    resp = requests.get(url, timeout=10, headers={"User-Agent": "CompetitorAnalysisBot/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # 移除脚本和样式
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return Document(page_content=text, metadata={"source": url})


def load_rss_feed(url: str) -> List[Document]:
    """
    加载 RSS 订阅源（需安装 feedparser）
    """
    try:
        import feedparser
    except ImportError:
        raise ImportError("RSS 加载需要安装: pip install feedparser")

    feed = feedparser.parse(url)
    docs = []
    for entry in feed.entries:
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        published = entry.get("published", "")
        content = f"{title}\n{summary}"
        docs.append(Document(
            page_content=content,
            metadata={"source": url, "title": title, "published": published}
        ))
    return docs


__all__ = ["load_csv", "load_text_files", "load_web_page", "load_rss_feed"]
