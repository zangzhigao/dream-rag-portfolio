# -*- coding: utf-8 -*-
"""检索器集合：FAISS 语义检索 + BM25 词法检索，各自独立、分别返回。

用法：
    from retriever import faiss_topk, bm25_topk, hybrid_topk
"""
from .bm25_retriever import bm25_topk
from .faiss_retriever import faiss_topk, faiss_available
from .hybrid_retrieval import hybrid_topk

__all__ = ["faiss_topk", "bm25_topk", "hybrid_topk", "faiss_available"]
