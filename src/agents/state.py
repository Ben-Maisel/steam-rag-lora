from typing import TypedDict, List
from langchain_core.documents import Document


class RAGState(TypedDict):
    question: str
    rewritten_question: str
    retrieved_docs: List[Document]
    filtered_docs: List[Document]
    answer: str
    route_decision: str