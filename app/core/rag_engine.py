import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.config import settings

class RAGEngine:
    """
    RAG (Retrieval-Augmented Generation) Engine
    Fulfills PDF Section 4 & 14:
    - Document Chunking
    - TF-IDF & Vector Search
    - Reranking Pipeline
    - Context Retrieval for Solution Recommendation
    """

    def __init__(self, kb_dir: Optional[Path] = None):
        self.kb_dir = kb_dir or (Path(__file__).resolve().parent.parent / "data" / "knowledge_base")
        self.documents: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_vectors = None
        self.load_and_index_documents()

    def load_and_index_documents(self):
        """Loads all markdown policy files and indexes them into chunks."""
        self.documents = []
        if not self.kb_dir.exists():
            return

        files = glob.glob(str(self.kb_dir / "*.md"))
        doc_id = 1
        raw_texts = []

        for filepath in files:
            filename = os.path.basename(filepath)
            title = filename.replace(".md", "").replace("_", " ").title()
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Split into chunks by headers or paragraphs (~400-500 chars)
            chunks = [c.strip() for c in content.split("\n\n") if len(c.strip()) > 30]
            for idx, chunk in enumerate(chunks):
                doc_obj = {
                    "id": f"DOC-{doc_id}",
                    "title": f"{title} (Part {idx+1})",
                    "file_name": filename,
                    "category": title,
                    "content": chunk
                }
                self.documents.append(doc_obj)
                raw_texts.append(f"{title} {chunk}")
                doc_id += 1

        if raw_texts:
            self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            self.doc_vectors = self.vectorizer.fit_transform(raw_texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top_k matching document chunks based on semantic similarity."""
        if not self.documents or self.vectorizer is None or self.doc_vectors is None:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.doc_vectors).flatten()

        # Sort indices by score descending
        top_indices = similarities.argsort()[::-1][:top_k * 2]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > 0.01:
                doc_copy = dict(self.documents[idx])
                doc_copy["score"] = round(score, 4)
                results.append(doc_copy)

        # Reranking pass: Boost exact title/category keyword matches
        for doc in results:
            boost = 0.0
            query_lower = query.lower()
            if any(term in query_lower for term in doc["category"].lower().split()):
                boost += 0.2
            if "refund" in query_lower and "refund" in doc["title"].lower():
                boost += 0.3
            if "order" in query_lower and "order" in doc["title"].lower():
                boost += 0.2
            doc["score"] = round(min(doc["score"] + boost, 0.99), 4)

        # Sort again after reranking
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

# Global singleton RAG instance
rag_service = RAGEngine()
