import sqlite3
import json
import numpy as np
import os
from typing import Optional

# Force local model loading — no network calls
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SemanticCache:
    def __init__(self, db_path: str = "data/cache/semantic_cache.db"):
        import os
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self._create_table()

    def _create_table(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                query_vector BLOB,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def _cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def get(self, query: str, threshold: float = 0.92) -> Optional[str]:
        """
        Retrieves a cached response if the semantic similarity is above the threshold.
        """
        query_vec = self.embeddings.embed_query(query)
        
        cursor = self.conn.execute("SELECT query, query_vector, response FROM cache")
        best_match = None
        max_sim = -1
        
        for row in cursor:
            stored_query, stored_vec_blob, response = row
            stored_vec = np.frombuffer(stored_vec_blob, dtype=np.float32)
            
            sim = self._cosine_similarity(query_vec, stored_vec)
            if sim > max_sim:
                max_sim = sim
                best_match = response
        
        if max_sim >= threshold:
            return best_match
        return None

    def set(self, query: str, response: str):
        query_vec = self.embeddings.embed_query(query)
        vec_blob = np.array(query_vec, dtype=np.float32).tobytes()
        
        self.conn.execute(
            "INSERT INTO cache (query, query_vector, response) VALUES (?, ?, ?)",
            (query, vec_blob, response)
        )
        self.conn.commit()

# Singleton instance
cache_instance = SemanticCache()
