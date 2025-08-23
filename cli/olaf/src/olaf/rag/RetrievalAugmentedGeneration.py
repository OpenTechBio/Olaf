import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# ── Dependencies ─────────────────────────────────────────────
try: 
    import re 
    from sentence_transformers import SentenceTransformer
    from rich.console import Console
    import matplotlib.pyplot as plt
    import numpy as np
    
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    sys.exit(1) 

# ── Paths and Constants ─────────────────────────────────────────────
console = Console()

RAG_DIR = Path(__file__).resolve().parent.parent / "rag"
EMBEDDING_FILE = RAG_DIR / "embeddings.jsonl"
FUNCTIONS_FILE = RAG_DIR / "functions.jsonl"

class RetrievalAugmentedGeneration():
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    def __init__(self):
        self.embeddings = self.load_embeddings()
        self.functions = self.load_functions()
        self.queries = []

    def load_embeddings(self) -> List[np.ndarray]:
        try:
            with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
                return [np.array(json.loads(line)) for line in f if line.strip()]
        except FileNotFoundError:
            console.log("[red]Embeddings file not found.")
            return []
        except json.JSONDecodeError:
            console.log("[red]Embeddings file is not valid JSONL.")
            return []
    
    def load_functions(self) -> List[Dict[str, str]]:
        try:
            with open(FUNCTIONS_FILE, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            console.log("[red]Functions file not found.")
            return []
        except json.JSONDecodeError:
            console.log("[red]Functions file is not valid JSONL.")
            return []

    @staticmethod
    def cosine_similarity(A: np.ndarray, B: List[np.ndarray]) -> List[float]:
        sims = [np.dot(A, emb) / (np.linalg.norm(A) * np.linalg.norm(emb)) for emb in B]
        return sims

    def retrieve_function(self, name:str) -> Optional[str]:
        for function in self.functions:
            if name in function["signature"]:
                return function["signature"]
        return None

    def query(self, text_query: str) -> Optional[np.ndarray]:
        self.queries.append(text_query)
        if not self.embeddings:
            console.log("[yellow]No embeddings to compare.")
            return None
        query_embedding = self.model.encode([text_query])[0]
        sims = self.cosine_similarity(query_embedding, self.embeddings)
        idx = np.argmax(sims)
        return self.functions[idx]["signature"]

 # ──────Implementation──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rag = RetrievalAugmentedGeneration()
    print(rag.query("What is pca"))
