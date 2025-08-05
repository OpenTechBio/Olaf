import uuid
from os import path
from urllib.request import urlopen
from urllib.error import URLError
from pathlib import Path
import sys

# ── Dependencies ------------------------------------------------------------
try:
    import requests
    import numpy as np
    from validators import url as is_url
    from rich.console import Console
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

# ── Collections Directory ---------------------------------------------------------
console = Console()
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
EMBEDDING_DIR = PARENT_DIR / "rag" / "collections"
'''
1. Function to convert documents into embeddings 
2. Store the documents in 'collections' folder 
3. Retrieve the embeddings 
Encode the query 
Compare to the query using cosine similarity 
'''

class RetrievalAugmentedGeneration:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    def __init__(self, query):
        self.docs = []
        self.query = [query]
        
    def create_embeddings(self, texts):
        embeddings = RetrievalAugmentedGeneration.model.encode(texts)
        with open("", w) as f:
            f.write(embeddings) 
        console.print(f"[green]New embeddings successfully generated and stored to: {EMBEDDING_DIR}")

    @classmethod
    def cosine_similarity(cls, A, B):
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        cos_sim = dot_product / (norm_A * norm_B)
        return cos_sim        
            
    def query(self, query):
        query_embedding = RetrievalAugmentedGeneration.model.encode([query])
        sims = cosine_similarity(query_embedding, embeddings)
        most_similar_idx = np.argmax(sims)
    