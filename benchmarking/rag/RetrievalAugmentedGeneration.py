import uuid
import json
import sys
from os import path
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
from typing import List, Dict, Union, Optional
# ── Dependencies ─────────────────────────────────────────────
try: 
    import requests
    import numpy as np
    from bs4 import BeautifulSoup
    from sentence_transformers import SentenceTransformer
    from validators import url as is_url
    from rich.console import Console
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    sys.exit(1) 

# ── Paths and Constants ─────────────────────────────────────────────
console = Console()
SCRIPT_DIR = Path(__file__).resolve().parent
EMBEDDING_FILE = SCRIPT_DIR / "embeddings.json"
FUNCTIONS_FILE = SCRIPT_DIR / "functions.json"

# ──────Class──────────────────────────────────────────────────────────
class RetrievalAugmentedGeneration:
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def __init__(self) -> None:
        self.embeddings = self.load_embeddings()
        self.functions = self.load_functions()
        self.query_history = []

    def view_history(self) -> None:
        print("Query history:", self.query_history)

    def load_embeddings(self) -> Optional[List[np.ndarray]]:
        embeddings = []
        try:
            with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue 
                    embedding = json.loads(line.strip())
                    embeddings.append(np.array(embedding))
        except FileNotFoundError:
            console.log("[red]Embeddings file not found.")
        return embeddings

    def load_functions(self) -> Optional[List[Dict[str, str]]]:
        functions = []
        try:
            with open(FUNCTIONS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue 
                    function = json.loads(line.strip())
                    functions.append(function)
        except FileNotFoundError:
            console.log("[red]Functions file not found.")
        return functions

    def extract_html(self, url: str) -> Optional[Dict[str, str]]:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        func_sig = soup.select_one("dt.sig.sig-object.py")
        if not func_sig:
            console.log("[red] No function signature found")
            return {}

        func_def = func_sig.get_text(strip=True)
        descr_tag = func_sig.find_next_sibling("dd")
        func_descr = descr_tag.p.get_text(strip=True) if descr_tag and descr_tag.p else ""
        return {"source": url, "definition": func_def, "description": func_descr} 

    def add_function(self, func: Dict[str, str]) -> Optional[Dict[str, str]]:
        try:
            with open(FUNCTIONS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(func) + "\n")
        except Exception as e:
            console.print(f"[red]Failed to write to FUNCTIONS_FILE")
            return {}
        self.functions.append(func)
        return func

    def create_embeddings(self, text: str) -> None:
        embeddings = self.model.encode([text])[0]
        try:
            with open(EMBEDDING_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(embeddings.tolist()) + "\n")
            self.embeddings.append(embeddings)
            console.print(f"[green]Embeddings successfully stored in {EMBEDDING_FILE}")
        except Exception as e:
            console.print(f"[red]Failed to create embeddings: {e}")

    def url_exists(self, url: str) -> bool:
        for f in self.functions:
            if url == f["source"]:
                console.print("[yellow] Function and embedding already exists.")
                return True
        return False 

    def find_by_url(self, url: str) -> Optional[Dict[str, str]]:
        for idx, f in enumerate(self.functions):
            if f["source"] == url:
                return f
        console.print("URL not found")
        return {}

    @staticmethod
    def cosine_similarity(A: np.ndarray, B: np.ndarray) -> List[float]:
        A = np.array(A)
        B = np.array(B)
        sims = [np.dot(A, emb) / (np.linalg.norm(A) * np.linalg.norm(emb)) for emb in B]
        return sims

    def query(self, text_query: str) -> Optional[Dict[str, str]]:
        self.query_history.append(text_query)
        if not self.embeddings:
            console.log("[yellow]No embeddings to compare.")
            return {}
        query_embedding = self.model.encode([text_query])[0]
        sims = self.cosine_similarity(query_embedding, self.embeddings)
        idx = np.argmax(sims)
        return self.functions[idx]

    def clear(self) -> None:
        self.embeddings = []
        self.query_history = []
        self.functions = []
        
#─────────────────────────────────────────────
if __name__ == "__main__":
    rag = RetrievalAugmentedGeneration()
    urls =["https://scib-metrics.readthedocs.io/en/latest/generated/scib_metrics.utils.pca.html", "https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_mtx.html", "https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_h5ad.html", "https://scanpy.readthedocs.io/en/stable/generated/scanpy.read_10x_mtx.html"]
    for url in urls:
        if not rag.url_exists(url):
            func = rag.extract_html(url)
            if func and func["description"]:
                rag.add_function(func)
                rag.create_embeddings(func["description"])
        else:
            func = rag.find_by_url(url)
    console.print(rag.embeddings)
    result = rag.query("Function to perform PCA on h5AD file")
    console.print("Response to the query is", result) 
