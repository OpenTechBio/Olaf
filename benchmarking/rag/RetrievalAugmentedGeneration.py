import uuid
import json
import sys
from os import path
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
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
PARENT_DIR = SCRIPT_DIR.parent
EMBEDDING_DIR = PARENT_DIR / "benchmarking" / "rag"
EMBEDDING_FILE = EMBEDDING_DIR / "embeddings.json"
FUNCTIONS_FILE = EMBEDDING_DIR / "functions.json"

# ──────Class──────────────────────────────────────────────────────────
class RetrievalAugmentedGeneration:
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def __init__(self):
        self.embeddings = self.load_embeddings()
        self.functions, self.urls = self.load_functions_and_urls()
        self.query_history = []

    def extract_scib(self, url):
        if self.url_exists(url):
            console.log("URL already exists")
            return self.find_by_source(url)

        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        func_sig = soup.select_one("dt.sig.sig-object.py")
        if not func_sig:
            console.log("No function signature found")
            return ""

        func_def = func_sig.get_text(strip=True)
        descr_tag = func_sig.find_next_sibling("dd")
        func_descr = descr_tag.p.get_text(strip=True) if descr_tag and descr_tag.p else ""

        try:
            with open(FUNCTIONS_FILE, "a") as f:
                f.write(json.dumps({
                    "source": url,
                    "definition": func_def,
                    "description": func_descr
                }) + "\n")
        except Exception as e:
            console.log(f"[red]Failed to write to FUNCTIONS_FILE: {e}")

        return func_def, func_descr

    def load_embeddings(self):
        embeddings = []
        try:
            with open(EMBEDDING_FILE, "r") as f:
                for line in f:
                    embedding = json.loads(line.strip())
                    embeddings.append(np.array(embedding))
        except FileNotFoundError:
            console.log("[red]Embeddings file not found.")
        return embeddings

    def load_functions_and_urls(self):
        functions = []
        urls = []
        try:
            with open(FUNCTIONS_FILE, "r") as f:
                for line in f:
                    function = json.loads(line.strip())
                    functions.append(function["definition"])
                    urls.append(function["source"])
        except FileNotFoundError:
            console.log("[red]Functions file not found.")
        return functions, urls

    def create_embeddings(self, texts):
        embeddings = self.model.encode(texts if isinstance(texts, list) else [texts])
        try:
            with open(EMBEDDING_FILE, "a") as f:
                for emb in embeddings:
                    f.write(json.dumps(emb.tolist()) + "\n")
            self.embeddings.extend([np.array(emb) for emb in embeddings])
            console.print(f"[green]Embeddings successfully stored in {EMBEDDING_FILE}")
        except Exception as e:
            console.print(f"[red]Failed to create embeddings: {e}")

    def url_exists(self, source):
        for url in self.urls:
            if url == source:
                return True
        return False 

    def find_by_source(self, url):
        for idx, source in enumerate(self.urls):
            if source == url:
                return self.functions[idx]
        console.log("URL not found")
        return ""

    @staticmethod
    def cosine_similarity(A, B):
        A = np.array(A)
        B = np.array(B)
        sims = [np.dot(A, emb) / (np.linalg.norm(A) * np.linalg.norm(emb)) for emb in B]
        return sims

    def query(self, text_query):
        self.query_history.append(text_query)
        if not self.embeddings:
            console.log("[yellow]No embeddings to compare.")
            return None
        query_embedding = self.model.encode([text_query])[0]
        sims = self.cosine_similarity(query_embedding, self.embeddings)
        idx = np.argmax(sims)
        return self.functions[idx]

# ── Example ─────────────────────────────────────────────
if __name__ == "__main__":
    rag = RetrievalAugmentedGeneration()
    url = "https://scib-metrics.readthedocs.io/en/latest/generated/scib_metrics.bras.html"
    func_def, func_descr = rag.extract_scib(url)
    rag.create_embeddings([func_def])
    result = rag.query("What is scib?")
    console.print(result)
