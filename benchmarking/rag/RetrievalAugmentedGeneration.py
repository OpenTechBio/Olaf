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
    import 
    from validators import url as is_url
    from rich.console import Console
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

# ── Collections Directory ---------------------------------------------------------
console = Console()
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
EMBEDDING_DIR = PARENT_DIR / "benchmarking" / "rag" 
EMBEDDING_FILE = EMBEDDING_DIR / "embeddings.json"
FUNCTIONS_FILE = EMBEDDING_DIR / "functions.json"
'''
1. Function to convert documents into embeddings 
2. Store the documents in 'collections' folder 
3. Retrieve the embeddings 
Encode the query 
Compare to the query using cosine similarity 
'''
class RetrievalAugmentedGeneration:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    def __init__(self):
        self.embeddings = load_embeddings()
        self.query = []

    def extract_and_embed_scib(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        func_def = soup.find('h1').get_text(strip=True)
        func_descr = soup.select_one("dt.sig.sig-object.py").get_text(strip=True)
        with open(FUNCTIONS_FILE, "w") as f:
            f.write(json.dumps(func_def.tolist()) + "\n")  
        create_embeddings(func_descr)
        return func_def, func_descr

    def load_embeddings(self):
        embeddings = []
        try:
            with open(EMBEDDING_FILE, "r") as f:
                for line in f:
                    embedding = json.loads(line.strip())
                    embeddings.append(np.array(embedding))
        except FileNotFoundError as e:
            console.log(f"[red]Empty or invalid file")
        return embeddings 
        
                
    def create_embeddings(self, texts):
        embedding = RetrievalAugmentedGeneration.model.encode(texts)
        try:
            with open(EMBEDDING_FILE, "w") as f:
                for emb in embedding:
                    f.write(json.dumps(emb.tolist()) + "\n")  #convert to list, add newline, and dump as json 
            self.embeddings.append(embedding)
            console.print(f"[green]New embeddings successfully generated and stored to {EMBEDDING_FILE}")
        except ValueError as e:
            console.print(f"[red]Invalid or missing input")

    @classmethod
    def cosine_similarity(cls, A, B):
        dot_product = np.dot(A, B)
        norm_A = np.linalg.norm(A)
        norm_B = np.linalg.norm(B)
        cos_sim = dot_product / (norm_A * norm_B)
        return cos_sim        
            
    def query(self, query):
        self.query.append(query)
        query_embedding = RetrievalAugmentedGeneration.model.encode([query])
        sims = cosine_similarity(query_embedding, self.embeddings)
        idx = np.argmax(sims)
        return idx, self.embeddings[idx]


rag = RetrievalAugmentedGeneration()
rag.extract_and_embed_scib("https://scib-metrics.readthedocs.io/en/latest/generated/scib_metrics.bras.html")
rag.query("What is scib")