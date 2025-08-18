import uuid
import json
import sys
from os import path
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError
from typing import List, Tuple, Dict, Union, Optional
# ── Dependencies ─────────────────────────────────────────────
try: 
    import requests
    from umap import UMAP
    import re 
    from bs4 import BeautifulSoup
    from sentence_transformers import SentenceTransformer
    from validators import url as is_url
    from rich.console import Console
    import random 
    import matplotlib.pyplot as plt
    import numpy as np
    import sklearn
    import seaborn as sns
    import wikipedia 
    
except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    sys.exit(1) 

# ── Paths and Constants ─────────────────────────────────────────────
console = Console()
EMBEDDING_LEN = 5
SCRIPT_DIR = Path(__file__).resolve().parent
EMBEDDING_FILE = SCRIPT_DIR / "embeddings.jsonl"
FUNCTIONS_FILE = SCRIPT_DIR / "functions.jsonl"

# ──────Class──────────────────────────────────────────────────────────
class RetrievalAugmentedGeneration:
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    def __init__(self) -> None:
        self.embeddings = self.load_embeddings()
        self.functions = self.load_functions()
        self.queries = []

    def view_query_history(self) -> None:
        console.log(f"Query history:")
        for i in range(len(self.queries)):
            console.log(f"[Query {i}] {self.queries[i]}")

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
    
    def load_functions(self) -> List[str]:
        try:
            with open(FUNCTIONS_FILE, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            console.log("[red]Functions file not found.")
            return []
        except json.JSONDecodeError:
            console.log("[red]Functions file is not valid JSONL.")
            return []


    def add_function(self, func: str) -> None:
        try:
            with open(FUNCTIONS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(func) + "\n")
            console.print(f"[green]Functions successfully stored in {FUNCTIONS_FILE}")
            self.functions.append(func)
        except Exception as e:
            console.print(f"[red]Failed to write to FUNCTIONS_FILE")

    def add_embedding(self, text: str) -> None:
        embeddings = self.model.encode([text])[0]
        try:
            with open(EMBEDDING_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(embeddings.tolist()) + "\n")
            console.print(f"[green]Embeddings successfully stored in {EMBEDDING_FILE}")
            self.embeddings.append(embeddings)
        except Exception as e:
            console.print(f"[red]Failed to create embeddings: {e}")

    def extract_html_scib(self, url: str) -> Optional[Tuple[str, str]]:
        try:
            response = requests.get(url)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            console.log(f"[red]Request timed out: URL={url} | Error={e}")
            return None 
        except requests.exceptions.RequestException as e:
            console.log(f"[red] Request failed: URL={url} | Error={e}")
            return None 
            
        soup = BeautifulSoup(response.text, 'html.parser')
        func_sig = soup.select_one("dt.sig.sig-object.py")
        if not func_sig:
            console.log("[red] No function signature found")
            return None

        func_def = ''.join(func_sig.find_all(text=True, recursive=False)).strip()
        descr_tag = func_sig.find_next_sibling("dd")
        func_descr = descr_tag.p.get_text(strip=True) if descr_tag and descr_tag.p else ""
        return func_def, func_descr 

    def extract_wiki_content(self, search_term:str) -> Optional[str]:
        if not search_term:
            return None 
        try: 
            page_title = wikipedia.search(search_term)[0]
            wiki_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
            response = requests.get(wiki_url)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            console.log(f"[red]Request timed out: URL={url} | Error={e}")
            return None
        except requests.exceptions.RequestException as e:
            console.log(f"[red] Request failed: URL={url} | Error={e}")
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", {"class": "mw-parser-output"})
        for tag in content.find_all(["table", "sup", "span", "math", "img", "figure", "style", "script"]):
            tag.decompose()  
        text = content.get_text(separator=" ", strip=True)
        
        page_sentences = re.split(r'(?<=[.!?]) +', text)
        embedding_content = " ".join(page_sentences[:EMBEDDING_LEN])
        
        return embedding_content

        
    def retrieve_function(self, name:str) -> Optional[str]:
        for function in self.functions:
            if name in function:
                return function
        return None


    def embedding_pipeline(self, url:str) -> None:
        func_definition, search_term  = self.extract_html_scib(url)
        if not func_definition or not search_term:
            return 
        func = self.retrieve_function(func_definition)
        if not func:
            embedding_content = self.extract_wiki_content(search_term)
            if embedding_content:
                self.add_embedding(embedding_content)
                self.add_function(func_definition)
        else:
            console.log(f"[yellow] Embedding for url {url} already exists.")
            

    @staticmethod
    def cosine_similarity(A: np.ndarray, B: np.ndarray) -> List[float]:
        sims = [np.dot(A, emb) / (np.linalg.norm(A) * np.linalg.norm(emb)) for emb in B]
        return sims
    
    def query(self, text_query: str) -> Optional[np.ndarray]:
        self.queries.append(text_query)
        if not self.embeddings:
            console.log("[yellow]No embeddings to compare.")
            return None 
        query_embedding = self.model.encode([text_query])[0]
        sims = self.cosine_similarity(query_embedding, self.embeddings)
        idx = np.argmax(sims)
        return self.functions[idx]

    def umap_plot(self) -> None:
        if not self.embeddings or not self.queries:
            console.log("[yellow]No embeddings and/or queries to plot.")
            return
        
        query_embeddings = self.model.encode(self.queries)
        embeddings_array = np.array(self.embeddings)
        all_embeddings = np.vstack([embeddings_array, query_embeddings])
        
        n_neighbors = min(15, len(all_embeddings) - 1)
        umap_embeddings = UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='cosine'
        ).fit_transform(all_embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_embeddings[:len(self.embeddings), 0],
                    umap_embeddings[:len(self.embeddings), 1],
                    label="Chunks", color="blue")
        plt.scatter(umap_embeddings[len(self.embeddings):, 0],
                    umap_embeddings[len(self.embeddings):, 1],
                    label="Queries", color="red", marker="x", s=100)
        
        for i, (x, y) in enumerate(umap_embeddings[:len(self.embeddings)]):
            plt.annotate(self.functions[i], (x, y), textcoords="offset points", xytext=(0, 5),
                         ha='center', fontsize=8, color='blue')
        
        for i, (x, y) in enumerate(umap_embeddings[len(self.embeddings):]):
            plt.annotate(self.queries[i], (x, y), textcoords="offset points", xytext=(0, 5),
                         ha='center', fontsize=10, color='red')
        
        plt.legend()
        plt.title("UMAP Projection of Embeddings and Queries")
        plt.savefig("umap_queries.png")
        console.log(f"[green]UMAP plot for all queries saved as umap_queries.png [/green]")
        plt.close()
        
    def cosine_distance_heatmap(self) -> None:
        if not self.embeddings or not self.queries:
            console.log("[yellow]No embeddings and/or queries to compare.")
            return
    
        query_embeddings = self.model.encode(self.queries)
        embeddings_array = np.array(self.embeddings)
    
        # Compute cosine distances between queries and embeddings
        distances = sklearn.metrics.pairwise_distances(
            X=query_embeddings,       
            Y=embeddings_array,  
            metric='cosine'
        )

        row_labels = [f"Query {i+1}" for i in range(len(self.queries))]
        col_labels = [f"Chunk {i+1}" for i in range(len(self.functions))]
    
        plt.figure(figsize=(len(col_labels)*0.5 + 6, len(row_labels)*0.5 + 2))
        sns.heatmap(distances, square=False, annot=True, cbar=True, cmap='Blues',
                    xticklabels=col_labels, yticklabels=row_labels)
        plt.title("Cosine Distance Heatmap (Queries × Embeddings)")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
    
        query_map = "\n".join([f"Query {i+1}: {q}" for i, q in enumerate(self.queries)])
        chunk_map = "\n".join([f"Chunk {i+1}: {kw}" for i, kw in enumerate(self.functions)])
    

        plt.figtext(1.02, 0.5, f"{query_map}\n\n{chunk_map}",
                    ha="left", va="center", fontsize=8)

        plt.savefig("full_cosine_distance_heatmap.png", bbox_inches="tight")
        console.log("[green]Full cosine distance heatmap saved as full_cosine_distance_heatmap.png[/green]")

        plt.close()


    def clear(self) -> None:
        self.embeddings = []
        self.queries = []
        self.functions = []
        
# ──────Implementation──────────────────────────────────────────────────────────

if __name__ == "__main__":
    rag = RetrievalAugmentedGeneration()
    url = "https://scib-metrics.readthedocs.io/en/latest/generated/scib_metrics.utils.pca.html"
    rag.embedding_pipeline(url)
    print(type(rag.embeddings))
    print(rag.query("What is pca"))

    
    
