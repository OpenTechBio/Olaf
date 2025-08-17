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
    from umap import UMAP
    import re 
    from bs4 import BeautifulSoup
    from sentence_transformers import SentenceTransformer
    from validators import url as is_url
    from rich.console import Console
    import wikipediaapi 
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
SCRIPT_DIR = Path(__file__).resolve().parent
EMBEDDING_FILE = SCRIPT_DIR / "embeddings.jsonl"
FUNCTIONS_FILE = SCRIPT_DIR / "functions.jsonl"

# ──────Class──────────────────────────────────────────────────────────
class RetrievalAugmentedGeneration:
    model = SentenceTransformer('intfloat/e5-base-v2')

    def __init__(self) -> None:
        self.embeddings = self.load_embeddings()
        self.functions = self.load_functions()
        self.queries = []

    def view_history(self) -> None:
        print("Query history:", self.queries)

    def load_embeddings(self) -> Optional[List[np.ndarray]]:
        try:
            with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
                return [np.array(json.loads(line)) for line in f if line.strip()]
        except FileNotFoundError:
            console.log("[red]Embeddings file not found.")
            return []
        except json.JSONDecodeError:
            console.log("[red]Embeddings file is not valid JSONL.")
            return []
    
    def load_functions(self) -> Optional[List[Dict[str, str]]]:
        try:
            with open(FUNCTIONS_FILE, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            console.log("[red]Functions file not found.")
            return []
        except json.JSONDecodeError:
            console.log("[red]Functions file is not valid JSONL.")
            return []

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

    
    def query(self, text_query: str) -> Optional[np.ndarray]:
        self.queries.append(text_query)
        if not self.embeddings:
            console.log("[yellow]No embeddings to compare.")
            return {}
        query_embedding = self.model.encode([text_query])[0]
        sims = self.cosine_similarity(query_embedding, self.embeddings)
        idx = np.argmax(sims)
        return self.embeddings[idx]

    def umap_plot(self, keywords: List[str]) -> None:
        if not self.embeddings or not self.queries:
            console.log("[yellow]No embeddings or queries to plot.")
            return
        
        if len(keywords) != len(self.embeddings):
            console.log("[red]Number of keywords must match number of embeddings![/red]")
            return
        
        # Encode all queries
        query_embeddings = self.model.encode(self.queries)
        embeddings_array = np.array(self.embeddings)
        all_embeddings = np.vstack([embeddings_array, query_embeddings])
        
        # Reduce to 2D with UMAP
        n_neighbors = min(15, len(all_embeddings) - 1)
        umap_embeddings = UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='cosine'
        ).fit_transform(all_embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_embeddings[:len(self.embeddings), 0],
                    umap_embeddings[:len(self.embeddings), 1],
                    label="Chunks", color="blue")
        plt.scatter(umap_embeddings[len(self.embeddings):, 0],
                    umap_embeddings[len(self.embeddings):, 1],
                    label="Queries", color="red", marker="x", s=100)
        
        # Annotate embeddings with keywords
        for i, (x, y) in enumerate(umap_embeddings[:len(self.embeddings)]):
            plt.annotate(keywords[i], (x, y), textcoords="offset points", xytext=(0, 5),
                         ha='center', fontsize=8, color='blue')
        
        # Annotate queries with actual query strings
        for i, (x, y) in enumerate(umap_embeddings[len(self.embeddings):]):
            plt.annotate(self.queries[i], (x, y), textcoords="offset points", xytext=(0, 5),
                         ha='center', fontsize=10, color='red')
        
        plt.legend()
        plt.title("UMAP Projection of All Embeddings + Queries")
        
        filename = f"umap_all_queries_{random.randint(0, 100)}.png"
        plt.savefig(filename)
        console.log(f"[green]UMAP plot for all queries saved as {filename}[/green]")
        plt.close()
        
    def cosine_distance_heatmap(self, keywords: List[str]) -> None:
        if not self.embeddings or not self.queries:
            console.log("[yellow]No embeddings or queries to compare.")
            return

        if len(keywords) != len(self.embeddings):
            console.log("[red]Number of keywords must match number of embeddings![/red]")
            return
    
        query_embeddings = self.model.encode(self.queries)
        embeddings_array = np.array(self.embeddings)
    
        # Compute cosine distances between queries and embeddings
        distances = sklearn.metrics.pairwise_distances(
            X=query_embeddings,       # rows: queries
            Y=embeddings_array,       # cols: embeddings
            metric='cosine'
        )
    
        # Labels
        row_labels = [f"Query {i+1}" for i in range(len(self.queries))]
        col_labels = [f"Chunk {i+1}" for i in range(len(self.embeddings))]
    
        # Plot heatmap
        plt.figure(figsize=(len(col_labels)*0.5 + 6, len(row_labels)*0.5 + 2))
        sns.heatmap(distances, square=False, annot=True, cbar=True, cmap='Blues',
                    xticklabels=col_labels, yticklabels=row_labels)
        plt.title("Cosine Distance Heatmap (Queries × Embeddings)")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=8)
    
        # Build mapping text
        query_map = "\n".join([f"Query {i+1}: {q}" for i, q in enumerate(self.queries)])
        chunk_map = "\n".join([f"Chunk {i+1}: {kw}" for i, kw in enumerate(keywords)])
    
        # Add mapping text to right side of plot
        plt.figtext(1.02, 0.5, f"{query_map}\n\n{chunk_map}",
                    ha="left", va="center", fontsize=8)

    
        # Save
        filename = f"full_cosine_distance_heatmap_{random.randint(0, 100)}.png"
        plt.savefig(filename, bbox_inches="tight")
        console.log(f"[green]Full cosine distance heatmap saved as {filename}[/green]")
        plt.close()



    def clear(self) -> None:
        self.embeddings = []
        self.queries = []
        self.functions = []
        
#─────────────────────────────────────────────
if __name__ == "__main__":
    rag = RetrievalAugmentedGeneration()
    urls = ["https://scib-metrics.readthedocs.io/en/latest/generated/scib_metrics.utils.pca.html"]
    keywords = ["1_sentence", "5_sentences", "10_sentences", "20_sentences", "30_sentences", "full"]
    prompts = ["SCIB Metrics Principal Component Analysis"]

    wiki = wikipediaapi.Wikipedia(language="en", user_agent="OlafBot")

    for i in range(len(urls)):
        url = urls[i]
        search_term = "Principal Component Analysis"
        func = rag.extract_html(url)
        if func and func["description"]:
            search_results = wikipedia.search(search_term)
            wiki_page = wiki.page(search_results[0])
            full_text = wiki_page.text
            sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', full_text) if s.strip()]

            sentence_lengths = [1, 5, 10, 20, 30, None]
            for n in sentence_lengths:
                if n is None:
                    text_variant = " ".join(sentences)  # full page
                else:
                    text_variant = " ".join(sentences[:n])  # first n sentences

                console.print(f"[red][bold]{n} sentences:\n")
                console.print(f"{text_variant[-500:]}…")  # preview first 500 chars

                func["definition"] += str(n)
                # rag.add_function(func)
                # rag.create_embeddings(text_variant)

    # rag.queries += prompts
    # rag.cosine_distance_heatmap(keywords)
