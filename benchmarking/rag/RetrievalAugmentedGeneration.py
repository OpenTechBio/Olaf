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
    import umap.UMAP
    import re 
    import numpy as np
    from bs4 import BeautifulSoup
    from sentence_transformers import SentenceTransformer
    from validators import url as is_url
    from rich.console import Console
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
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def __init__(self) -> None:
        self.embeddings = self.load_embeddings()
        self.functions = self.load_functions()
        self.query_history = []

    def view_history(self) -> None:
        print("Query history:", self.query_history)

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

    def query(self, text_query: str) -> Optional[Dict[str, str]]:
        self.query_history.append(text_query)
        if not self.embeddings:
            console.log("[yellow]No embeddings to compare.")
            return {}
    
        # Encode the query
        query_embedding = self.model.encode([text_query])[0]
    
        # Find most similar embedding
        sims = self.cosine_similarity(query_embedding, self.embeddings)
        idx = np.argmax(sims)
    
        # Stack embeddings + query embedding
        all_embeddings = np.vstack([self.embeddings, query_embedding.reshape(1, -1)])
    
        # Reduce to 2D with UMAP
        umap_embeddings = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(all_embeddings)
    
        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(umap_embeddings[:-1, 0], umap_embeddings[:-1, 1], label="Chunks")
        plt.scatter(umap_embeddings[-1, 0], umap_embeddings[-1, 1], color="red", label="Query", marker="x", s=100)
        plt.legend()
        plt.title("UMAP Projection of Embeddings + Query")
        plt.show()
    
        return self.functions[idx]


    def clear(self) -> None:
        self.embeddings = []
        self.query_history = []
        self.functions = []
        
#─────────────────────────────────────────────
if __name__ == "__main__":
    rag = RetrievalAugmentedGeneration()
    urls = [
    # Cupcake
    "A cupcake is a small, single-serving cake that is typically baked in a thin paper or foil cup. It is often topped with frosting, sprinkles, or other decorations to enhance both its flavor and appearance. Cupcakes can come in a wide variety of flavors, such as vanilla, chocolate, red velvet, or lemon. They are popular for parties and celebrations because they are easy to serve and require no slicing. Many bakeries also create gourmet cupcakes with creative fillings and toppings.",
    
    # Cake
    "Cake is a sweet baked dessert made from a mixture of flour, sugar, eggs, butter or oil, and a leavening agent such as baking powder. It can be flavored with a variety of ingredients, including cocoa, vanilla, fruit, or spices. Cakes are often layered and frosted, making them a centerpiece for birthdays, weddings, and other celebrations. They can range in texture from light and fluffy to rich and dense. Over time, countless cultural variations of cake have emerged worldwide.",
    
    # Dosa
    "A dosa is a thin, crispy South Indian crepe made from a fermented batter of rice and lentils. It is typically served hot with chutney and sambar, a spicy lentil-based vegetable stew. Dosas can be plain or filled with a variety of ingredients, the most popular being spiced mashed potatoes in a masala dosa. The fermentation process gives the dosa a slightly tangy flavor and a light texture. It is a staple breakfast dish in many parts of India and is also enjoyed internationally.",
    
    # Biryani
    "Biryani is a flavorful and aromatic rice dish that is popular across South Asia and the Middle East. It is typically made with long-grain basmati rice, meat such as chicken, mutton, or fish, and a blend of fragrant spices. The dish is often layered and slow-cooked to allow the flavors to meld together. Biryani can also be prepared in vegetarian versions using vegetables and paneer. It is often served with raita, salad, or boiled eggs as accompaniments.",
    
    # Pakistan
    "Pakistan is a country in South Asia, bordered by India, Afghanistan, Iran, and China. It has a rich cultural heritage influenced by Persian, Central Asian, and South Asian traditions. The country is known for its diverse landscapes, ranging from mountains like K2 in the north to coastal areas along the Arabian Sea. Pakistan’s cuisine, music, literature, and architecture reflect centuries of history and cultural exchange. It is also home to several UNESCO World Heritage sites, including Mohenjo-daro and the Lahore Fort."
]

    keywords = [
        "Principal component analysis",    # for scib_metrics.utils.pca
        "Matrix market exchange format",   # for scanpy.read_mtx
        "HDF5",                            # for scanpy.read_h5ad (since h5ad files are based on HDF5)
        "10x Genomics"                     # for scanpy.read_10x_mtx
    ]

    for i in range(len(urls)):
        # url = urls[i]
        # keyword = keywords[i]
        # if not rag.url_exists(url):
        #     func = rag.extract_html(url)
        #     if func and func["description"]:
        #         rag.add_function(func)
                # search_results = wikipedia.search(keyword)
                # if not search_results:
                #     continue
                # search = search_results[0]
                # try:
                #     wiki_summary = wikipedia.summary(search, sentences=20)
                # except:
                #     wiki_summary = ""
        embedding_text = urls[i] #+ func["description"]
        rag.functions.append(urls[i])
        rag.create_embeddings(embedding_text)

    print(rag.query("What type of rice is most commonly used in biryani?"))
