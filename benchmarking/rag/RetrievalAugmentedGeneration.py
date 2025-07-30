import requests
from os import path
import uuid
import chromadb
import langchain
from urllib.request import urlopen
from urllib.error import URLError
from validators import url as is_url
import chromadb.utils.embedding_functions as ef
import langchain.text_splitter as txt_splitter
from pathlib import Path

# ── Collections Directory ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
COLLECTION_DIR = PARENT_DIR / "rag" / "collections"

class RetrievalAugmentedGeneration:
    def __init__(self,
                 embedding_fn=None,
                 collection_name="OLAF",
                 distance_metric="cosine",
                 seperators=["\n\n", "\n", ".", "!", "?", ","],
                 chunk_size=600,
                 chunk_overlap=50):

        # Persistent Collection is retained on disk memory if directory found, else, create temporary client
        if COLLECTION_DIR.is_dir():
            self.client = chromadb.PersistentClient(path=str(COLLECTION_DIR))
        else:
            self.client = chromadb.Client()

        # Implement with User Parameters
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_fn or ef.DefaultEmbeddingFunction(),
                metadata={"hnsw:space": distance_metric}
            )
        except Exception as e:
            print(f"Failed to create collection '{collection_name}' with embedding function {embedding_fn} and distance '{distance_metric}': {e}")
            print("Reverting to default configuration")
            self.collection = self.client.get_or_create_collection(name="OLAF")

        self.text_splitter = txt_splitter.RecursiveCharacterTextSplitter(
            separators=seperators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self._docs = []

    def load_file(self, file_name: str) -> str:
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                contents = f.read()
                if not contents:
                    raise FileNotFoundError("Empty or Invalid file")
                return contents
        except FileNotFoundError as e:
            print(f"{e}")
            return ""

    def load_url(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            contents = response.text
            if not contents:
                raise ValueError("Empty or Invalid URL")
            return contents
        except (requests.exceptions.RequestException, ValueError):
            try:
                contents = urlopen(url).read().decode('utf-8')
                if not contents:
                    raise URLError("Empty or Invalid URL")
                return contents
            except Exception as e:
                print(f"Failed to fetch via requests and urlopen: {e}")
                return ""

    @property
    def docs(self) -> list[str]:
        return self._docs

    @docs.setter
    def docs(self, file_name_or_url: str):
        if path.isfile(file_name_or_url):
            file_contents = self.load_file(file_name_or_url)
            if file_contents and file_contents not in self._docs:
                self._docs.append(file_contents)
                print(f"Loaded file: {file_name_or_url}")
        elif is_url(file_name_or_url):
            url_contents = self.load_url(file_name_or_url)
            if url_contents and url_contents not in self._docs:
                self._docs.append(url_contents)
                print(f"Loaded URL: {file_name_or_url}")
        else:
            print("Could not find valid URL or file")

    def add_to_collection(self, file_name_or_url: str, n_results: int = 1):
        self.docs = file_name_or_url
        chunks = self.text_splitter.create_documents(self._docs)
        chunks = [chunk.page_content for chunk in chunks]

        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

        self.collection.add(documents=chunks, ids=ids)

    def query(self, query: str, n_results: int = 1):
        return self.collection.query(query_texts=[query], n_results=n_results)
