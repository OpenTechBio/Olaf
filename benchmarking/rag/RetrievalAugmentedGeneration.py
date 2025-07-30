from os import path
from urllib.request import urlopen
from urllib.error import URLError
from chromadb import Client
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from validators import url as is_url



"""
Fixes to make for the RAG model:
1. Think about altering collections to be class variables shared by all instances of the class
2. Think about adding some functions that allow the user to play around with diff settings for text
splitting and embedding distance calculations 
3. Is web parsing happening accurately? 
4. Can I spice up the custom error for expanded functionality?
"""
class RetrievalAugmentedGeneration:
    def __init__(self):
        self.embedding_fn = DefaultEmbeddingFunction()
        self.client = Client()
        self.collection = self.client.get_or_create_collection(
            name="OLAF_collection", 
            embedding_function=self.embedding_fn, 
            metadata={"hnsw:space": "cosine"}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", "!", "?", ","], 
            chunk_size=200, 
            chunk_overlap=50
        )
        self._docs = []

    def load_file(self, file_name):
        try:
            with open(file_name, "r") as f:
                contents = f.read()
                if not contents:
                    raise FileNotFoundError("Empty file")
                return contents
        except FileNotFoundError as e:
            print(f"{e}")
            return ""

    def load_url(self, url):
        try:
            response = requests.get("https://api.example.com/data")
            data = response.json()
            if not data:
                raise Exception
            return data
        except Exception:
            try:
                contents = urlopen(url)
                contents = contents.read().decode('utf-8')
                if not contents:
                    raise URLError("Empty URL")
                return contents
            except URLError as e:
                print(f"Failed to fetch via urlopen: {e}")
                return ""


    @property
    def docs(self):
        return self._docs

    @docs.setter
    def docs(self, file_name_or_url):
        if path.isfile(file_name_or_url):
            file_contents = self.load_file(file_name_or_url)
            if file_contents:
                self._docs.append(file_contents)
        elif is_url(file_name_or_url):
            url_contents = self.load_url(file_name_or_url)
            if url_contents:
                self._docs.append(url_contents)
        else:
            print("Could not find valid URL or file")
    
    def chunks(self):
        chunks = self.text_splitter.create_documents(self.docs)
        return [chunk.page_content for chunk in chunks]

    def add_to_collection(self, chunk_texts):
        self.collection.add(documents=chunk_texts, ids=[f"id_{i}" for i in range(len(chunk_texts))])

    def query(self, query: str, n_results: int):
        return self.collection.query(query_texts=[query], n_results=n_results)

    def rag(self, query, n_results=1, file_name_or_url=None):
        if file_name_or_url:
            self.docs = file_name_or_url
        self.add_to_collection(self.chunks())
        return self.query(query, n_results)
