from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

embeddings = OllamaEmbeddings(model="llama3.2:latest")

file_path = "C:/Users/gmlab/Desktop/rag/bitcoin.pdf"
loader = PyPDFLoader(file_path)
data = loader.load_and_split()


url=""
api_key="


qdrant_client = QdrantClient(
    url="",
    api_key=""
)

print(qdrant_client.get_collections())



qdrant = QdrantVectorStore.from_documents(
    data,
    embeddings,
    url=url,
    prefer_grpc=True,
    api_key=api_key,
    collection_name="bitcoin",
)
