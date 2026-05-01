from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

CHROMA_DIR = "data/chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def check_db():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    print(f"Total documents in Chroma: {len(vectorstore.get()['ids'])}")
    
    # Check a few documents to see metadata
    docs = vectorstore.get(limit=5)
    for i in range(len(docs['ids'])):
        print(f"ID: {docs['ids'][i]}")
        print(f"Metadata: {docs['metadatas'][i]}")
        print("-" * 20)

if __name__ == "__main__":
    check_db()
