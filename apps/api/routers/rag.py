from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import logging

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rag",
    tags=["rag"],
    responses={404: {"description": "Not found"}},
)

CHROMA_DIR = "data/chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Lazy initialization of vector store
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        logger.info("Initializing Chroma vector store client...")
        try:
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            _vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {str(e)}")
            raise HTTPException(status_code=500, detail="Vector store unavailable")
    return _vectorstore

class QueryRequest(BaseModel):
    query: str
    top_k: int = 4

class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    query: str
    results: List[DocumentChunk]

@router.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Query the RAG vector store for relevant document chunks based on semantic similarity.
    """
    logger.info(f"RAG query received: '{request.query}'")
    vectorstore = get_vectorstore()
    
    try:
        # Perform similarity search
        docs = vectorstore.similarity_search(request.query, k=request.top_k)
        
        results = []
        for doc in docs:
            results.append(DocumentChunk(
                content=doc.page_content,
                metadata=doc.metadata
            ))
            
        return QueryResponse(query=request.query, results=results)
        
    except Exception as e:
        logger.error(f"Error during similarity search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
