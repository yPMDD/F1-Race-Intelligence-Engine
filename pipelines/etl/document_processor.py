import os
import sys
import logging

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = "data/raw"
CHROMA_DIR = "data/chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def process_documents():
    if not os.path.exists(RAW_DATA_DIR):
        logger.error(f"Raw data directory '{RAW_DATA_DIR}' does not exist.")
        return

    logger.info(f"Loading PDFs from {RAW_DATA_DIR}...")
    loader = PyPDFDirectoryLoader(RAW_DATA_DIR)
    documents = loader.load()

    if not documents:
        logger.warning(f"No documents found in {RAW_DATA_DIR}. Please add FIA PDFs.")
        return

    logger.info(f"Loaded {len(documents)} pages. Cleaning and splitting text...")
    
    import re
    def clean_text(text):
        # Fix split numbers followed by units: "72 6kg" -> "726kg"
        text = re.sub(r'(\d)\s+(\d)(?=kg|mm|cm|km|m/s|kW|V|A)', r'\1\2', text)
        # Fix common PDF hyphenation issues
        text = text.replace('− ', '-')
        return text

    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Generated {len(chunks)} text chunks.")

    logger.info(f"Initializing HuggingFace embeddings ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logger.info(f"Upserting chunks into ChromaDB at {CHROMA_DIR}...")
    # Initialize Chroma and add documents (this creates/updates the persistent DB)
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_DIR
    )
    
    logger.info("RAG ingestion complete!")

if __name__ == "__main__":
    process_documents()
