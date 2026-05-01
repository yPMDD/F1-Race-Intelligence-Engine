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

    import shutil
    if os.path.exists(CHROMA_DIR):
        logger.info(f"Clearing existing ChromaDB at {CHROMA_DIR}...")
        shutil.rmtree(CHROMA_DIR)
        
    all_chunks = []
    
    # Process each year subdirectory
    for year_dir in os.listdir(RAW_DATA_DIR):
        year_path = os.path.join(RAW_DATA_DIR, year_dir)
        if not os.path.isdir(year_path):
            continue
            
        logger.info(f"Loading PDFs for Year {year_dir} from {year_path}...")
        loader = PyPDFDirectoryLoader(year_path)
        documents = loader.load()

        if not documents:
            logger.warning(f"No documents found in {year_path}.")
            continue

        import re
        def clean_text(text):
            text = re.sub(r'(\d)\s+(\d)(?=kg|mm|cm|km|m/s|kW|V|A)', r'\1\2', text)
            text = text.replace('− ', '-')
            return text

        for doc in documents:
            doc.page_content = clean_text(doc.page_content)
            # Add year metadata
            doc.metadata['year'] = int(year_dir) if year_dir.isdigit() else 2026
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        year_chunks = text_splitter.split_documents(documents)
        all_chunks.extend(year_chunks)
        logger.info(f"Generated {len(year_chunks)} chunks for {year_dir}.")

    if not all_chunks:
        logger.error("No chunks generated from any directory.")
        return

    logger.info(f"Initializing HuggingFace embeddings ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    logger.info(f"Upserting {len(all_chunks)} chunks into ChromaDB at {CHROMA_DIR}...")
    vectorstore = Chroma.from_documents(
        documents=all_chunks, 
        embedding=embeddings, 
        persist_directory=CHROMA_DIR
    )
    
    logger.info("Multi-year RAG ingestion complete!")

if __name__ == "__main__":
    process_documents()
