# ingest.py
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from tqdm import tqdm
from config import *

def clean_text(text: str) -> str:
    """Clean noisy text from PDFs"""
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip page numbers and very short lines that look like headers/footers
        if line.replace(' ', '').isdigit() or len(line) < 8:
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)


def ingest_pdfs(pdf_folder: str = "documents"):
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    
    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    
    all_chunks = []
    metadata_list = []
    
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in 'documents/' folder!")
        return
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        path = os.path.join(pdf_folder, pdf_file)
        try:
            doc = fitz.open(path)
            
            for page_num in range(len(doc)):
                page_text = doc[page_num].get_text("text")
                cleaned_text = clean_text(page_text)
                
                if len(cleaned_text.strip()) < 50:
                    continue
                
                chunks = text_splitter.split_text(cleaned_text)
                
                for chunk in chunks:
                    if len(chunk.strip()) > 30:   # avoid tiny chunks
                        all_chunks.append(chunk)
                        metadata_list.append({
                            "source": pdf_file,
                            "page": page_num + 1
                        })
            
            doc.close()
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")
    
    if not all_chunks:
        print("No valid text extracted from PDFs.")
        return
    
    print(f"\nCreated {len(all_chunks)} chunks from {len(pdf_files)} PDF(s)")
    
    # Generate embeddings
    print("Generating embeddings... This may take a while.")
    embeddings = model.encode(all_chunks, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create and save FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    faiss.write_index(index, os.path.join(VECTOR_STORE_PATH, "index.faiss"))
    
    with open(os.path.join(VECTOR_STORE_PATH, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)
    with open(os.path.join(VECTOR_STORE_PATH, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata_list, f)
    
    print(f"✅ Ingestion completed! {len(all_chunks)} chunks indexed.")