# query.py
import faiss
import pickle
import ollama
import os
from sentence_transformers import SentenceTransformer
from config import *

class LocalRAG:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        
        index_path = os.path.join(VECTOR_STORE_PATH, "index.faiss")
        chunks_path = os.path.join(VECTOR_STORE_PATH, "chunks.pkl")
        metadata_path = os.path.join(VECTOR_STORE_PATH, "metadata.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError("Vector index not found. Please ingest documents first (Option 1).")
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load chunks and metadata
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
    
    def retrieve(self, query: str, k: int = TOP_K):
        """Retrieve top-k relevant chunks"""
        query_embedding = self.model.encode([query])[0].astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        
        retrieved = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                retrieved.append({
                    "text": self.chunks[idx],
                    "metadata": self.metadata[idx]
                })
        return retrieved
    
    def generate_answer(self, query: str, context_chunks: list):
        """Generate well-reframed answer using Ollama"""
        if not context_chunks:
            return "Insufficient information in the provided documents."

        # Build clean context
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            context_parts.append(
                f"Source {i+1} (Document: {chunk['metadata']['source']}, Page {chunk['metadata']['page']}):\n"
                f"{chunk['text']}"
            )
        
        context = "\n\n".join(context_parts)

        prompt = f"""You are a helpful, precise, and professional assistant.

Your task is to answer the user's question by **rephrasing and structuring** the information from the given context.
Rules:
- Do NOT copy-paste sentences directly from the context.
- Rephrase the information in clear, natural, and fluent language.
- Make the answer well-structured, concise, and easy to read.
- Use bullet points or short paragraphs where appropriate.
- Only use information present in the context. Do not add external knowledge.
- If the context does not have enough information, clearly state it.

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    "temperature": 0.2,
                    "num_ctx": 8192,
                    "top_p": 0.95,
                }
            )
            return response['message']['content'].strip()
        
        except Exception as e:
            return f"Error while generating answer: {str(e)}"

    def query(self, question: str):
        """Main query function with improved UX"""
        print(f"\n🔍 Searching for relevant information...")
        
        contexts = self.retrieve(question)
        
        if not contexts:
            print("❌ No relevant information found in the documents.")
            return "Insufficient information in the provided documents."

        print(f"✅ Retrieved {len(contexts)} relevant chunks. Generating response...\n")
        
        answer = self.generate_answer(question, contexts)
        
        # Final Output
        print("=" * 75)
        print("📝 FINAL ANSWER")
        print("=" * 75)
        print(answer)
        print("=" * 75)
        
        # Show Sources
        print("\n📚 Sources Used:")
        for i, ctx in enumerate(contexts, 1):
            print(f"   {i}. {ctx['metadata']['source']} (Page {ctx['metadata']['page']})")
        
        return answer