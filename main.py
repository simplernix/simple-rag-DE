# main.py
from ingest import ingest_pdfs
from query import LocalRAG
import sys

def main():
    rag = None
    print("🚀 Local RAG System Started (using Ollama + Llama3)\n")
    
    while True:
        print("\n" + "-" * 60)
        print("MAIN MENU")
        print("-" * 60)
        print("1. Ingest / Re-index PDFs")
        print("2. Ask a Question")
        print("3. Exit")
        print("-" * 60)
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\nStarting document ingestion...")
            ingest_pdfs()
            print("\n✅ Documents indexed successfully!")
            # Reload RAG after ingestion
            try:
                rag = LocalRAG()
                print("✅ RAG system ready with new documents.")
            except Exception as e:
                print(f"⚠️  Could not load RAG after ingestion: {e}")
            
        elif choice == "2":
            if rag is None:
                try:
                    rag = LocalRAG()
                except FileNotFoundError:
                    print("❌ No index found. Please ingest documents first using Option 1.")
                    continue
                except Exception as e:
                    print(f"❌ Error loading RAG system: {e}")
                    continue
            
            question = input("\nEnter your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye!")
                break
                
            if question:
                rag.query(question)
            else:
                print("Please enter a valid question.")
                
        elif choice == "3":
            print("👋 Thank you for using Local RAG System. Goodbye!")
            break
            
        else:
            print("⚠️  Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()