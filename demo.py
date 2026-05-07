import os
import argparse
from rag import ConfidenceCalibratedRAG

def main():
    parser = argparse.ArgumentParser(description="Confidence-Calibrated RAG Demo")
    parser.add_argument("--data", type=str, default="../data/corpus.txt", help="Path to data file")
    parser.add_argument("--query", type=str, help="Query to run")
    parser.add_argument("--naive", action="store_true", help="Run naive RAG (disable abstention)")
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(__file__), args.data)
    print("Initializing RAG Pipeline...")
    
    # Needs OPENAI_API_KEY environment variable set
    try:
        rag = ConfidenceCalibratedRAG(data_path=data_path, default_threshold=0.75)
    except Exception as e:
        print(f"Error initializing RAG: {e}")
        print("Please ensure OPENAI_API_KEY is set in your environment.")
        return

    queries = [args.query] if args.query else [
        "When was Columbia University founded?",
        "What is DeepEval?",
        "What is the weather in New York today?"  # This should trigger abstention
    ]

    print("\n" + "="*50)
    print(f"Running mode: {'NAIVE RAG' if args.naive else 'CONFIDENCE-CALIBRATED RAG'}")
    print("="*50)

    for q in queries:
        print(f"\n[Question] {q}")
        try:
            answer, docs = rag.query(q, enable_abstention=not args.naive)
            print(f"[Answer]   {answer}")
            if docs:
                print(f"[Context]  Top doc score: {docs[0][1]:.3f}")
        except Exception as e:
            print(f"[Error]    {e}")

if __name__ == "__main__":
    main()
