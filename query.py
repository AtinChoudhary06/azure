"""
Quick terminal test of the ask_question pipeline, without needing the
FastAPI server or Streamlit UI running.
Usage:  python scripts/cli_query.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import rag

if __name__ == "__main__":
    print("RAG Chatbot ready. Type 'exit' to quit.\n")
    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            break
        answer = rag.ask_question(question)
        print(f"\nAnswer: {answer}\n")