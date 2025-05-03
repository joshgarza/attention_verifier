import argparse
import time
import config
from src.data_loader import load_document
from src.model_handler import load_model_and_tokenizer, get_answer_and_attention
from src.attention_processor import find_evidence_spans

def run_verification(doc_path, question):
    """Runs the full attention-guided verification pipeline."""
    print("--- Starting Attention-Guided Verification ---")
    start_time = time.time()

    # 1. Load Document
    print(f"\n[1/4] Loading document: {doc_path}")
    document_text = load_document(doc_path)
    if not document_text:
        print("Failed to load document. Exiting.")
        return
    print(f"Document loaded ({len(document_text)} characters).")

    # 2. Load Model and Tokenizer (only once)
    print("\n[2/4] Loading model and tokenizer...")
    try:
        model, tokenizer = load_model_and_tokenizer()
        if not model or not tokenizer:
            raise RuntimeError("Model or Tokenizer failed to load.")
        print("Model and tokenizer ready.")
    except Exception as e:
        print(f"Fatal Error: Could not load model/tokenizer. {e}")
        return

    # 3. Get Answer and Attention from Model
    print("\n[3/4] Running model for answer and attention...")
    try:
        answer_text, attention_weights = get_answer_and_attention(
            model, tokenizer, document_text, question
        )
        print("Model inference complete.")
    except Exception as e:
        print(f"Error during model inference: {e}")
        # Decide if you want to continue without attention or exit
        answer_text = f"Error during inference: {e}"
        attention_weights = None
        # return # Uncomment to exit on inference error

    # 4. Process Attention to Find Evidence
    print("\n[4/4] Processing attention weights...")
    evidence_spans = find_evidence_spans(
        attention_weights, tokenizer, document_text, answer_text
    )
    print("Attention processing complete.")


    # --- Output Results ---
    print("\n--- Verification Results ---")
    print(f"Question: {question}")
    print(f"\nModel Answer:\n{answer_text}") # Might need cleaning/parsing

    print("\nPotential Evidence Spans (Based on Attention Analysis):")
    if evidence_spans:
        for i, span in enumerate(evidence_spans):
            # Basic formatting, improve as needed
            print(f"Evidence {i+1}:\n---\n{span}\n---")
    else:
        print("Could not identify evidence spans.")

    end_time = time.time()
    print(f"\n--- Process completed in {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Attention-Guided Response Verification on a document."
    )
    parser.add_argument(
        "doc_path",
        help="Path to the input document file (e.g., data/sample_document.txt)"
    )
    parser.add_argument(
        "question",
        help="The question to ask about the document."
    )
    args = parser.parse_args()

    run_verification(args.doc_path, args.question)