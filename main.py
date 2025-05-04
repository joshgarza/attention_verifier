# main.py
import argparse
import time
import sys
import torch # Still needed for tuple check

# Local project imports
import config
from src.data_loader import load_document
from src.model_handler import load_model_and_tokenizer, get_answer_and_attention
from src.tokenizer_utils import prepare_tokenizer_info_for_attention
from src.attention_processor import find_evidence_spans
from src.output_formatter import display_results, generate_html_report
# Import dummy data functions only if needed for testing/dummy run
from src.utils import create_dummy_attention, create_dummy_tokenizer_output


def run_dummy_pipeline():
    """Runs the pipeline using dummy data."""
    print("\n[*] --- Generating Dummy Data ---")
    dummy_doc = "This is the first sentence. Here is the second sentence which contains important info. The third sentence."
    dummy_question = "What info is in the second sentence?"
    dummy_answer = "The second sentence has info."
    prompt_text = f"Document:\n{dummy_doc}\n\nQuestion: {dummy_question}\n\nAnswer:"

    print("[*] Generating dummy tokenizer info...")
    tokenizer_output_dict = create_dummy_tokenizer_output(prompt_text, dummy_answer)
    if not tokenizer_output_dict:
        print("Error: Failed to create dummy tokenizer output.")
        return None, None, None, None # Return Nones

    print("[*] Generating dummy attention weights...")
    num_layers = 18; num_heads = 32 # Example
    prompt_len = tokenizer_output_dict['prompt_len']
    answer_len = tokenizer_output_dict['answer_len']
    attention_weights = create_dummy_attention(num_layers, num_heads, prompt_len, answer_len)
    print("[*] --- Dummy Data Generation Complete ---")

    print("\n[*] Processing dummy attention weights...")
    evidence_spans = find_evidence_spans(
        attention_weights,
        tokenizer_output_dict,
        prompt_text,
        top_k=config.TOP_K_EVIDENCE
    )
    print("[*] Dummy attention processing complete.")

    return dummy_question, dummy_answer, evidence_spans


def run_real_pipeline(doc_path, question):
    """Runs the main verification pipeline with the real model."""

    # 1. Load Document
    print(f"\n[1/4] Loading document: {doc_path}")
    document_text = load_document(doc_path)
    if not document_text:
        print("Failed to load document. Exiting.")
        return None, "(Failed to load document)", None # Return error state
    print(f"Document loaded ({len(document_text)} characters).")

    # Construct the prompt
    prompt_text = f"""Document:
    {document_text}

    Question: {question}

    Answer:"""

    # 2. Load Model and Tokenizer
    print("\n[2/4] Loading model and tokenizer...")
    try:
        model, tokenizer = load_model_and_tokenizer()
        if not model or not tokenizer:
            raise RuntimeError("Model or Tokenizer failed to load.")
        print("Model and tokenizer ready.")
    except Exception as e:
        print(f"Fatal Error: Could not load model/tokenizer. {e}")
        return question, f"(Fatal Error loading model: {e})", None

    # 3. Get Answer and Attention from Model
    print("\n[3/4] Running model for answer and attention...")
    answer_text, attention_weights = get_answer_and_attention(
        model, tokenizer, document_text, question
    )
    print("Model inference complete.")

    # Check if inference itself returned an error message
    if answer_text.startswith("Error"):
        print("Model inference failed, skipping attention processing.")
        return question, answer_text, ["(Model inference failed)"]

    # 4. Process Attention to Find Evidence
    print("\n[4/4] Processing attention weights...")
    evidence_spans = []
    if attention_weights is not None:
        tokenizer_output_dict = prepare_tokenizer_info_for_attention(
            tokenizer, prompt_text, answer_text, attention_weights
        )
        if tokenizer_output_dict:
            evidence_spans = find_evidence_spans(
                attention_weights,
                tokenizer_output_dict,
                prompt_text,
                top_k=config.TOP_K_EVIDENCE
            )
        else:
            evidence_spans = ["(Could not prepare tokenizer info for attention processing)"]
    else:
        evidence_spans = ["(Attention weights not available from model)"]
    print("Attention processing step complete.")

    return question, answer_text, evidence_spans


# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Attention-Guided Response Verification on a document."
    )
    parser.add_argument(
        '--dummy-run',
        action='store_true',
        help="Run using dummy data for testing attention processing without loading the model."
    )
    parser.add_argument("doc_path", nargs='?', default=None, help="Path to the input document file (required if not --dummy-run)")
    parser.add_argument("question", nargs='?', default=None, help="The question to ask about the document (required if not --dummy-run)")
    args = parser.parse_args()

    # Validate arguments for real run
    if not args.dummy_run and (args.doc_path is None or args.question is None):
        parser.error("doc_path and question are required when not using --dummy-run")
        sys.exit(1)

    overall_start_time = time.time()
    q_final, answer_final, evidence_final = None, None, None

    try:
        if args.dummy_run:
            # Use specific function for dummy pipeline
            q_final, answer_final, evidence_final = run_dummy_pipeline()
            # Assign args.question if needed for display, though dummy q is used
            if q_final is None: q_final = "Dummy Question"
        else:
            # Use specific function for real pipeline
            q_final, answer_final, evidence_final = run_real_pipeline(args.doc_path, args.question)
            doc_text_for_report = load_document(args.doc_path) or "(Could not reload document for report)"

    except Exception as e:
        # Catch unexpected errors during pipeline execution
        print(f"\n--- UNEXPECTED PIPELINE ERROR ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        # Assign error messages for final display
        q_final = args.question if not args.dummy_run else "Dummy Question"
        answer_final = f"(Pipeline Error: {e})"
        evidence_final = ["(Pipeline failed with unexpected error)"]

    # Ensure final variables have values before displaying
    if q_final is None: q_final = args.question if args.question else "N/A"
    if answer_final is None: answer_final = "(Processing error)"
    if evidence_final is None: evidence_final = ["(Processing error)"]

    # Display results using the formatter
    display_results(q_final, answer_final, evidence_final)
    generate_html_report(
        q_final,
        answer_final,
        evidence_final, # Pass the list of (score, sentence) tuples
        doc_text_for_report,
        output_filename="verification_report.html" # Output filename
    )

    overall_end_time = time.time()
    print(f"\n--- Total script execution time: {overall_end_time - overall_start_time:.2f} seconds ---")