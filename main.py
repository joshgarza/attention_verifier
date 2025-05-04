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
from src.nli_checker import get_nli_judgment

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
    # Initialize results
    pipeline_results = {
        "question": question,
        "answer_text": "(Pipeline did not run fully)",
        "evidence_spans_with_scores": [(0.0, "(Pipeline did not run fully)")],
        "nli_judgment": "NOT_RUN",
        "confidence_score": 0.0
    }

    # 1. Load Document
    print(f"\n[1/5] Loading document: {doc_path}") # Step count increases
    document_text = load_document(doc_path)
    if not document_text:
        print("Failed to load document. Exiting.")
        pipeline_results["answer_text"] = "(Failed to load document)"
        return pipeline_results # Return partial results
    print(f"Document loaded ({len(document_text)} characters).")

    # Construct the prompt
    prompt_text = f"""Document:
    {document_text}

    Question: {question}

    Answer:"""

    # 2. Load Model and Tokenizer
    print("\n[2/5] Loading model and tokenizer...")
    model, tokenizer = None, None # Initialize
    try:
        model, tokenizer = load_model_and_tokenizer()
        if not model or not tokenizer: raise RuntimeError("Model or Tokenizer failed to load.")
        print("Model and tokenizer ready.")
    except Exception as e:
        print(f"Fatal Error: Could not load model/tokenizer. {e}")
        pipeline_results["answer_text"] = f"(Fatal Error loading model: {e})"
        return pipeline_results

    # 3. Get Answer and Attention from Model
    print("\n[3/5] Running model for answer and attention...")
    answer_text, attention_weights = get_answer_and_attention(
        model, tokenizer, document_text, question
    )
    pipeline_results["answer_text"] = answer_text # Store the answer
    print("Model inference complete.")

    # Check if inference itself returned an error message
    if answer_text.startswith("Error"):
        print("Model inference failed, skipping attention processing and NLI check.")
        pipeline_results["evidence_spans_with_scores"] = [(0.0, "(Model inference failed)")]
        pipeline_results["nli_judgment"] = "NOT_APPLICABLE"
        return pipeline_results

    # 4. Process Attention to Find Evidence
    print("\n[4/5] Processing attention weights...")
    evidence_spans_with_scores = [(0.0, "(Attention processing failed)")] # Default
    if attention_weights is not None:
        tokenizer_output_dict = prepare_tokenizer_info_for_attention(
            tokenizer, prompt_text, answer_text, attention_weights
        )
        if tokenizer_output_dict:
            # find_evidence_spans returns list of (score, sentence) tuples
            evidence_spans_with_scores = find_evidence_spans(
                attention_weights,
                tokenizer_output_dict,
                prompt_text,
                top_k=config.TOP_K_EVIDENCE
            )
            if not evidence_spans_with_scores: # Handle empty list case
                evidence_spans_with_scores = [(0.0, "(No evidence spans found by attention)")]
        else:
            evidence_spans_with_scores = [(0.0, "(Could not prepare tokenizer info)")]
    else:
        evidence_spans_with_scores = [(0.0, "(Attention weights not available)")]
    pipeline_results["evidence_spans_with_scores"] = evidence_spans_with_scores
    print("Attention processing step complete.")

    # --- 5. Perform NLI Check ---
    print("\n[5/5] Performing NLI check...")
    nli_judgment = "ERROR" # Default
    # Proceed only if we have evidence and a non-error answer
    is_evidence_placeholder = any("(Attention" in span[1] or "(Cannot process" in span[1] for span in evidence_spans_with_scores)
    if not is_evidence_placeholder and not answer_text.startswith("Error"):
        # Extract just the sentence text for the NLI premise
        evidence_texts = [span[1] for span in evidence_spans_with_scores]
        nli_judgment = get_nli_judgment(
            model, # Reuse the loaded model
            tokenizer,
            evidence_texts,
            answer_text
        )
    elif is_evidence_placeholder:
         nli_judgment = "NOT_APPLICABLE (No valid evidence)"
    else: # Answer was error
         nli_judgment = "NOT_APPLICABLE (Answer generation failed)"
    pipeline_results["nli_judgment"] = nli_judgment
    print(f"NLI Judgment: {nli_judgment}")

    # Calculate Confidence Score based on NLI
    if nli_judgment == "ENTAILMENT":
        pipeline_results["confidence_score"] = 0.9 # High confidence
    elif nli_judgment == "NEUTRAL":
        pipeline_results["confidence_score"] = 0.5 # Medium confidence
    elif nli_judgment == "CONTRADICTION":
        pipeline_results["confidence_score"] = 0.1 # Low confidence (high hallucination likelihood)
    else: # ERROR or NOT_APPLICABLE
        pipeline_results["confidence_score"] = 0.0 # Or None? 0.0 indicates low/failed confidence check


    return pipeline_results


# --- Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Attention-Guided Response Verification + NLI Check."
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
            results = run_real_pipeline(args.doc_path, args.question)
            doc_text_for_report = load_document(args.doc_path) or "(Could not reload document for report)"

    except Exception as e:
        # Catch unexpected errors during pipeline execution
        print(f"\n--- UNEXPECTED PIPELINE ERROR ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        # Assign error messages for final display
        results["answer_text"] = f"(Pipeline Error: {e})"
        results["evidence_spans_with_scores"] = [(0.0, "(Pipeline failed)")] # Use tuple format
        results["nli_judgment"] = "ERROR"
        results["confidence_score"] = 0.0

    # Ensure final variables have values before displaying
    final_question = results.get("question", "N/A")
    final_answer = results.get("answer_text", "(Error processing answer)")
    # Ensure evidence is a list of (score, text) tuples before passing
    final_evidence_raw = results.get("evidence_spans_with_scores", [(0.0,"(Error processing evidence)")])
    final_evidence_tuples = []
    if isinstance(final_evidence_raw, list):
        for item in final_evidence_raw:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                final_evidence_tuples.append((float(item[0]) if isinstance(item[0], (int, float)) else 0.0, str(item[1])))
            else: # Handle unexpected items
                final_evidence_tuples.append((0.0, f"(Invalid evidence item: {item})"))
    else: # Handle case where evidence is not a list
         final_evidence_tuples.append((0.0, f"(Invalid evidence format: {type(final_evidence_raw)})"))


    final_nli = results.get("nli_judgment", "N/A")
    final_confidence = results.get("confidence_score", "N/A")


    # Generate the report
    generate_html_report(
        final_question,
        final_answer,
        final_evidence_tuples, # Pass the cleaned list of tuples
        doc_text_for_report,
        nli_judgment=final_nli,
        confidence_score=final_confidence,
        output_filename="verification_report.html"
    )
    # Display/Save using formatter (formatter needs update to show NLI score)
    display_results(
        results.get("question", "N/A"),
        results.get("answer_text", "(Error)"),
        results.get("evidence_spans_with_scores", [(0.0,"(Error)")])
    )

    overall_end_time = time.time()
    print(f"\n--- Total script execution time: {overall_end_time - overall_start_time:.2f} seconds ---")