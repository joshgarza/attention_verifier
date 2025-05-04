# app.py
import time
import os
import json
import uuid # For unique job IDs
from flask import Flask, request, jsonify, render_template, send_from_directory

# Local project imports
import config
from src.data_loader import load_document
from src.model_handler import load_model_and_tokenizer, get_answer_and_attention # Llama 3 NLI
from src.llama4_api_client import get_nli_judgment_via_api # Llama 4 API NLI
from src.tokenizer_utils import prepare_tokenizer_info_for_attention
from src.attention_processor import find_evidence_spans
from src.output_formatter import generate_html_report

# --- Configuration ---
# Decide which NLI function to use (e.g., based on config or environment variable)
USE_LLAMA4_API_NLI = True # Set to False to use local Llama 3 NLI
REPORTS_DIR = "reports"
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here' # Good practice, though less critical for local hackathon

# --- Simple In-Memory Storage ---
# WARNING: This is NOT suitable for production! Data is lost on restart.
# Stores intermediate results between inference and grading.
results_store = {}

# --- Load Model Globally? (Hackathon Simplification: Load on demand) ---
# For simplicity in a hackathon, we'll load the model inside the request.
# A better approach for performance would be to load once globally,
# but that requires careful handling of state and potential concurrency issues.
# model_global, tokenizer_global = None, None
# def get_model_tokenizer():
#     global model_global, tokenizer_global
#     if model_global is None or tokenizer_global is None:
#         print("Loading model/tokenizer globally...")
#         model_global, tokenizer_global = load_model_and_tokenizer()
#     return model_global, tokenizer_global

# --- Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/run_inference', methods=['POST'])
def handle_run_inference():
    """
    Handles the first step: Load doc, run Llama 3 inference, store results.
    """
    start_req_time = time.time()
    print("\n--- Received /run_inference request ---")
    doc_content = request.form.get('document', '')
    question = request.form.get('question', '')

    if not doc_content or not question:
        return jsonify({"status": "error", "message": "Document and question are required."}), 400

    # --- Mimic pipeline steps ---
    # (No separate doc loading needed as content is passed directly)
    document_text = doc_content # Use passed content

    # Construct prompt
    prompt_text = f"""Document:
{document_text}

Question: {question}

Answer:"""

    # Load Model/Tokenizer (on demand - will be slow first time)
    print("Loading model/tokenizer for inference...")
    try:
        # Using on-demand loading for simplicity
        model, tokenizer = load_model_and_tokenizer()
        if not model or not tokenizer:
            raise RuntimeError("Model or Tokenizer could not be loaded.")
    except Exception as e:
        print(f"Fatal Error loading model: {e}")
        return jsonify({"status": "error", "message": f"Error loading model: {e}"}), 500

    # Run Inference (Llama 3)
    print("Running Llama 3 inference...")
    answer_text, attention_weights = get_answer_and_attention(
        model, tokenizer, document_text, question
    )
    print(f"Inference completed. Answer generated.")

    # Check for inference errors stored in answer_text
    if answer_text.startswith("Error"):
        print(f"Inference returned an error state: {answer_text}")
        # Decide if you want to return error or proceed without grading possible
        # Let's return error for clarity
        return jsonify({"status": "error", "message": answer_text}), 500


    # Generate unique ID and store necessary data
    job_id = str(uuid.uuid4())
    results_store[job_id] = {
        "timestamp": time.time(),
        "question": question,
        "answer_text": answer_text,
        "attention_weights": attention_weights, # Might be large! Consider if needed later
        "prompt_text": prompt_text,
        "document_text": document_text, # Store original doc text
        # Storing tokenizer isn't practical, assume it's loaded again if needed
    }
    print(f"Stored results for job_id: {job_id}")

    end_req_time = time.time()
    print(f"--- /run_inference request completed in {end_req_time - start_req_time:.2f}s ---")

    return jsonify({
        "status": "inference_complete",
        "job_id": job_id,
        "answer": answer_text
    })

@app.route('/grade_response', methods=['POST'])
def handle_grade_response():
    """
    Handles the second step: Process attention, run NLI check, generate report.
    """
    start_req_time = time.time()
    print("\n--- Received /grade_response request ---")
    job_id = request.form.get('job_id')

    if not job_id or job_id not in results_store:
        return jsonify({"status": "error", "message": "Invalid or expired job ID."}), 400

    # Retrieve stored data
    stored_data = results_store[job_id]
    question = stored_data["question"]
    answer_text = stored_data["answer_text"]
    attention_weights = stored_data["attention_weights"]
    prompt_text = stored_data["prompt_text"]
    document_text = stored_data["document_text"] # Needed for report context

    # Reload tokenizer (needed for preparing info) - inefficient but simple
    # Alternatively, pass tokenizer info directly if prepared in step 1
    print("Reloading tokenizer for processing...")
    try:
        # Using on-demand loading for simplicity
        _, tokenizer = load_model_and_tokenizer()
        if not tokenizer: raise RuntimeError("Tokenizer could not be loaded.")
    except Exception as e:
        print(f"Fatal Error loading tokenizer for grading: {e}")
        return jsonify({"status": "error", "message": f"Error loading tokenizer: {e}"}), 500

    # --- Mimic pipeline steps ---
    # 1. Prepare Tokenizer Info
    print("Preparing tokenizer info...")
    tokenizer_output_dict = prepare_tokenizer_info_for_attention(
        tokenizer, prompt_text, answer_text, attention_weights
    )

    # 2. Process Attention
    print("Processing attention...")
    evidence_spans_with_scores = [(0.0, "(Processing Error)")] # Default
    if attention_weights is not None and tokenizer_output_dict is not None:
         required_keys = ['prompt_len', 'answer_len', 'all_token_offsets', 'document_token_indices', 'doc_content_start_char', 'doc_content_end_char']
         if all(k in tokenizer_output_dict and tokenizer_output_dict[k] is not None for k in required_keys):
              evidence_spans_with_scores = find_evidence_spans(
                  attention_weights, tokenizer_output_dict, prompt_text, config.TOP_K_EVIDENCE
              )
              if not evidence_spans_with_scores: evidence_spans_with_scores = [(0.0, "(No evidence spans found)")]
         else:
              evidence_spans_with_scores = [(0.0, "(Missing tokenizer info)")]
    elif attention_weights is None:
         evidence_spans_with_scores = [(0.0, "(Attention weights unavailable)")]
    else: # tokenizer_output_dict is None
         evidence_spans_with_scores = [(0.0, "(Tokenizer info prep failed)")]


    # 3. Perform NLI Check (Conditional - Llama 4 API or Llama 3 Local)
    print("Performing NLI check...")
    nli_judgment = "ERROR"
    is_evidence_placeholder = any("(Attention" in span[1] or "(Cannot process" in span[1] for span in evidence_spans_with_scores) # Basic check
    if not is_evidence_placeholder and not answer_text.startswith("Error"):
        evidence_texts = [span[1] for span in evidence_spans_with_scores] # Extract text only
        if USE_LLAMA4_API_NLI:
            print("Using Llama 4 API for NLI...")
            nli_judgment = get_nli_judgment_via_api(
                premise=" ".join(evidence_texts), # Combine evidence for premise
                hypothesis=answer_text
            )
        else:
            print("Using Local Llama 3 for NLI...")
            # Need model for local NLI check - must reload or have cached
            try:
                model, _ = load_model_and_tokenizer() # Reload model if not global
                if model:
                     nli_judgment = get_nli_judgment(
                         model, tokenizer, evidence_texts, answer_text
                     )
                else:
                     print("Error: Model not available for local NLI check.")
                     nli_judgment = "ERROR (Model Load Fail)"
            except Exception as e:
                 print(f"Error during local NLI check: {e}")
                 nli_judgment = f"ERROR ({e})"
    else:
        nli_judgment = "NOT_APPLICABLE"
    print(f"NLI Judgment: {nli_judgment}")


    # 4. Calculate Confidence Score
    confidence_score = 0.0
    if nli_judgment == "ENTAILMENT": confidence_score = 0.9
    elif nli_judgment == "NEUTRAL": confidence_score = 0.5
    elif nli_judgment == "CONTRADICTION": confidence_score = 0.1
    # All others (ERROR, API_ERROR, NOT_APPLICABLE) default to 0.0
    print(f"Calculated Confidence Score: {confidence_score}")

    # 5. Generate HTML Report
    report_filename_base = f"report_{job_id}.html"
    report_filepath = os.path.join(REPORTS_DIR, report_filename_base)
    generate_html_report(
        question,
        answer_text,
        evidence_spans_with_scores,
        document_text, # Pass full original document text
        nli_judgment=nli_judgment,
        confidence_score=confidence_score,
        output_filename=report_filepath
    )

    # Clean up stored data for this job ID (optional, prevents memory leak)
    if job_id in results_store:
        del results_store[job_id]

    end_req_time = time.time()
    print(f"--- /grade_response request completed in {end_req_time - start_req_time:.2f}s ---")

    return jsonify({
        "status": "grading_complete",
        "report_path": f"/report/{job_id}" # Relative path for browser link
    })

@app.route('/report/<job_id>')
def serve_report(job_id):
    """Serves the generated static HTML report."""
    # Basic security check (optional): validate job_id format if needed
    filename = f"report_{job_id}.html"
    print(f"Serving report: {filename}")
    return send_from_directory(REPORTS_DIR, filename)

# --- Run the App ---
if __name__ == '__main__':
    # Make accessible on the network, use a port unlikely to conflict
    app.run(debug=False, host='0.0.0.0', port=5001)