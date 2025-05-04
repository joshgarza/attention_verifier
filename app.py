# app.py
import time
import os
import json
import uuid
from flask import Flask, request, jsonify, render_template, send_from_directory
import torch # Needed for device checks etc.
import gc

# Local project imports
import config
from src.data_loader import load_document # Keep this if loading from file path later
from src.model_handler import load_model_and_tokenizer, get_answer_and_attention # Llama 3 NLI
from src.llama4_api_client import get_nli_judgment_via_api # Llama 4 API NLI
from src.tokenizer_utils import prepare_tokenizer_info_for_attention
from src.attention_processor import find_evidence_spans
from src.output_formatter import generate_html_report

# --- Configuration ---
USE_LLAMA4_API_NLI = True # Set to False to use local Llama 3 NLI
REPORTS_DIR = "reports"
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# --- Flask App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-for-hackathon'

# --- Global Variables for Model/Tokenizer ---
# These will be loaded once on startup
model_global = None
tokenizer_global = None
model_load_error = None # Store any error during initial load

# --- Simple In-Memory Storage (Hackathon context) ---
results_store = {}

# --- Function to Load Model on Startup ---
def initialize_model():
    """Loads the model and tokenizer into global variables."""
    global model_global, tokenizer_global, model_load_error
    if model_global is None or tokenizer_global is None:
        print("--- Initializing Model and Tokenizer (loading once) ---")
        try:
            model_global, tokenizer_global = load_model_and_tokenizer()
            if not model_global or not tokenizer_global:
                raise RuntimeError("load_model_and_tokenizer returned None")
            print("--- Model and Tokenizer Initialized Successfully ---")
        except Exception as e:
            print(f"!!!!!!!! FATAL ERROR DURING MODEL INITIALIZATION !!!!!!!!")
            print(f"{e}")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            model_load_error = e # Store the error
            # Set globals to None to indicate failure
            model_global = None
            tokenizer_global = None

# --- Routes ---

@app.route('/')
def index():
    """Renders the main HTML page."""
    if model_load_error:
         # Optionally render an error page or pass error message to template
         return f"Error initializing model: {model_load_error}. Please check server logs and restart.", 500
    return render_template('index.html')

@app.route('/run_inference', methods=['POST'])
def handle_run_inference():
    """
    Handles the first step: Use pre-loaded model for inference, store results.
    """
    start_req_time = time.time()
    print("\n--- Received /run_inference request ---")

    # --- Check if model loaded successfully on startup ---
    if model_load_error:
        print(f"Returning error: Model failed to initialize.")
        return jsonify({"status": "error", "message": f"Model failed to initialize: {model_load_error}"}), 500
    if model_global is None or tokenizer_global is None:
        print("Returning error: Model not loaded.")
        # This shouldn't happen if initialize_model was called, indicates startup issue
        return jsonify({"status": "error", "message": "Model not loaded. Check server startup."}), 500
    # --- Use global model and tokenizer ---
    model = model_global
    tokenizer = tokenizer_global

    doc_content = request.form.get('document', '')
    question = request.form.get('question', '')

    if not doc_content or not question:
        return jsonify({"status": "error", "message": "Document and question are required."}), 400

    document_text = doc_content

    # Construct prompt
    prompt_text = f"""Document:
    {document_text}

    Question: {question}

    Answer:"""

    # Run Inference (Llama 3 using pre-loaded model)
    print("Running Llama 3 inference...")
    answer_text, attention_weights = get_answer_and_attention(
        model, tokenizer, document_text, question
    )
    print(f"Inference completed.")

    if answer_text.startswith("Error"):
        print(f"Inference returned an error state: {answer_text}")
        return jsonify({"status": "error", "message": answer_text}), 500

    # Generate unique ID and store necessary data
    job_id = str(uuid.uuid4())
    # Critical: Avoid storing large tensors like attention_weights in memory
    # if possible, especially if many users hit this. For the hackathon,
    # maybe store just enough info to re-run parts if needed, or process immediately.
    # Let's store it for now but be aware of the memory implication.
    results_store[job_id] = {
        "timestamp": time.time(),
        "question": question,
        "answer_text": answer_text,
        "attention_weights": attention_weights, # STORED (MEMORY RISK!)
        "prompt_text": prompt_text,
        "document_text": document_text,
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
    Uses pre-loaded tokenizer (and model if local NLI).
    """
    start_req_time = time.time()
    print("\n--- Received /grade_response request ---")
    job_id = request.form.get('job_id')

    if not job_id or job_id not in results_store:
        return jsonify({"status": "error", "message": "Invalid or expired job ID."}), 400

    # --- Check if tokenizer loaded successfully on startup ---
    if model_load_error: # Check the global error flag
        print(f"Returning error: Model/Tokenizer failed to initialize.")
        return jsonify({"status": "error", "message": f"Tokenizer failed to initialize: {model_load_error}"}), 500
    if tokenizer_global is None:
        print("Returning error: Tokenizer not loaded.")
        return jsonify({"status": "error", "message": "Tokenizer not loaded. Check server startup."}), 500
    # --- Use global tokenizer ---
    tokenizer = tokenizer_global


    # Retrieve stored data
    stored_data = results_store[job_id]
    question = stored_data["question"]
    answer_text = stored_data["answer_text"]
    attention_weights = stored_data["attention_weights"]
    prompt_text = stored_data["prompt_text"]
    document_text = stored_data["document_text"]

    # --- Pipeline Steps ---
    # 1. Prepare Tokenizer Info
    print("Preparing tokenizer info...")
    tokenizer_output_dict = prepare_tokenizer_info_for_attention(
        tokenizer, prompt_text, answer_text, attention_weights
    )

    # 2. Process Attention
    print("Processing attention...")
    # ... (logic to call find_evidence_spans - same as before) ...
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

    # 3. Perform NLI Check (Conditional)
    print("Performing NLI check...")
    nli_judgment = "ERROR"
    nli_reasoning = "(NLI check not performed)" # Default reasoning
    is_evidence_placeholder = any("(Attention" in span[1] or "(Cannot process" in span[1] or "(No evidence" in span[1] for span in evidence_spans_with_scores if isinstance(span, tuple) and len(span)>1) # Updated check
    if not is_evidence_placeholder and not answer_text.startswith("Error"):
        evidence_texts = [span[1] for span in evidence_spans_with_scores if isinstance(span, tuple) and len(span)>1] # Extract text safely
        if USE_LLAMA4_API_NLI:
            print("Using Llama 4 API for NLI...")
            nli_judgment = get_nli_judgment_via_api(
                premise=" ".join(evidence_texts),
                hypothesis=answer_text
            )
        else:
            print("Using Local Llama 3 for NLI...")
            # --- Use global model ---
            if model_global:
                 try:
                     nli_judgment, nli_reasoning = get_nli_judgment_via_api(
                premise=" ".join(evidence_texts),
                hypothesis=answer_text
                 except Exception as e:
                      print(f"Error during local NLI check: {e}")
                      nli_judgment = f"ERROR ({e})"
            else:
                print("Using Local Llama 3 for NLI...")
                if model_global:
                  try:
                      # --- NOTE: Local Llama 3 NLI function needs update to return reasoning too ---
                      # Assuming get_nli_judgment is updated similarly to return (judgment, reasoning)
                      nli_judgment, nli_reasoning = get_nli_judgment(
                          model_global, tokenizer, evidence_texts, answer_text
                      )
                      # If get_nli_judgment only returns judgment:
                      # nli_judgment = get_nli_judgment(...)
                      # nli_reasoning = "(Reasoning not captured for local Llama 3 NLI)"
                  except Exception as e:
                      print(f"Error during local NLI check: {e}")
                      nli_judgment = f"ERROR ({e})"
                      nli_reasoning = f"(Error during local NLI check: {e})"
                else:
                  nli_judgment = "ERROR (Model Not Loaded)"
                  nli_reasoning = "(Local model for NLI was not loaded)"
    elif is_evidence_placeholder:
        nli_judgment = "NOT_APPLICABLE"
        nli_reasoning = "(NLI check not applicable due to invalid evidence)"
    else: # Answer was error
        nli_judgment = "NOT_APPLICABLE"
        nli_reasoning = "(NLI check not applicable due to answer generation error)"

    print(f"NLI Judgment: {nli_judgment}")
    print(f"NLI Reasoning (trunc): {nli_reasoning[:200]}...")

    # 4. Calculate Confidence Score
    # ... (logic to calculate confidence_score based on nli_judgment - same as before) ...
    confidence_score = 0.0
    if nli_judgment == "ENTAILMENT": confidence_score = 0.9
    elif nli_judgment == "NEUTRAL": confidence_score = 0.5
    elif nli_judgment == "CONTRADICTION": confidence_score = 0.1
    # All others default to 0.0
    print(f"Calculated Confidence Score: {confidence_score}")


    # 5. Generate HTML Report
    report_filename_base = f"report_{job_id}.html"
    report_filepath = os.path.join(REPORTS_DIR, report_filename_base)
    generate_html_report(
        question,
        answer_text,
        evidence_spans_with_scores,
        document_text,
        nli_judgment=nli_judgment,
        nli_reasoning=nli_reasoning, # *** Pass reasoning ***
        confidence_score=confidence_score,
        output_filename=report_filepath
    )

    # Clean up stored data for this job ID
    # Important: Clear the large attention_weights tensor!
    if job_id in results_store:
        if 'attention_weights' in results_store[job_id]:
            del results_store[job_id]['attention_weights'] # Explicitly delete large item
        del results_store[job_id] # Remove entry
        gc.collect() # Encourage garbage collection


    end_req_time = time.time()
    print(f"--- /grade_response request completed in {end_req_time - start_req_time:.2f}s ---")

    return jsonify({
        "status": "grading_complete",
        "report_path": f"/report/{job_id}" # Relative path for browser link
    })

@app.route('/report/<job_id>')
def serve_report(job_id):
    """Serves the generated static HTML report."""
    filename = f"report_{job_id}.html"
    print(f"Serving report: {filename}")
    # Add security: Prevent directory traversal (though basic here)
    safe_filename = os.path.basename(filename)
    if safe_filename != filename:
         return "Invalid filename", 400
    return send_from_directory(REPORTS_DIR, safe_filename)

# --- Run the App ---
if __name__ == '__main__':
    # --- Load Model ONCE before starting server ---
    initialize_model()
    # --- Start the Flask server ---
    # Use port 5001 as planned, accessible on network
    # Set debug=False for stability, use terminal logs
    print("Starting Flask server on http://0.0.0.0:5001")
    app.run(debug=False, host='0.0.0.0', port=5001)