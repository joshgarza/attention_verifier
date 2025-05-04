import argparse
import time
import sys
import json # Needed for potential complex data handling later, good practice
import torch # Needed for tensor checks and device handling in tokenizer prep

# Local project imports
import config
from src.data_loader import load_document
from src.model_handler import load_model_and_tokenizer, get_answer_and_attention
from src.attention_processor import find_evidence_spans
# Import dummy data functions only if needed for testing/dummy run
from src.utils import create_dummy_attention, create_dummy_tokenizer_output

def run_verification(doc_path, question, dummy_run=False):
    """Runs the full attention-guided verification pipeline."""
    print(f"--- Starting Attention-Guided Verification {'(DUMMY RUN)' if dummy_run else ''} ---")
    start_time = time.time()

    # --- Initialize variables ---
    document_text = None
    model = None
    tokenizer = None
    answer_text = "(No answer generated)"
    attention_weights = None
    tokenizer_output_dict = None
    prompt_text = "(No prompt constructed)" # Initialize prompt_text

    # --- Conditional Execution based on dummy_run ---
    if dummy_run:
        # --- Dummy Run Logic ---
        print("\n[*] --- Generating Dummy Data ---")
        # Define dummy content directly
        dummy_doc = "This is the first sentence of the document. Here is the second sentence which contains important information. The third sentence concludes the main point."
        dummy_question = "What information is in the second sentence?"
        document_text = dummy_doc # Use dummy doc as document_text
        question = dummy_question # Override question

        # Construct the prompt like the real run would
        prompt_text = f"""Document:
{document_text}

Question: {question}

Answer:""" # The model would generate text after this

        # Generate dummy answer
        answer_text = "The second sentence has important info."

        # Generate dummy tokenizer info based on prompt and dummy answer
        print("[*] Generating dummy tokenizer info...")
        tokenizer_output_dict = create_dummy_tokenizer_output(prompt_text, answer_text)
        if not tokenizer_output_dict:
             print("Error: Failed to create dummy tokenizer output. Exiting.")
             return

        # Generate dummy attention weights
        print("[*] Generating dummy attention weights...")
        # Example config, adjust if needed
        num_layers = 18 # Match Llama 3 8B layer count for realism if possible
        num_heads = 32 # Example head count
        prompt_len = tokenizer_output_dict['prompt_len']
        answer_len = tokenizer_output_dict['answer_len']
        attention_weights = create_dummy_attention(num_layers, num_heads, prompt_len, answer_len)
        print("[*] --- Dummy Data Generation Complete ---")

    else:
        # --- Real Run Logic ---
        # 1. Load Document
        print(f"\n[1/4] Loading document: {doc_path}")
        document_text = load_document(doc_path)
        if not document_text:
            print("Failed to load document. Exiting.")
            return
        print(f"Document loaded ({len(document_text)} characters).")

        # Construct the prompt (needed for both inference and attention processing)
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
            return # Exit if model loading fails

        # 3. Get Answer and Attention from Model
        print("\n[3/4] Running model for answer and attention...")
        answer_text, attention_weights = get_answer_and_attention(
            model, tokenizer, document_text, question
        )
        print("Model inference complete.") # This prints even if generation had errors stored in answer_text

        # --- NEW: Prepare REAL Tokenizer Output Dict ---
        print("\n[~] Preparing tokenizer info for attention processing...")
        tokenizer_output_dict = None # Initialize
        # Only proceed if we have a real tokenizer and attention weights were likely generated
        if tokenizer is not None and attention_weights is not None:
            try:
                # 1. Tokenize the original prompt to get length and offset mapping for it
                # Use return_offsets_mapping=True and match generate() settings
                max_model_input_len = 4096 # Match the length used in generate
                prompt_encoding = tokenizer(
                    prompt_text,
                    return_offsets_mapping=True,
                    max_length=max_model_input_len,
                    truncation=True,
                    return_tensors="pt"
                )
                prompt_len_calc = prompt_encoding.input_ids.shape[-1]
                # Offset mapping is usually nested in a batch list, take first element
                # Handle potential None return if tokenizer doesn't support it for some reason
                prompt_offsets = prompt_encoding.get('offset_mapping')
                if prompt_offsets is not None:
                     prompt_offsets = prompt_offsets[0].tolist()
                else:
                     print("Warning: Tokenizer did not return offset_mapping. Cannot map accurately.")
                     prompt_offsets = [(None, None)] * prompt_len_calc # Fallback


                # 2. Estimate answer length from attention tensor dimensions (most reliable)
                answer_len_calc = 0
                # Check if attention_weights is a tuple/list and has elements
                if attention_weights and isinstance(attention_weights, (list, tuple)) and len(attention_weights) > 0:
                    # Check if the last layer's attention is a tensor
                    last_layer_attn = attention_weights[-1]
                    # Handle potential nested tuple structure for attention tensors
                    if isinstance(last_layer_attn, tuple) and len(last_layer_attn) > 0:
                         last_layer_attn = last_layer_attn[0]

                    if isinstance(last_layer_attn, torch.Tensor):
                         # Shape: (batch, head, query_len, key_len)
                         key_len = last_layer_attn.shape[-1]
                         if key_len >= prompt_len_calc:
                              answer_len_calc = key_len - prompt_len_calc
                         else:
                              print(f"Warning: Attention key dim ({key_len}) < calculated prompt len ({prompt_len_calc}). Setting answer len to 0.")
                    else:
                        print(f"Warning: Last attention layer element is not a Tensor ({type(last_layer_attn)}). Cannot get answer len from attention.")
                elif answer_text and not answer_text.startswith("Error"):
                     # Fallback: tokenize generated answer text (less reliable)
                     answer_tokens_est = tokenizer(answer_text, add_special_tokens=False).input_ids
                     answer_len_calc = len(answer_tokens_est)
                     print("Warning: Estimating answer length by tokenizing answer text.")
                else:
                     print("Warning: Could not determine answer length. Setting to 0.")


                # 3. Combine offsets (Prompt offsets + None padding for answer)
                # Ensure prompt_offsets is a list before padding
                if not isinstance(prompt_offsets, list):
                     print("Warning: prompt_offsets is not a list. Using placeholder.")
                     prompt_offsets = [(None, None)] * prompt_len_calc # Fallback
                all_offsets_calc = prompt_offsets + [(None, None)] * answer_len_calc

                # 4. Calculate document token indices and boundaries based on prompt text
                doc_content_start_char = -1
                doc_content_end_char = -1
                doc_start_marker = "Document:\n"
                doc_end_marker = "\n\nQuestion:"
                doc_content_start_idx = prompt_text.find(doc_start_marker)
                doc_content_end_idx = prompt_text.find(doc_end_marker)

                # Determine boundaries robustly
                if doc_content_start_idx != -1:
                    doc_content_start_char = doc_content_start_idx + len(doc_start_marker)
                if doc_content_end_idx != -1 and (doc_content_start_char == -1 or doc_content_start_char <= doc_content_end_idx):
                    doc_content_end_char = doc_content_end_idx
                else: # Fallback if markers not found or end is before start
                    doc_content_end_char = len(prompt_text)
                # If start marker wasn't found, assume start of text is doc start
                if doc_content_start_char == -1:
                     doc_content_start_char = 0
                # Ensure start is not >= end after fallbacks
                if doc_content_start_char >= doc_content_end_char:
                     doc_content_start_char = 0
                     doc_content_end_char = len(prompt_text)
                     print("Warning: Could not reliably determine document boundaries. Using full prompt.")


                document_token_indices_calc = []
                # Iterate only through prompt tokens using prompt_len_calc
                for i in range(prompt_len_calc):
                    # Check if index is valid for prompt_offsets
                    if i < len(prompt_offsets):
                        offset = prompt_offsets[i]
                        # Check offset is valid tuple and falls within the detected document character span
                        if isinstance(offset, tuple) and len(offset) == 2 and \
                           offset[0] is not None and offset[1] is not None and \
                           offset != (0, 0) and \
                           offset[0] >= doc_content_start_char and \
                           offset[1] <= doc_content_end_char and \
                           offset[0] < offset[1]: # Ensure start < end
                             document_token_indices_calc.append(i) # Append the index i

                # 5. Create the dictionary with all necessary keys
                tokenizer_output_dict = {
                    "prompt_len": prompt_len_calc,
                    "answer_len": answer_len_calc,
                    "all_token_offsets": all_offsets_calc, # Includes prompt + None padding
                    "document_token_indices": document_token_indices_calc, # Indices within prompt_len
                    "doc_content_start_char": doc_content_start_char,
                    "doc_content_end_char": doc_content_end_char
                }
                print("Tokenizer info prepared using real tokenizer data.")
                # Optional Debug Print:
                # print(f"  Prepared Tokenizer Info: {json.dumps({k: (v[:5] + ['...'] if isinstance(v, list) and len(v) > 5 else v) for k, v in tokenizer_output_dict.items()}, indent=2)}")

            except Exception as e:
                print(f"Warning: Could not prepare tokenizer info for attention processing: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                tokenizer_output_dict = None # Ensure it's None if prep fails
        else:
            print("Warning: Real tokenizer or attention weights not available, cannot prepare detailed info for attention.")
        # --- *** END OF NEW SECTION *** ---


    # --- Common Processing Step ---
    # 4. Process Attention to Find Evidence
    print(f"\n[{'*/4' if dummy_run else '4/4'}] Processing attention weights...")
    evidence_spans = []
    # Check if we have the necessary inputs for attention processing
    if attention_weights is not None and tokenizer_output_dict is not None:
         # Check required keys again before passing
         required_keys = ['prompt_len', 'answer_len', 'all_token_offsets', 'document_token_indices', 'doc_content_start_char', 'doc_content_end_char']
         # Check that keys exist and their values are not None
         if all(k in tokenizer_output_dict and tokenizer_output_dict[k] is not None for k in required_keys):
              evidence_spans = find_evidence_spans(
                  attention_weights,
                  tokenizer_output_dict, # Pass the prepared dict
                  prompt_text,           # Pass the full prompt text used
                  top_k=config.TOP_K_EVIDENCE
              )
         else:
              missing_or_none = {k: tokenizer_output_dict.get(k) for k in required_keys}
              print(f"Warning: Cannot process attention, missing or None values in tokenizer_output_dict: {missing_or_none}")
              evidence_spans = ["(Cannot process attention due to missing tokenizer info)"]

    elif attention_weights is None:
        evidence_spans = ["(Attention weights not available or generation failed)"]
    else: # Tokenizer info missing or failed prep
        evidence_spans = ["(Could not prepare tokenizer info needed for attention processing)"]
    print("Attention processing step complete.")


    # --- Output Results ---
    print("\n--- Verification Results ---")
    print(f"Question: {question}") # Use the actual question used (dummy or real)
    print(f"\nModel Answer:\n{answer_text}")

    print("\nPotential Evidence Spans (Based on Attention Analysis):")
    # Ensure evidence_spans is a list before iterating
    if isinstance(evidence_spans, list) and evidence_spans:
        for i, span in enumerate(evidence_spans):
            print(f"Evidence {i+1}:\n---\n{span}\n---")
    elif not evidence_spans: # Handles empty list case
         print("No evidence spans identified.")
    else: # Handle non-list return values if something went very wrong
         print(f"Could not identify evidence spans (unexpected format: {type(evidence_spans)})")


    end_time = time.time()
    print(f"\n--- Process completed in {end_time - start_time:.2f} seconds ---")


# --- Argparse and Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Attention-Guided Response Verification on a document."
    )
    parser.add_argument(
        '--dummy-run',
        action='store_true',
        help="Run using dummy data for testing attention processing without loading the model."
    )
    parser.add_argument(
        "doc_path",
        nargs='?', # Make optional
        default=None,
        help="Path to the input document file (required if not --dummy-run)"
    )
    parser.add_argument(
        "question",
        nargs='?', # Make optional
        default=None,
        help="The question to ask about the document (required if not --dummy-run)"
    )
    args = parser.parse_args()

    # Check if required args are provided for a real run
    if not args.dummy_run and (args.doc_path is None or args.question is None):
        parser.error("doc_path and question are required when not using --dummy-run")
        sys.exit(1) # Exit if arguments are missing for a real run

    # Call the main function
    run_verification(args.doc_path, args.question, args.dummy_run)