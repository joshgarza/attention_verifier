# src/attention_processor.py
import torch
import config
# Use dummy data functions if needed directly for testing
from src.utils import create_dummy_attention, create_dummy_tokenizer_output
import nltk
from collections import defaultdict
import numpy as np # For checking scores

# Ensure punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' resource not found.")
    print("Please run 'import nltk; nltk.download(\"punkt\")' manually in a Python interpreter.")
    # Optionally raise an error or exit if essential
    # raise RuntimeError("NLTK 'punkt' resource not found. Please download it.")


def aggregate_attention_scores(attention_weights, prompt_len, answer_len, aggregation_layer=-1, aggregate_heads='mean'):
    """
    Aggregates attention scores focusing on attention from answer tokens to prompt tokens.
    """
    print(f"[AttnProc] Aggregating scores: prompt_len={prompt_len}, answer_len={answer_len}, layer={aggregation_layer}, heads={aggregate_heads}")
    if not attention_weights:
        print("[AttnProc] Warning: No attention weights provided to aggregate.")
        return torch.zeros(prompt_len) # Return zeros matching expected prompt length

    # Select the desired layer's attention tensor
    try:
        layer_attn_maybe_tuple = attention_weights[aggregation_layer]

        # Check if the retrieved element is a tuple containing the tensor
        if isinstance(layer_attn_maybe_tuple, tuple) and len(layer_attn_maybe_tuple) > 0 and isinstance(layer_attn_maybe_tuple[0], torch.Tensor):
            print("[AttnProc] Extracted tensor from inner tuple in attention weights.")
            layer_attn = layer_attn_maybe_tuple[0]
        elif isinstance(layer_attn_maybe_tuple, torch.Tensor):
            # It was already a tensor
            layer_attn = layer_attn_maybe_tuple
        else:
            # Unexpected structure
            print(f"[AttnProc] Error: Unexpected type for layer attention: {type(layer_attn_maybe_tuple)}. Expected Tensor or Tuple[Tensor].")
            return torch.zeros(prompt_len, device='cpu') # Return zeros on error

    except IndexError:
        print(f"[AttnProc] Error: Aggregation layer index {aggregation_layer} out of bounds for {len(attention_weights)} layers.")
        return torch.zeros(prompt_len)

    batch_size, num_heads, query_seq_len, key_seq_len = layer_attn.shape
    print(f"[AttnProc] Layer {aggregation_layer} attention tensor shape: {layer_attn.shape}")

    # --- Sanity check dimensions ---
    expected_key_len = prompt_len + answer_len # Expected length of the context attended TO
    if key_seq_len < expected_key_len:
        print(f"[AttnProc] Warning: Attention key dim ({key_len}) < expected prompt+answer ({expected_key_len}). Using {key_len}.")
        effective_prompt_len = min(prompt_len, key_seq_len)
    elif key_seq_len > expected_key_len:
        print(f"[AttnProc] Warning: Attention key dim ({key_len}) > expected prompt+answer ({expected_key_len}). Using expected len for slicing.")
        effective_prompt_len = prompt_len
    else:
        effective_prompt_len = prompt_len

    prompt_key_indices = slice(0, effective_prompt_len)
    print(f"[AttnProc] Effective prompt key indices: {prompt_key_indices}")

    # --- NEW LOGIC ---
    # If query dim is 1, assume it's the last token's attention we received
    if query_seq_len == 1:
        print("[AttnProc] Query sequence length is 1. Using attention from last token only.")
        # Select attention FROM Last Token (Query=0) TO Prompt (Key)
        # Shape: (batch_size, num_heads, 1, effective_prompt_len)
        attn_last_token_to_prompt = layer_attn[:, :, 0, prompt_key_indices] # Query index is 0

        # We don't sum across answer tokens anymore, just use this directly
        # Shape: (batch_size, num_heads, effective_prompt_len)
        attn_received_by_prompt = attn_last_token_to_prompt.squeeze(2) # Remove the query dim of size 1
        print(f"[AttnProc] Attention Received by Prompt Tokens (from last token) shape: {attn_received_by_prompt.shape}")

    # --- ORIGINAL LOGIC (Fallback, might still fail if query_seq_len > 1 but < answer_len) ---
    else:
        print(f"[AttnProc] Query sequence length is {query_seq_len}. Attempting original aggregation.")
        # Indices corresponding to answer tokens in the *query* dimension
        # This logic might be flawed if query_seq_len doesn't actually match answer_len
        answer_query_start_idx = max(0, query_seq_len - answer_len)
        answer_query_indices = slice(answer_query_start_idx, query_seq_len)
        print(f"[AttnProc] Assuming answer query indices: {answer_query_indices}")

        # Select attention FROM Answer (Query) TO Prompt (Key)
        attn_answer_to_prompt = layer_attn[:, :, answer_query_indices, prompt_key_indices]
        print(f"[AttnProc] Attention Answer->Prompt shape: {attn_answer_to_prompt.shape}")

        # Aggregate across Answer Tokens (Query Dimension: dim 2)
        attn_received_by_prompt = attn_answer_to_prompt.sum(dim=2)
        print(f"[AttnProc] Attention Received by Prompt Tokens (summed) shape: {attn_received_by_prompt.shape}")
    # Aggregate across Heads (Dimension: dim 1)
    if aggregate_heads == 'mean':
        aggregated_scores_batch = attn_received_by_prompt.mean(dim=1)
    elif aggregate_heads == 'max':
        aggregated_scores_batch, _ = attn_received_by_prompt.max(dim=1)
    else:
        print(f"[AttnProc] Warning: Unknown head aggregation '{aggregate_heads}'. Defaulting to 'mean'.")
        aggregated_scores_batch = attn_received_by_prompt.mean(dim=1)
    print(f"[AttnProc] Aggregated Scores (batch) shape: {aggregated_scores_batch.shape}")


    # Squeeze batch dimension (assuming batch_size is 1)
    if aggregated_scores_batch.shape[0] == 1:
        final_scores_effective = aggregated_scores_batch.squeeze(0)
    else:
         # Handle potential future batching? For now, assume batch=1 or take first element.
         print(f"[AttnProc] Warning: Batch size is {aggregated_scores_batch.shape[0]}. Taking scores for the first batch item.")
         final_scores_effective = aggregated_scores_batch[0]

    print(f"[AttnProc] Final Effective Scores shape: {final_scores_effective.shape}")

    # --- Pad or truncate to match original expected prompt_len ---
    final_scores = torch.zeros(prompt_len, device=final_scores_effective.device)
    copy_len = min(prompt_len, len(final_scores_effective))
    final_scores[:copy_len] = final_scores_effective[:copy_len]

    # --- DEBUG: Check score values ---
    if len(final_scores) > 0:
         scores_np = final_scores.cpu().numpy()
         print(f"[AttnProc] Aggregated Scores (first 10): {scores_np[:10]}")
         print(f"[AttnProc] Score Stats: Min={np.min(scores_np):.4f}, Max={np.max(scores_np):.4f}, Mean={np.mean(scores_np):.4f}")
    else:
         print("[AttnProc] Aggregated scores tensor is empty.")

    return final_scores


def map_scores_to_sentences(aggregated_scores, prompt_text, token_offsets, document_token_indices):
    """
    Maps aggregated attention scores per token to the sentences containing those tokens.
    """
    print("[AttnProc] Mapping scores to sentences...")
    if aggregated_scores is None or len(aggregated_scores) == 0:
        print("[AttnProc] No aggregated scores provided for mapping.")
        return []
    if not document_token_indices:
         print("[AttnProc] Warning: document_token_indices list is empty. No scores can be mapped to document.")
         return []

    # --- Tokenize the prompt text into sentences ---
    try:
        # This is the line that failed - it should now find the 'punkt' resource
        sentences = nltk.sent_tokenize(prompt_text)
        print(f"[AttnProc] Found {len(sentences)} sentences in prompt.")
    except Exception as e:
        print(f"[AttnProc] Error tokenizing sentences: {e}. Returning empty list.")
        return []
    if not sentences:
        print("[AttnProc] NLTK returned no sentences.")
        return []

    # --- Create sentence boundaries (character offsets) ---
    sentence_boundaries = []
    current_pos = 0
    for sentence in sentences:
        try:
            start = prompt_text.index(sentence, current_pos) # Use index for robustness
            end = start + len(sentence)
            sentence_boundaries.append({"text": sentence, "start": start, "end": end})
            current_pos = end
        except ValueError:
            print(f"[AttnProc] Warning: Could not find sentence offset for: '''{sentence[:50]}...''' starting from {current_pos}")
            # Attempt basic recovery: search from beginning (might find wrong instance)
            start = prompt_text.find(sentence)
            if start != -1:
                end = start + len(sentence)
                sentence_boundaries.append({"text": sentence, "start": start, "end": end})
                # Don't update current_pos reliably here
            else:
                 print(f"[AttnProc] Error: Could not find sentence offset even from beginning.")


    print(f"[AttnProc] Created {len(sentence_boundaries)} sentence boundaries.")

    # --- Aggregate scores per sentence ---
    sentence_scores = defaultdict(float)
    num_scores = len(aggregated_scores)
    num_offsets = len(token_offsets)
    print(f"[AttnProc] Number of scores: {num_scores}, Number of offsets: {num_offsets}")
    print(f"[AttnProc] Document Token Indices to process: {document_token_indices[:10]}... (Total: {len(document_token_indices)})")


    mapped_token_count = 0
    # Iterate through the tokens identified as belonging to the document
    for token_idx in document_token_indices:
        # Check if the token index is valid for both scores and offsets
        if 0 <= token_idx < num_scores and 0 <= token_idx < num_offsets:
            score = aggregated_scores[token_idx].item()
            offsets = token_offsets[token_idx]

            # Check for valid offsets from the dummy generator
            if offsets and offsets[0] != -1 and offsets[1] != -1 and offsets[0] != -2:
                token_start, token_end = offsets
                found_sentence = False
                # Find which sentence this token falls into
                for boundary in sentence_boundaries:
                    # Check if token midpoint falls within sentence boundary
                    token_midpoint = token_start + (token_end - token_start) / 2
                    if boundary["start"] <= token_midpoint < boundary["end"]:
                        sentence_key = boundary["text"] # Using text as key
                        sentence_scores[sentence_key] += score
                        mapped_token_count += 1
                        found_sentence = True
                        # --- DEBUG PRINT ---
                        # print(f"  - Token idx {token_idx} (Score {score:.3f}, Chars {token_start}-{token_end}) mapped to sentence: '{sentence_key[:30]}...'")
                        break # Move to the next token once sentence is found
                # if not found_sentence:
                #     print(f"[AttnProc] Debug: Token idx {token_idx} (Chars {token_start}-{token_end}) did not map to any sentence boundary.")

            # else: # Optional: Warning about missing/invalid offsets
            #     print(f"[AttnProc] Debug: Skipping token {token_idx} due to invalid offsets {offsets}")
        else:
             print(f"[AttnProc] Warning: Document token index {token_idx} out of bounds for scores ({num_scores}) or offsets ({num_offsets}).")

    print(f"[AttnProc] Mapped {mapped_token_count} document tokens to sentences.")
    print(f"[AttnProc] Found {len(sentence_scores)} sentences with aggregated scores.")
    # --- DEBUG: Print sentences and scores before sorting ---
    # print("[AttnProc] Sentence Scores (Unsorted):")
    # for txt, score in sentence_scores.items():
    #      print(f"  - Score: {score:.4f}, Sentence: '{txt[:50]}...'")


    # --- Convert aggregated scores back to sorted list ---
    scored_sentences = [
        (total_score, sentence_text)
        for sentence_text, total_score in sentence_scores.items()
        # Filter out sentences with zero or negligible score if desired
        if total_score > 1e-6 # Add a small threshold to avoid pure noise
    ]

    # Sort by aggregated score descending
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    print(f"[AttnProc] Found {len(scored_sentences)} sentences with score > 1e-6 after sorting.")
    # print("[AttnProc] Scored Sentences (Sorted, Top 5):")
    # for score, txt in scored_sentences[:5]:
    #      print(f"  - Score: {score:.4f}, Sentence: '{txt[:60]}...'")


    return scored_sentences


def find_evidence_spans(attention_weights, tokenizer_output, prompt_text, top_k=5):
    """
    Main function to analyze attention and find top evidence spans (sentences).
    """
    print("[AttnProc] Finding evidence spans (sentences)...") # Updated print statement

    # --- Input Validation ---
    if not attention_weights:
        print("[AttnProc] Error: Missing attention_weights.")
        return ["(Missing attention weights)"]
    if not tokenizer_output:
        print("[AttnProc] Error: Missing tokenizer_output.")
        return ["(Missing tokenizer output)"]
    if not prompt_text:
         print("[AttnProc] Error: Missing prompt_text.")
         return ["(Missing prompt text)"]

    prompt_len = tokenizer_output.get('prompt_len')
    answer_len = tokenizer_output.get('answer_len')
    all_token_offsets = tokenizer_output.get('all_token_offsets')
    doc_indices = tokenizer_output.get('document_token_indices')
    doc_start_char = tokenizer_output.get('doc_content_start_char') # Get from util output
    doc_end_char = tokenizer_output.get('doc_content_end_char') # Get from util output

    # Check if necessary keys exist in tokenizer_output
    if None in [prompt_len, answer_len, all_token_offsets, doc_indices, doc_start_char, doc_end_char]:
        missing_keys = [k for k, v in tokenizer_output.items() if v is None]
        print(f"[AttnProc] Warning: tokenizer_output dict is missing required keys: {missing_keys}. Cannot reliably process spans.")
        # Attempt to continue but results might be compromised
        # return ["(Tokenizer info missing required keys)"] # Option to exit early

    print(f"[AttnProc] Using Prompt Len: {prompt_len}, Answer Len: {answer_len}")
    print(f"[AttnProc] Using Doc Char Boundaries: Start={doc_start_char}, End={doc_end_char}")


    # 1. Aggregate scores per prompt token
    aggregated_scores = aggregate_attention_scores(
        attention_weights,
        prompt_len if prompt_len is not None else 0, # Handle None case
        answer_len if answer_len is not None else 0, # Handle None case
    )

    # Check if scores are valid before proceeding
    if aggregated_scores is None or len(aggregated_scores) == 0:
         print("[AttnProc] Aggregated scores are empty. Cannot map to sentences.")
         return ["(Failed to aggregate attention scores)"]


    # 2. Map scores to sentences
    # Ensure required inputs for mapping are valid
    if all_token_offsets is None or doc_indices is None:
         print("[AttnProc] Cannot map scores to sentences due to missing offsets or doc indices.")
         return ["(Missing offsets/indices for sentence mapping)"]

    scored_sentences = map_scores_to_sentences(
        aggregated_scores,
        prompt_text,
        all_token_offsets,
        doc_indices
    )

    # 3. Extract top K sentences that fall within document boundaries
    top_sentences = []
    print(f"[AttnProc] Filtering {len(scored_sentences)} scored sentences using doc boundaries [{doc_start_char}, {doc_end_char}).")
    if doc_start_char is None or doc_end_char is None:
         print("[AttnProc] Warning: Document boundaries are None, cannot filter sentences by location.")
         # Fallback: Just take top K without location filtering
         top_sentences = [s[1].strip() for s in scored_sentences[:top_k]]
    else:
        # Filter by location
        for score, sentence_text in scored_sentences:
            # Find the sentence's start position to check if it's within the doc boundaries
            try:
                 # Use find which is simpler than index for potentially duplicate sentences
                 sentence_start_char = prompt_text.find(sentence_text)
            except Exception as e:
                 print(f"[AttnProc] Error finding start char for sentence: {e}")
                 sentence_start_char = -1 # Mark as not found

            # Check if found and within valid document character range
            if sentence_start_char != -1 and \
               sentence_start_char >= doc_start_char and \
               sentence_start_char < doc_end_char: # Use '<' for end boundary typically

                 if len(top_sentences) < top_k:
                      # --- DEBUG PRINT ---
                      print(f"  - Keeping sentence (Score {score:.3f}, Start Char {sentence_start_char}): '{sentence_text[:60].strip()}...'")
                      top_sentences.append(sentence_text.strip()) # Add cleaned sentence
            # else:
                 # --- DEBUG PRINT ---
                 # print(f"  - Discarding sentence (Score {score:.3f}, Start Char {sentence_start_char}): '{sentence_text[:60].strip()}...'")

            # Stop early if we have enough sentences
            if len(top_sentences) >= top_k:
                 break


    print(f"[AttnProc] Identified {len(top_sentences)} potential evidence sentences after filtering.")
    return top_sentences


# --- Example Usage with Dummy Data ---
if __name__ == "__main__":
    print("\n" + "="*30 + "\nRunning src/attention_processor.py Standalone Test\n" + "="*30)
    # Dummy parameters
    num_layers = 4 # Example
    num_heads = 12 # Example

    # --- Use the Test Setup from utils.py ---
    print("\n--- Setting up Dummy Data using src/utils.py ---")
    dummy_doc_content = "This is the first sentence. Here is the second sentence which contains important information. The third sentence concludes the main point."
    dummy_question_content = "What information is in the second sentence?"
    dummy_answer_content = "The second sentence has important info."
    # Construct prompt with EXACT markers
    dummy_prompt_text = f"Document:\n{dummy_doc_content}\n\nQuestion: {dummy_question_content}\n\nAnswer:"
    print(f"Test Prompt:\n'''{dummy_prompt_text}'''")

    # Generate dummy tokenizer output (includes offsets and boundaries)
    dummy_tokenizer_info = create_dummy_tokenizer_output(dummy_prompt_text, dummy_answer_content)
    print("--- Finished Setting up Dummy Data ---\n")


    # Check if tokenizer info generation was successful
    if not dummy_tokenizer_info or not all(k in dummy_tokenizer_info for k in ['prompt_len', 'answer_len', 'all_token_offsets', 'document_token_indices']):
         print("Error: Failed to generate necessary dummy tokenizer info. Exiting test.")
    else:
        prompt_len = dummy_tokenizer_info['prompt_len']
        answer_len = dummy_tokenizer_info['answer_len']

        # --- Add Signal to Dummy Attention ---
        target_phrase = "second sentence"
        target_indices = []
        start_char_target = dummy_prompt_text.find(target_phrase)
        if start_char_target != -1:
            end_char_target = start_char_target + len(target_phrase)
            for i, offset in enumerate(dummy_tokenizer_info['all_token_offsets'][:prompt_len]):
                if offset != (-1,-1) and offset != (-2,-2):
                    if max(offset[0], start_char_target) < min(offset[1], end_char_target):
                        target_indices.append(i)
        print(f"[Test Setup] Indices for '{target_phrase}': {target_indices}")

        # Generate dummy attention weights WITH signal
        dummy_weights = create_dummy_attention(
             num_layers, num_heads, prompt_len, answer_len,
             make_signal=True,
             signal_indices=target_indices, # Target the phrase "second sentence"
             signal_strength=20.0 # Make signal stronger
        )

        # --- Run the evidence finding function ---
        print("\n--- Running find_evidence_spans ---")
        evidence = find_evidence_spans(
             dummy_weights,
             dummy_tokenizer_info,
             dummy_prompt_text,
             top_k=config.TOP_K_EVIDENCE if 'config' in globals() else 3 # Use config or default
        )
        print("--- Finished find_evidence_spans ---")


        print("\n--- Dummy Evidence Sentences Found ---")
        if evidence:
            for i, span in enumerate(evidence):
                print(f"Evidence {i+1}: {span}")
        else:
            print("No evidence sentences identified.")
        print("--- End Dummy Example ---")