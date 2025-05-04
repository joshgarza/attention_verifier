# src/attention_processor.py

# --- Keep imports: torch, config, nltk, defaultdict, np ---
import torch
import config
import nltk
from collections import defaultdict
import numpy as np
# --- Keep the previously corrected aggregate_attention_scores ---
# --- Keep the previously corrected map_scores_to_sentences ---

def aggregate_attention_scores(attention_weights, prompt_len, answer_len, aggregation_layer=-1, aggregate_heads='mean'):
    """
    Aggregates attention scores focusing on attention from answer tokens to prompt tokens.
    Handles potential last-token-only attention format.
    """
    print(f"[AttnProc] Aggregating scores: prompt_len={prompt_len}, answer_len={answer_len}, layer={aggregation_layer}, heads={aggregate_heads}")
    if not attention_weights:
        print("[AttnProc] Warning: No attention weights provided to aggregate.")
        return torch.zeros(prompt_len, device='cpu')

    try:
        layer_attn_maybe_tuple = attention_weights[aggregation_layer]
        if isinstance(layer_attn_maybe_tuple, tuple) and len(layer_attn_maybe_tuple) > 0 and isinstance(layer_attn_maybe_tuple[0], torch.Tensor):
            print("[AttnProc] Extracted tensor from inner tuple in attention weights.")
            layer_attn = layer_attn_maybe_tuple[0]
        elif isinstance(layer_attn_maybe_tuple, torch.Tensor):
            layer_attn = layer_attn_maybe_tuple
        else:
            print(f"[AttnProc] Error: Unexpected type for layer attention: {type(layer_attn_maybe_tuple)}. Expected Tensor or Tuple[Tensor].")
            return torch.zeros(prompt_len, device='cpu')
    except IndexError:
        print(f"[AttnProc] Error: Aggregation layer index {aggregation_layer} out of bounds for {len(attention_weights)} layers.")
        return torch.zeros(prompt_len, device='cpu')
    except Exception as e:
        print(f"[AttnProc] Error accessing attention layer {aggregation_layer}: {e}")
        return torch.zeros(prompt_len, device='cpu')

    if not isinstance(layer_attn, torch.Tensor):
        print(f"[AttnProc] Error: layer_attn is not a Tensor after processing ({type(layer_attn)}). Cannot get shape.")
        return torch.zeros(prompt_len, device='cpu')

    batch_size, num_heads, query_seq_len, key_seq_len = layer_attn.shape
    print(f"[AttnProc] Layer {aggregation_layer} attention tensor shape: {layer_attn.shape}")

    # Adjust effective prompt length based on attention tensor's key dimension
    expected_key_len = prompt_len + answer_len
    if key_seq_len < expected_key_len:
         print(f"[AttnProc] Warning: Attention key dim ({key_seq_len}) < expected prompt+answer ({expected_key_len}). Using {key_seq_len}.")
         effective_prompt_len = min(prompt_len, key_seq_len)
    elif key_seq_len > expected_key_len:
         print(f"[AttnProc] Warning: Attention key dim ({key_seq_len}) > expected prompt+answer ({expected_key_len}). Using expected len for slicing.")
         effective_prompt_len = prompt_len
    else:
         effective_prompt_len = prompt_len

    prompt_key_indices = slice(0, effective_prompt_len)
    print(f"[AttnProc] Effective prompt key indices: {prompt_key_indices}")

    # Adapt logic based on query sequence length
    if query_seq_len == 1:
        print("[AttnProc] Query sequence length is 1. Using attention from last token only.")
        attn_last_token_to_prompt = layer_attn[:, :, 0, prompt_key_indices]
        attn_received_by_prompt = attn_last_token_to_prompt.squeeze(2)
        print(f"[AttnProc] Attention Received by Prompt Tokens (from last token) shape: {attn_received_by_prompt.shape}")
    else:
        print(f"[AttnProc] Query sequence length is {query_seq_len}. Attempting original aggregation (summing over query dim).")
        # This assumes query dim reflects answer tokens, may need adjustment if not true
        answer_query_start_idx = max(0, query_seq_len - answer_len) # Use original answer_len here
        answer_query_indices = slice(answer_query_start_idx, query_seq_len)
        print(f"[AttnProc] Assuming answer query indices: {answer_query_indices}")
        attn_answer_to_prompt = layer_attn[:, :, answer_query_indices, prompt_key_indices]
        print(f"[AttnProc] Attention Answer->Prompt shape: {attn_answer_to_prompt.shape}")
        if attn_answer_to_prompt.shape[2] == 0: # Check if answer slice resulted in zero length
             print("[AttnProc] Error: Calculated answer query slice has zero length. Cannot sum.")
             return torch.zeros(prompt_len, device='cpu')
        attn_received_by_prompt = attn_answer_to_prompt.sum(dim=2)
        print(f"[AttnProc] Attention Received by Prompt Tokens (summed) shape: {attn_received_by_prompt.shape}")

    # Aggregate across Heads
    if aggregate_heads == 'mean':
        aggregated_scores_batch = attn_received_by_prompt.mean(dim=1)
    elif aggregate_heads == 'max':
        aggregated_scores_batch, _ = attn_received_by_prompt.max(dim=1)
    else:
        aggregated_scores_batch = attn_received_by_prompt.mean(dim=1)
    print(f"[AttnProc] Aggregated Scores (batch) shape: {aggregated_scores_batch.shape}")

    # Squeeze batch dimension
    if aggregated_scores_batch.shape[0] == 1:
        final_scores_effective = aggregated_scores_batch.squeeze(0)
    else:
         print(f"[AttnProc] Warning: Batch size is {aggregated_scores_batch.shape[0]}. Taking scores for the first batch item.")
         final_scores_effective = aggregated_scores_batch[0]
    print(f"[AttnProc] Final Effective Scores shape: {final_scores_effective.shape}")

    # Pad or truncate to match original expected prompt_len
    final_scores = torch.zeros(prompt_len, device=final_scores_effective.device)
    copy_len = min(prompt_len, len(final_scores_effective))
    final_scores[:copy_len] = final_scores_effective[:copy_len]

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
    Returns list of (score, sentence_text) tuples, sorted descending by score.
    """
    print("[AttnProc] Mapping scores to sentences...")
    if aggregated_scores is None or len(aggregated_scores) == 0:
        print("[AttnProc] No aggregated scores provided for mapping.")
        return []
    if not document_token_indices:
         print("[AttnProc] Warning: document_token_indices list is empty. No scores can be mapped to document.")
         return []

    try:
        sentences = nltk.sent_tokenize(prompt_text)
        print(f"[AttnProc] Found {len(sentences)} sentences in prompt.")
    except Exception as e:
        print(f"[AttnProc] Error tokenizing sentences: {e}. Returning empty list.")
        return []
    if not sentences:
        print("[AttnProc] NLTK returned no sentences.")
        return []

    sentence_boundaries = []
    current_pos = 0
    for sentence in sentences:
        try:
            start = prompt_text.index(sentence, current_pos)
            end = start + len(sentence)
            sentence_boundaries.append({"text": sentence, "start": start, "end": end})
            current_pos = end
        except ValueError:
            print(f"[AttnProc] Warning: Could not find sentence offset for: '''{sentence[:50]}...''' starting from {current_pos}")
            start = prompt_text.find(sentence) # Fallback search
            if start != -1:
                 end = start + len(sentence)
                 sentence_boundaries.append({"text": sentence, "start": start, "end": end})
            else:
                 print(f"[AttnProc] Error: Could not find sentence offset even from beginning.")

    print(f"[AttnProc] Created {len(sentence_boundaries)} sentence boundaries.")

    sentence_scores = defaultdict(float)
    num_scores = len(aggregated_scores)
    num_offsets = len(token_offsets)
    print(f"[AttnProc] Number of scores: {num_scores}, Number of offsets: {num_offsets}")
    print(f"[AttnProc] Document Token Indices to process: {document_token_indices[:10]}... (Total: {len(document_token_indices)})")

    mapped_token_count = 0
    for token_idx in document_token_indices:
        if 0 <= token_idx < num_scores and 0 <= token_idx < num_offsets:
            score = aggregated_scores[token_idx].item()
            offset = token_offsets[token_idx]

            if isinstance(offset, (list, tuple)) and len(offset) == 2 and \
               offset[0] is not None and offset[1] is not None and \
               isinstance(offset[0], int) and isinstance(offset[1], int) and \
               offset != (0, 0) and offset != [0, 0] and \
               offset[0] < offset[1]:

                token_start, token_end = offset
                for boundary in sentence_boundaries:
                    token_midpoint = token_start + (token_end - token_start) / 2
                    if boundary["start"] <= token_midpoint < boundary["end"]:
                        sentence_key = boundary["text"]
                        sentence_scores[sentence_key] += score
                        mapped_token_count += 1
                        break
            # else: # Debug invalid offsets if needed
        else:
             print(f"[AttnProc] Warning: Document token index {token_idx} out of bounds for scores ({num_scores}) or offsets ({num_offsets}).")

    print(f"[AttnProc] Mapped {mapped_token_count} document tokens to sentences.")
    print(f"[AttnProc] Found {len(sentence_scores)} sentences with aggregated scores.")

    scored_sentences = [
        (total_score, sentence_text)
        for sentence_text, total_score in sentence_scores.items()
        if total_score > 1e-6 # Optional threshold
    ]

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    print(f"[AttnProc] Found {len(scored_sentences)} sentences with score > 1e-6 after sorting.")

    return scored_sentences # Returns list of (score, sentence_text) tuples


# --- THIS IS THE FUNCTION TO FOCUS ON ---
def find_evidence_spans(attention_weights, tokenizer_output, prompt_text, top_k=5):
    """
    Main function to analyze attention and find top evidence spans (sentences).
    Returns a list of (score, sentence_text) tuples, sorted by score.
    """
    print("[AttnProc] Finding evidence spans (sentences)...")

    # --- Input Validation ---
    if not attention_weights:
        print("[AttnProc] Error: Missing attention_weights.")
        return [(0.0, "(Missing attention weights)")] # Return list with tuple
    if not tokenizer_output:
        print("[AttnProc] Error: Missing tokenizer_output.")
        return [(0.0, "(Missing tokenizer output)")] # Return list with tuple
    if not prompt_text:
         print("[AttnProc] Error: Missing prompt_text.")
         return [(0.0, "(Missing prompt text)")] # Return list with tuple

    # --- Extract necessary info from tokenizer_output ---
    prompt_len = tokenizer_output.get('prompt_len')
    answer_len = tokenizer_output.get('answer_len')
    all_token_offsets = tokenizer_output.get('all_token_offsets')
    doc_indices = tokenizer_output.get('document_token_indices')
    doc_start_char = tokenizer_output.get('doc_content_start_char')
    doc_end_char = tokenizer_output.get('doc_content_end_char')

    # Validate required keys
    required_keys = ['prompt_len', 'answer_len', 'all_token_offsets', 'document_token_indices', 'doc_content_start_char', 'doc_content_end_char']
    if None in [prompt_len, answer_len, all_token_offsets, doc_indices, doc_start_char, doc_end_char]:
        missing = {k: tokenizer_output.get(k) for k in required_keys if tokenizer_output.get(k) is None}
        print(f"[AttnProc] Warning: tokenizer_output dict is missing required keys or has None values: {missing}.")
        return [(0.0, "(Tokenizer info missing required keys)")] # Return list with tuple

    print(f"[AttnProc] Using Prompt Len: {prompt_len}, Answer Len: {answer_len}")
    print(f"[AttnProc] Using Doc Char Boundaries: Start={doc_start_char}, End={doc_end_char}")

    # 1. Aggregate scores per prompt token
    aggregated_scores = aggregate_attention_scores(
        attention_weights,
        prompt_len,
        answer_len,
    )

    if aggregated_scores is None or len(aggregated_scores) == 0:
         print("[AttnProc] Aggregated scores are empty. Cannot map to sentences.")
         return [(0.0, "(Failed to aggregate attention scores)")]

    # 2. Map scores to sentences -> returns list of (score, sentence) tuples
    scored_sentences = map_scores_to_sentences(
        aggregated_scores,
        prompt_text,
        all_token_offsets,
        doc_indices
    )

    if not scored_sentences:
         print("[AttnProc] No sentences were scored after mapping.")
         # Return empty list instead of placeholder if mapping worked but found nothing
         return []

    # 3. Filter top K sentences that fall within document boundaries
    # --- THIS IS WHERE THE RETURNED VALUE IS BUILT ---
    top_evidence_tuples = [] # Initialize list for (score, text) tuples
    print(f"[AttnProc] Filtering {len(scored_sentences)} scored sentences using doc boundaries [{doc_start_char}, {doc_end_char}).")

    for score, sentence_text in scored_sentences:
        try:
             sentence_start_char = prompt_text.find(sentence_text) # Find sentence start
        except Exception as e:
             print(f"[AttnProc] Error finding start char for sentence: {e}")
             sentence_start_char = -1

        # Check if found and within valid document character range
        if sentence_start_char != -1 and \
           sentence_start_char >= doc_start_char and \
           sentence_start_char < doc_end_char: # Use '<' for end boundary

             if len(top_evidence_tuples) < top_k:
                  print(f"  - Keeping sentence (Score {score:.3f}, Start Char {sentence_start_char}): '{sentence_text[:60].strip()}...'")
                  # --- APPEND THE TUPLE ---
                  top_evidence_tuples.append((score, sentence_text.strip()))
        # else: # Debugging discarded sentences if needed
             # print(f"  - Discarding sentence (Score {score:.3f}, Start Char {sentence_start_char}): '{sentence_text[:60].strip()}...'")

        # Stop early if we have enough sentences
        if len(top_evidence_tuples) >= top_k:
             break

    print(f"[AttnProc] Identified {len(top_evidence_tuples)} potential evidence sentences after filtering.")
    # --- RETURN THE LIST OF TUPLES ---
    return top_evidence_tuples

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