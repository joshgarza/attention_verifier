# src/utils.py
import torch
import random

# --- Keep create_dummy_attention function as before ---
def create_dummy_attention(num_layers, num_heads, prompt_len, answer_len, device='cpu', make_signal=True, signal_indices=None, signal_strength=10.0):
    """Creates a dummy attention tuple matching transformers output structure.
       Can optionally add a stronger signal to specific indices."""
    total_seq_len = prompt_len + answer_len
    dummy_attentions = []
    for layer_idx in range(num_layers):
        layer_attention = torch.rand((1, num_heads, total_seq_len, total_seq_len), device=device)
        if make_signal and signal_indices:
             answer_start_idx = prompt_len
             valid_signal_indices = [idx for idx in signal_indices if 0 <= idx < prompt_len]
             if valid_signal_indices:
                layer_attention[0, :, answer_start_idx:, valid_signal_indices] += signal_strength * torch.rand_like(layer_attention[0, :, answer_start_idx:, valid_signal_indices])
        dummy_attentions.append(layer_attention)
    print(f"[Dummy Util] Created dummy attention: {num_layers} layers, {num_heads} heads, seq_len {total_seq_len}")
    return tuple(dummy_attentions)

# --- Keep create_dummy_tokenizer_output function as before ---
# (Including the refined boundary detection and index calculation logic)
def create_dummy_tokenizer_output(prompt, answer, tokenizer_vocab_size=32000):
    """Creates dummy token IDs and simulates offset mapping."""
    print("[Dummy Util] Creating dummy tokenizer output...")
    # --- Refined Document Boundary Detection ---
    doc_start_marker = "Document:\n"
    doc_end_marker = "\n\nQuestion:"
    doc_content_start_char = -1
    doc_content_end_char = -1
    doc_content_start_idx = prompt.find(doc_start_marker)
    doc_content_end_idx = prompt.find(doc_end_marker)
    if doc_content_start_idx != -1:
        doc_content_start_char = doc_content_start_idx + len(doc_start_marker)
    if doc_content_end_idx != -1 and (doc_content_start_char == -1 or doc_content_start_char <= doc_content_end_idx):
        doc_content_end_char = doc_content_end_idx
    else:
        doc_content_end_char = len(prompt)
    if doc_content_start_char == -1:
         doc_content_start_char = 0
    if doc_content_start_char >= doc_content_end_char:
         doc_content_start_char = 0
         doc_content_end_char = len(prompt)
         print("[Dummy Util] Warning: Could not reliably determine document boundaries. Using full prompt.")
    print(f"[Dummy Util] Document boundaries (chars): Start={doc_content_start_char}, End={doc_content_end_char}")

    # Simple space-split tokenization
    words = prompt.split()
    answer_words = answer.split()
    prompt_len = len(words)
    answer_len = len(answer_words)
    total_len = prompt_len + answer_len

    all_token_offsets = []
    document_token_indices = []
    current_pos = 0
    for i, word in enumerate(words):
        try:
            stripped_word = word.strip()
            if not stripped_word:
                 all_token_offsets.append((-1,-1))
                 continue
            start = prompt.index(stripped_word, current_pos)
            end = start + len(stripped_word)
            offset_tuple = (start, end)
            all_token_offsets.append(offset_tuple)
            if start >= doc_content_start_char and end <= doc_content_end_char:
                document_token_indices.append(i)
            current_pos = end
        except ValueError:
            print(f"[Dummy Util] Warning: Could not find word '{word}'. Appending invalid offset.")
            all_token_offsets.append((-1,-1))
            current_pos += len(word)

    # Add dummy offsets for answer words
    all_token_offsets.extend([(-2,-2)] * answer_len) # Use different marker for answer

    # Ensure list length matches (handle potential errors during loop)
    while len(all_token_offsets) < total_len: all_token_offsets.append((-1,-1))
    all_token_offsets = all_token_offsets[:total_len]

    print(f"[Dummy Util] Prompt len (tokens): {prompt_len}")
    print(f"[Dummy Util] Answer len (tokens): {answer_len}")
    print(f"[Dummy Util] Document token indices found: {document_token_indices[:10]}...")
    print(f"[Dummy Util] Total offsets generated: {len(all_token_offsets)}")

    input_ids = torch.tensor([list(range(1000, 1000 + total_len))])

    return {
        "input_ids": input_ids,
        "document_token_indices": document_token_indices,
        "all_token_offsets": all_token_offsets,
        "prompt_len": prompt_len,
        "answer_len": answer_len,
        "doc_content_start_char": doc_content_start_char,
        "doc_content_end_char": doc_content_end_char
    }