# src/utils.py
import torch
import random
import string # For more varied dummy text

def create_dummy_attention(num_layers, num_heads, prompt_len, answer_len, device='cpu', make_signal=True, signal_indices=None, signal_strength=10.0):
    """Creates a dummy attention tuple matching transformers output structure.
       Can optionally add a stronger signal to specific indices."""
    total_seq_len = prompt_len + answer_len
    dummy_attentions = []
    for layer_idx in range(num_layers):
        # Create random attention scores
        layer_attention = torch.rand(
            (1, num_heads, total_seq_len, total_seq_len), device=device
        )

        # --- Add Optional Signal ---
        # Make attention from answer tokens TO specific prompt tokens higher
        if make_signal and signal_indices:
             answer_start_idx = prompt_len
             # Ensure indices are within prompt length bounds
             valid_signal_indices = [idx for idx in signal_indices if 0 <= idx < prompt_len]
             if valid_signal_indices:
                # For each answer token (query), increase attention to signal indices (key)
                layer_attention[0, :, answer_start_idx:, valid_signal_indices] += signal_strength * torch.rand_like(layer_attention[0, :, answer_start_idx:, valid_signal_indices])

        # Optional: Softmax for more realistic distribution (can make signal less obvious)
        # layer_attention = torch.softmax(layer_attention, dim=-1)
        dummy_attentions.append(layer_attention)

    print(f"[Dummy Util] Created dummy attention: {num_layers} layers, {num_heads} heads, seq_len {total_seq_len}")
    return tuple(dummy_attentions)

def create_dummy_tokenizer_output(prompt, answer, tokenizer_vocab_size=32000):
    """Creates dummy token IDs and simulates offset mapping."""
    print("[Dummy Util] Creating dummy tokenizer output...")
    # --- Refined Document Boundary Detection ---
    doc_start_marker = "Document:\n"
    doc_end_marker = "\n\nQuestion:" # Use the exact marker
    doc_content_start_char = -1
    doc_content_end_char = -1

    doc_content_start_idx = prompt.find(doc_start_marker)
    doc_content_end_idx = prompt.find(doc_end_marker)

    if doc_content_start_idx != -1:
        doc_content_start_char = doc_content_start_idx + len(doc_start_marker)
    else:
        print("[Dummy Util] Warning: Could not find 'Document:\\n' marker.")
        doc_content_start_char = 0 # Fallback

    if doc_content_end_idx != -1:
         # Ensure end marker is after start marker if both found
         if doc_content_start_char <= doc_content_end_idx :
              doc_content_end_char = doc_content_end_idx
         else:
              print("[Dummy Util] Warning: 'Question:' marker found before 'Document:'. Using end of prompt.")
              doc_content_end_char = len(prompt) # Fallback
    else:
        print("[Dummy Util] Warning: Could not find '\\n\\nQuestion:' marker.")
        doc_content_end_char = len(prompt) # Fallback

    # Ensure start is not after end
    if doc_content_start_char >= doc_content_end_char and doc_content_end_char != -1:
         print(f"[Dummy Util] Warning: Calculated doc start ({doc_content_start_char}) >= end ({doc_content_end_char}). Check markers/prompt. Resetting to full prompt.")
         doc_content_start_char = 0
         doc_content_end_char = len(prompt)

    print(f"[Dummy Util] Document boundaries (chars): Start={doc_content_start_char}, End={doc_content_end_char}")

    # Simple space-split tokenization for dummy mapping
    words = prompt.split() # Split by any whitespace
    answer_words = answer.split()

    prompt_len = len(words)
    answer_len = len(answer_words)
    total_len = prompt_len + answer_len

    all_token_offsets = [] # Offsets for ALL tokens (prompt + answer)
    document_token_indices = [] # Indices *within the prompt part* that belong to the doc

    current_pos = 0
    # Calculate offsets for prompt words
    for i, word in enumerate(words):
        try:
            # Find word occurrence after current_pos, ignoring leading whitespace difference
            stripped_word = word.strip()
            if not stripped_word: # Skip empty strings from multiple spaces
                 all_token_offsets.append((-1,-1)) # Mark as invalid offset
                 continue

            start = prompt.index(stripped_word, current_pos)
            end = start + len(stripped_word)
            offset_tuple = (start, end)
            all_token_offsets.append(offset_tuple)

            # Check if this word falls within the document character boundaries
            if start >= doc_content_start_char and end <= doc_content_end_char:
                document_token_indices.append(i) # Append the 0-based index 'i'

            current_pos = end

        except ValueError:
            # Handle cases where word might not be found simply
            print(f"[Dummy Util] Warning: Could not find word '{word}' starting from pos {current_pos}. Appending invalid offset.")
            all_token_offsets.append((-1,-1)) # Indicate error/skip
            current_pos += len(word) # Approximate move past word

    # Add dummy offsets for answer words (not strictly needed for current logic, but completes the structure)
    current_pos = len(prompt) # Rough estimate
    for i, word in enumerate(answer_words):
         stripped_word = word.strip()
         if not stripped_word:
              all_token_offsets.append((-1,-1))
              continue
         # We don't have the real answer text appended, so just add dummy markers
         all_token_offsets.append((-2,-2)) # Use different marker for answer tokens

    # Ensure offset list matches total length (adjust if split created empty strings)
    # Recalculate prompt_len based on valid offsets found for prompt
    prompt_len = sum(1 for offset in all_token_offsets[:prompt_len] if offset != (-1, -1) and offset != (-2,-2))
    # Ensure all_token_offsets matches total length (may not if errors occurred)
    # Pad if necessary, although errors are more likely
    while len(all_token_offsets) < total_len:
         all_token_offsets.append((-1,-1))
    all_token_offsets = all_token_offsets[:total_len] # Truncate if too long


    print(f"[Dummy Util] Prompt len (tokens): {prompt_len}")
    print(f"[Dummy Util] Answer len (tokens): {answer_len}")
    print(f"[Dummy Util] Total sequence len (tokens): {total_len}")
    print(f"[Dummy Util] Document token indices found: {document_token_indices}")
    print(f"[Dummy Util] Total offsets generated: {len(all_token_offsets)}")

    # Generate fake input IDs (less important now, but keeps structure)
    input_ids = torch.tensor([list(range(1000, 1000 + total_len))])

    return {
        "input_ids": input_ids,
        "document_token_indices": document_token_indices,
        "all_token_offsets": all_token_offsets, # Use this one
        "prompt_len": prompt_len,
        "answer_len": answer_len,
        # Keep doc boundaries for potential use later if needed
        "doc_content_start_char": doc_content_start_char,
        "doc_content_end_char": doc_content_end_char
    }

# --- Add a simple test block for the dummy util itself ---
if __name__ == "__main__":
     print("--- Testing Dummy Util ---")
     dummy_doc = "This is the first sentence. Here is the second sentence which has info. The third sentence."
     dummy_question = "What info is in the second sentence?"
     dummy_answer = "The second sentence has info."
     # Ensure the prompt contains the EXACT markers
     dummy_prompt = f"Document:\n{dummy_doc}\n\nQuestion: {dummy_question}\n\nAnswer:"
     print(f"Test Prompt:\n'''{dummy_prompt}'''")

     tokenizer_info = create_dummy_tokenizer_output(dummy_prompt, dummy_answer)
     print("\nGenerated Tokenizer Info:")
     for key, value in tokenizer_info.items():
          if isinstance(value, list) and len(value) > 10:
               print(f"  {key}: List of length {len(value)} starting with {value[:5]}...")
          else:
               print(f"  {key}: {value}")

     print("\nCreating dummy attention with signal...")
     # Find indices corresponding to "second sentence"
     target_phrase = "second sentence"
     target_indices = []
     words = dummy_prompt.split()
     start_char_target = dummy_prompt.find(target_phrase)
     if start_char_target != -1:
          end_char_target = start_char_target + len(target_phrase)
          for i, offset in enumerate(tokenizer_info['all_token_offsets'][:tokenizer_info['prompt_len']]):
               if offset != (-1,-1):
                    # Check if token overlaps with target phrase
                    if max(offset[0], start_char_target) < min(offset[1], end_char_target):
                         target_indices.append(i)
     print(f"Indices for '{target_phrase}': {target_indices}")

     attn = create_dummy_attention(
          4, 12,
          tokenizer_info['prompt_len'],
          tokenizer_info['answer_len'],
          make_signal=True,
          signal_indices=target_indices # Add signal to where "second sentence" is
     )
     print(f"Dummy attention created (type: {type(attn)}, length: {len(attn)})")
     print("--- End Dummy Util Test ---")