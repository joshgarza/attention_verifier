# src/tokenizer_utils.py
import torch

def prepare_tokenizer_info_for_attention(
    tokenizer,
    prompt_text,
    answer_text,
    attention_weights,
    max_model_input_len=4096 # Match generate length
    ):
    """
    Prepares a dictionary containing tokenizer information needed for attention processing.

    Args:
        tokenizer: The loaded Hugging Face tokenizer.
        prompt_text (str): The full prompt text fed to the model.
        answer_text (str): The generated answer text.
        attention_weights (tuple): The attention weights tuple from model.generate().
        max_model_input_len (int): The max_length used during tokenization for generate().

    Returns:
        dict: A dictionary containing 'prompt_len', 'answer_len',
              'all_token_offsets', 'document_token_indices',
              'doc_content_start_char', 'doc_content_end_char'.
              Returns None if essential information cannot be determined.
    """
    print("[Tokenizer Utils] Preparing tokenizer info for attention processing...")
    if tokenizer is None:
        print("[Tokenizer Utils] Error: Tokenizer is None.")
        return None

    try:
        # 1. Tokenize the original prompt to get length and offset mapping
        prompt_encoding = tokenizer(
            prompt_text,
            return_offsets_mapping=True,
            max_length=max_model_input_len,
            truncation=True,
            return_tensors="pt"
        )
        prompt_len_calc = prompt_encoding.input_ids.shape[-1]
        prompt_offsets = prompt_encoding.get('offset_mapping')
        if prompt_offsets is not None:
             prompt_offsets = prompt_offsets[0].tolist()
        else:
             print("[Tokenizer Utils] Warning: Tokenizer did not return offset_mapping.")
             prompt_offsets = [(None, None)] * prompt_len_calc # Fallback

        # 2. Estimate answer length from attention tensor dimensions
        answer_len_calc = 0
        if attention_weights and isinstance(attention_weights, (list, tuple)) and len(attention_weights) > 0:
            last_layer_attn = attention_weights[-1]
            if isinstance(last_layer_attn, tuple) and len(last_layer_attn) > 0:
                 last_layer_attn = last_layer_attn[0] # Handle nested tuple
            if isinstance(last_layer_attn, torch.Tensor):
                 key_len = last_layer_attn.shape[-1]
                 if key_len >= prompt_len_calc:
                      answer_len_calc = key_len - prompt_len_calc
                 else:
                      print(f"[Tokenizer Utils] Warning: Attn key dim ({key_len}) < prompt len ({prompt_len_calc}). ans_len=0.")
            else:
                 print(f"[Tokenizer Utils] Warning: Last attn layer not Tensor ({type(last_layer_attn)}).")
        elif answer_text and not answer_text.startswith("Error"):
             answer_tokens_est = tokenizer(answer_text, add_special_tokens=False).input_ids
             answer_len_calc = len(answer_tokens_est)
             print("[Tokenizer Utils] Warning: Estimating answer length by tokenizing answer text.")
        else:
             print("[Tokenizer Utils] Warning: Could not determine answer length. Setting to 0.")

        # 3. Combine offsets (Prompt offsets + None padding for answer)
        if not isinstance(prompt_offsets, list): prompt_offsets = [(None, None)] * prompt_len_calc
        all_offsets_calc = prompt_offsets + [(None, None)] * answer_len_calc

        # 4. Calculate document token indices and boundaries
        doc_content_start_char, doc_content_end_char = find_document_boundaries(prompt_text)
        document_token_indices_calc = calculate_document_indices(prompt_offsets, doc_content_start_char, doc_content_end_char)

        # 5. Create and validate the dictionary
        output_dict = {
            "prompt_len": prompt_len_calc,
            "answer_len": answer_len_calc,
            "all_token_offsets": all_offsets_calc,
            "document_token_indices": document_token_indices_calc,
            "doc_content_start_char": doc_content_start_char,
            "doc_content_end_char": doc_content_end_char
        }

        # Validate required keys have non-None values
        required_keys = list(output_dict.keys())
        if not all(k in output_dict and output_dict[k] is not None for k in required_keys):
             missing = {k: output_dict.get(k) for k in required_keys if output_dict.get(k) is None}
             print(f"[Tokenizer Utils] Warning: Prepared dict has None values for required keys: {missing}")
             # Decide whether to return None or the incomplete dict
             # return None # Stricter approach
             print("[Tokenizer Utils] Returning potentially incomplete tokenizer info dict.")


        print("[Tokenizer Utils] Tokenizer info prepared.")
        return output_dict

    except Exception as e:
        print(f"[Tokenizer Utils] Error during tokenizer info preparation: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_document_boundaries(prompt_text):
    """Helper to find start/end character indices of the document within the prompt."""
    doc_content_start_char = -1
    doc_content_end_char = -1
    doc_start_marker = "Document:\n"
    doc_end_marker = "\n\nQuestion:"
    doc_content_start_idx = prompt_text.find(doc_start_marker)
    doc_content_end_idx = prompt_text.find(doc_end_marker)

    if doc_content_start_idx != -1:
        doc_content_start_char = doc_content_start_idx + len(doc_start_marker)
    if doc_content_end_idx != -1 and (doc_content_start_char == -1 or doc_content_start_char <= doc_content_end_idx):
        doc_content_end_char = doc_content_end_idx
    else:
        doc_content_end_char = len(prompt_text)
    if doc_content_start_char == -1:
         doc_content_start_char = 0
    if doc_content_start_char >= doc_content_end_char:
         doc_content_start_char = 0
         doc_content_end_char = len(prompt_text)
         print("[Tokenizer Utils] Warning: Could not reliably determine document boundaries. Using full prompt.")

    return doc_content_start_char, doc_content_end_char


def calculate_document_indices(prompt_offsets, doc_start_char, doc_end_char):
    """Helper to find token indices corresponding to the document span."""
    document_token_indices = []
    if prompt_offsets is None: return document_token_indices

    for i, offset in enumerate(prompt_offsets):
        if isinstance(offset, tuple) and len(offset) == 2 and \
           offset[0] is not None and offset[1] is not None and \
           offset != (0, 0) and \
           offset[0] >= doc_start_char and \
           offset[1] <= doc_end_char and \
           offset[0] < offset[1]:
             document_token_indices.append(i)
    return document_token_indices