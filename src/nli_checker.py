import torch

def get_nli_judgment(
    model,
    tokenizer,
    evidence_sentences, # List of strings
    hypothesis_answer, # The original model answer string
    max_input_length=2048 # Use a smaller context for the NLI task
    ):
    """
    Uses the loaded LLM to perform an NLI check between evidence and hypothesis.

    Args:
        model: The loaded language model.
        tokenizer: The loaded tokenizer.
        evidence_sentences (list[str]): List of evidence sentences.
        hypothesis_answer (str): The model's answer to evaluate.
        max_input_length (int): Max tokens for the NLI prompt.

    Returns:
        str: The judgment ("ENTAILMENT", "CONTRADICTION", "NEUTRAL", or "ERROR")
    """
    print("[NLI Check] Performing NLI check using the LLM...")

    if not evidence_sentences:
        print("[NLI Check] No evidence sentences provided.")
        return "NEUTRAL" # Or maybe ERROR? Neutral seems safer.

    # Combine evidence sentences into a single premise string
    premise = " ".join(evidence_sentences)

    # --- Construct the NLI Prompt ---
    # This needs careful crafting to constrain the model's output.
    # Using delimiters and clear instructions helps.
    nli_prompt = f"""You are an expert linguistic analyst performing a Natural Language Inference task.
    Analyze the relationship between the following Premise and Hypothesis.
    Based ONLY on the information presented in the Premise, determine if the Hypothesis is entailed by the Premise, contradicts the Premise, or is neutral with respect to the Premise.

    <Premise>
    {premise}
    </Premise>

    <Hypothesis>
    {hypothesis_answer}
    </Hypothesis>

    Classification (Choose ONE and output ONLY the single chosen word):
    ENTAILMENT
    CONTRADICTION
    NEUTRAL

    Your Answer:""" # Prompt the model to fill in the classification

    print(f"[NLI Check] NLI Prompt (truncated):\n'''{nli_prompt[:500]}...'''")

    # Determine input device (same logic as get_answer_and_attention)
    try:
        input_device = next(model.parameters()).device
        if input_device == torch.device('meta'):
             input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    except Exception:
        input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Tokenize the NLI prompt
    try:
        inputs = tokenizer(
            nli_prompt,
            return_tensors="pt",
            max_length=max_input_length, # Keep NLI context reasonable
            truncation=True
        ).to(input_device)
    except Exception as e:
        print(f"[NLI Check] Error during NLI prompt tokenization: {e}")
        return "ERROR"

    # Initialize output variable
    nli_output_text = "ERROR"

    # Generate the classification using the LLM
    try:
        with torch.no_grad():
            # Generate only a few tokens, expecting just the classification word
            outputs = model.generate(
                **inputs,
                max_new_tokens=5, # Should be enough for one word
                output_attentions=False, # No need for attention here
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
                # Optional: Use low temperature for deterministic classification?
                # temperature=0.1,
                # do_sample=True # if using temperature < 1
            )

        if outputs is not None:
             output_sequence = outputs.sequences[0]
             output_start_index = inputs.input_ids.shape[-1]
             # Decode generated text and clean it up
             decoded_output = tokenizer.decode(output_sequence[output_start_index:], skip_special_tokens=True)
             nli_output_text = decoded_output.strip().upper() # Get first word, uppercase
             print(f"[NLI Check] Raw NLI output: '{decoded_output}' -> Parsed: '{nli_output_text}'")
        else:
             print("[NLI Check] NLI generation returned None.")
             nli_output_text = "ERROR"

    except Exception as e:
        print(f"[NLI Check] Error during NLI generation: {e}")
        nli_output_text = "ERROR"
        if torch.cuda.is_available():
             print(torch.cuda.memory_summary(device=None, abbreviated=False)) # Check for OOM

    finally:
        # Cleanup
        if 'outputs' in locals() and outputs is not None: del outputs
        if 'inputs' in locals() and inputs is not None: del inputs
        # No need for heavy cache clearing here usually unless OOM happens

    # Validate and return the judgment
    valid_judgments = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
    # Take only the first word in case model generated more
    first_word = nli_output_text.split()[0] if nli_output_text else "ERROR"
    if first_word in valid_judgments:
        return first_word
    else:
        print(f"[NLI Check] Warning: Model output '{nli_output_text}' is not a valid judgment.")
        return "ERROR" # Return ERROR if output is unexpected