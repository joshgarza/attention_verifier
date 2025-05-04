# src/model_handler.py
import config # Your configuration file (config.py)
import torch
# Import necessary Hugging Face classes
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc # Garbage Collector

# --- Global variables to hold the loaded model and tokenizer ---
MODEL = None
TOKENIZER = None

def load_model_and_tokenizer():
    """
    Loads the language model and tokenizer specified in config.py.
    Uses INT4 quantization via bitsandbytes.
    Handles model caching and potential errors during loading.
    """
    global MODEL, TOKENIZER

    # Only load if they haven't been loaded already
    if MODEL is None or TOKENIZER is None:
        print(f"Loading model and tokenizer from Hugging Face ID: {config.MODEL_ID} with INT4...")
        print("Ensure you are logged in via 'huggingface-cli login'")

        # Attempt to clear CUDA cache before loading to maximize free memory
        if torch.cuda.is_available():
            print("Clearing CUDA cache before model load...")
            # torch.cuda.empty_cache()
            # gc.collect()

        try:
            # Load the tokenizer associated with the model ID
            TOKENIZER = AutoTokenizer.from_pretrained(config.MODEL_ID)

            # Configure INT4 quantization using bitsandbytes
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,                # Enable 4-bit loading
                bnb_4bit_quant_type="fp4",        # Use NF4 quantization type (common standard)
                bnb_4bit_use_double_quant=False,   # Use nested quantization for memory saving
                bnb_4bit_compute_dtype=torch.bfloat16 # Perform computations in BF16 for speed/accuracy
            )
            print("Using BitsAndBytes INT4 configuration.")

            # Load the model using AutoModelForCausalLM
            MODEL = AutoModelForCausalLM.from_pretrained(
                config.MODEL_ID,
                quantization_config=quantization_config, # Apply the INT4 config
                # device_map="auto",                 # Use accelerate to automatically distribute layers (tries to fit on available GPUs)
                trust_remote_code=True,            # Necessary for some models with custom code
                attn_implementation="eager"
                # low_cpu_mem_usage=True,          # Optional: Can help with CPU RAM, but remove if issues occur
                # No torch_dtype needed when quantization_config is specified
            )

            # Set the model to evaluation mode (disables dropout, etc.)
            MODEL.eval()
            print("Model and tokenizer loaded successfully via Transformers with INT4 quantization.")

        except Exception as e:
            # Provide detailed error message if loading fails
            print(f"Error loading model/tokenizer via Transformers with INT4: {e}")
            print("Troubleshooting:")
            print("- Ensure 'bitsandbytes' is installed correctly (`pip install bitsandbytes`).")
            print("- Did you run 'huggingface-cli login' and grant access to the gated repo?")
            print("- Check GPU memory (nvidia-smi). Is the 80GB H100 nearly full from other processes?")
            print("- If OOM during load: Try removing `device_map='auto'` to force loading only on GPU 0 (might fail earlier if >80GB needed).")
            print("- Check model ID in config.py is correct ('meta-llama/Llama-4-Scout-17B-16E-Instruct').")
            # Reset globals on failure
            MODEL, TOKENIZER = None, None
            # Clear cache on failure too
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            # Re-raise the exception to stop the program if loading fails
            raise

        finally:
            # Clear cache AFTER successful load too, to free temporary loading memory
            if torch.cuda.is_available():
                 print("Clearing CUDA cache post-load...")
                 torch.cuda.empty_cache()
            gc.collect()

    # Return the loaded (or previously loaded) model and tokenizer
    return MODEL, TOKENIZER


def get_answer_and_attention(model, tokenizer, document_text, question):
    """
    Generates an answer to a question based on a document using the loaded model,
    and attempts to retrieve attention weights.

    Args:
        model: The loaded language model.
        tokenizer: The loaded tokenizer.
        document_text (str): The context document.
        question (str): The question being asked.

    Returns:
        tuple: (answer_text: str, attention_weights: object or None)
               Attention weights structure depends on the model/library (usually tuple of tensors).
               Returns None for attention_weights if generation fails or attention is not output.
    """
    print("Generating answer and extracting attention using Transformers (INT4)...")

    # Construct the prompt in the expected format for the Instruct model
    prompt = f"""Document:
{document_text}

Question: {question}

Answer:"""

    # Define a maximum input length to avoid OOM during tokenization/generation
    # Start relatively small, especially for Scout on a single GPU. Increase if feasible.
    max_model_input_len = 4096
    print(f"Tokenizing input with max_length={max_model_input_len} and truncation...")
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",            # Return PyTorch tensors
            max_length=max_model_input_len, # Apply max length
            truncation=True                 # Enable truncation if prompt exceeds max length
        )
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return f"Error during tokenization: {e}", None

    # Determine the correct device for the input tensors
    # This can be tricky with device_map and quantization
    try:
        # A common way is to check the device of the first parameter
        input_device = next(model.parameters()).device
        # If the first parameter is offloaded ('meta'), default to cuda:0
        if input_device == torch.device('meta'):
             input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
             print(f"First model parameter on 'meta' device, placing input on {input_device}.")
    except Exception:
        # Fallback if accessing parameters fails
        input_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Could not determine model device reliably, assuming {input_device} for input.")

    # Move tokenized inputs to the determined device
    try:
        inputs = inputs.to(input_device)
        print(f"Inputs moved to device: {inputs.input_ids.device}")
    except Exception as e:
         print(f"Error moving inputs to device {input_device}: {e}")
         return f"Error moving inputs to device: {e}", None

    # Initialize variables for the results
    outputs = None
    answer_text = "Error during generation." # Default error message
    attention_weights = None

    try:
        # Perform inference within torch.no_grad() to save memory
        with torch.no_grad():
            # Call the generate method
            outputs = model.generate(
                **inputs,                     # Pass tokenized inputs
                max_new_tokens=250,           # Limit the length of the generated answer
                output_attentions=True,       # *** Request attention scores ***
                return_dict_in_generate=True, # Return a structured output object
                pad_token_id=tokenizer.eos_token_id # Set padding token for generation
            )

        # Process the output if generation succeeded
        if outputs is not None:
             # Decode the generated tokens, skipping the input prompt part
             answer_sequence = outputs.sequences[0]
             # Determine the start index of the generated answer tokens
             answer_start_index = inputs.input_ids.shape[-1] # Length of the input sequence fed to generate
             answer_text = tokenizer.decode(answer_sequence[answer_start_index:], skip_special_tokens=True)

             # Extract attention weights if they were output
             if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                attention_weights = outputs.attentions
                print(f"Successfully generated answer and obtained attention weights (Tuple with {len(attention_weights)} layer tensors).")
             else:
                print("Generation completed, but attention weights were not found in the output.")
                # Keep answer_text, but attention_weights remains None
        else:
             # This case indicates generate() returned None unexpectedly
             print("Generation outputs object is None despite no explicit error.")
             answer_text = "Generation returned None unexpectedly."


    except Exception as e:
        # Handle errors during the generation process
        print(f"Error during model generation: {e}")
        answer_text = f"Error during generation: {e}" # Store the error message
        attention_weights = None # Ensure attention is None on error
        # Print memory summary on error for debugging
        if torch.cuda.is_available():
             print(torch.cuda.memory_summary(device=None, abbreviated=False))

    finally:
        # Clean up tensors to free GPU memory, checking if they exist first
        if 'outputs' in locals() and outputs is not None:
             del outputs
        if 'inputs' in locals() and inputs is not None:
             del inputs
        # Clear CUDA cache after generation attempt
        if torch.cuda.is_available():
             print("Clearing CUDA cache post-generation...")
             torch.cuda.empty_cache()
        gc.collect()

    # Return the generated text and the attention weights (which might be None)
    return answer_text, attention_weights