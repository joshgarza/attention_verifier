# src/llama4_api_client.py (or add to model_handler.py)
import requests
import os
import json

# --- Get API Key ---
# It's better practice to load from env var than hardcode
LLAMA_API_KEY = os.environ.get("LLAMA_API_KEY")
if not LLAMA_API_KEY:
    print("Warning: LLAMA_API_KEY environment variable not set.")
    # Optionally raise an error or allow proceeding if key might be passed differently

LLAMA_API_ENDPOINT = "https://api.llama.com/v1/chat/completions"

# --- CHOOSE MODEL ID ---
# Check API docs or experiment if Scout is preferred/available
# NLI_MODEL_ID = "Llama-4-Scout-17B-16E-Instruct" # Try this first
NLI_MODEL_ID = "Llama-4-Maverick-17B-128E-Instruct-FP8" # Fallback if Scout not on API

def get_nli_judgment_via_api(premise: str, hypothesis: str) -> str:
    """
    Uses the Llama 4 API to perform an NLI check.

    Args:
        premise (str): The combined evidence sentences.
        hypothesis (str): The model's answer to evaluate.

    Returns:
        str: The judgment ("ENTAILMENT", "CONTRADICTION", "NEUTRAL", or "API_ERROR")
    """
    print("[NLI Check API] Performing NLI check using Llama 4 API...")

    if not LLAMA_API_KEY:
        print("[NLI Check API] Error: API Key not found.")
        return "API_ERROR"

    if not premise:
        print("[NLI Check API] No premise (evidence) provided.")
        return "NEUTRAL" # Treat no evidence as neutral

    # Construct the NLI Prompt (same careful structure as before)
    nli_prompt_content = f"""You are an expert linguistic analyst performing a Natural Language Inference task.
    Analyze the relationship between the following Premise and Hypothesis.
    Based ONLY on the information presented in the Premise, determine if the Hypothesis is entailed by the Premise, contradicts the Premise, or is neutral with respect to the Premise.
    <Premise>
    {premise}
    </Premise>
    <Hypothesis>
    {hypothesis}
    </Hypothesis>
    Classification (Choose ONE and output ONLY the single chosen word):
    ENTAILMENT
    CONTRADICTION
    NEUTRAL
    Your Answer:"""

    # Format messages for the API
    messages = [
        # Optional: System prompt for the NLI task (might help constrain output)
        # {"role": "system", "content": "You are an NLI classification assistant. Only output one word: ENTAILMENT, CONTRADICTION, or NEUTRAL."},
        {"role": "user", "content": nli_prompt_content}
    ]

    # Construct headers and payload
    headers = {
        "Authorization": f"Bearer {LLAMA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": NLI_MODEL_ID,
        "messages": messages,
        "max_tokens": 10, # Max tokens for the classification word + slight buffer
        "temperature": 0.1, # Low temp for more deterministic classification
    }

    nli_output_text = "API_ERROR"
    try:
        response = requests.post(
            LLAMA_API_ENDPOINT,
            headers=headers,
            json=payload, # Use json= instead of data= with json.dumps
            timeout=60 # Add a reasonable timeout (e.g., 60 seconds)
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        response_data = response.json()
        print(f"[NLI Check API] Full API Response: {response_data}") # Log full response for debugging

        # Extract the text response - based on the example structure
        if "completion_message" in response_data and isinstance(response_data["completion_message"], dict):
            content = response_data["completion_message"].get("content")
            if isinstance(content, dict):
                 generated_text = content.get("text", "").strip()
                 nli_output_text = generated_text.upper() # Uppercase for consistency
                 print(f"[NLI Check API] Raw API output: '{generated_text}' -> Parsed: '{nli_output_text}'")
            else:
                 print(f"[NLI Check API] Warning: 'content' field has unexpected type or missing 'text': {content}")
        elif "choices" in response_data and len(response_data["choices"]) > 0: # Check for OpenAI compatible format too
             message = response_data["choices"][0].get("message", {})
             generated_text = message.get("content", "").strip()
             nli_output_text = generated_text.upper()
             print(f"[NLI Check API] Raw API output (OpenAI format): '{generated_text}' -> Parsed: '{nli_output_text}'")
        else:
            print(f"[NLI Check API] Warning: Could not find expected text in response structure: {response_data}")


    except requests.exceptions.Timeout:
        print("[NLI Check API] Error: Request timed out.")
    except requests.exceptions.RequestException as e:
        print(f"[NLI Check API] Error: API request failed: {e}")
        # Log response body if available on error
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            try:
                print(f"Response body: {e.response.json()}")
            except json.JSONDecodeError:
                print(f"Response body (non-JSON): {e.response.text}")
    except Exception as e:
         print(f"[NLI Check API] Error processing API response: {e}")


    # Validate and return the judgment
    valid_judgments = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
    first_word = nli_output_text.split()[0] if nli_output_text else "API_ERROR"
    if first_word in valid_judgments:
        print(f"[NLI Check API] Final Judgment: {first_word}")
        return first_word
    else:
        print(f"[NLI Check API] Warning: Model output '{nli_output_text}' is not a valid judgment.")
        return "API_ERROR" # Return API_ERROR if output is unexpected or errors occurred