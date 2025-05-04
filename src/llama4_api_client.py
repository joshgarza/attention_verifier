# src/llama4_api_client.py
import requests
import os
import json
import re # Import regular expressions

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
    Improved parsing to handle chatty output.

    Args:
        premise (str): The combined evidence sentences.
        hypothesis (str): The model's answer to evaluate.

    Returns:
        str: The judgment ("ENTAILMENT", "CONTRADICTION", "NEUTRAL", or "API_ERROR")
    """
    print("[NLI Check API] Performing NLI check using Llama 4 API...")
    # ... (API Key check, premise check remain the same) ...
    if not LLAMA_API_KEY: return "API_ERROR"
    if not premise: return "NEUTRAL"

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

    messages = [{"role": "user", "content": nli_prompt_content}]
    headers = {"Authorization": f"Bearer {LLAMA_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": NLI_MODEL_ID, "messages": messages, "max_tokens": 300, "temperature": 0.1} # Increased max_tokens slightly just in case

    nli_output_text_raw = "" # Store the raw text for parsing
    nli_parsed_judgment = "API_ERROR" # Default return

    try:
        response = requests.post(LLAMA_API_ENDPOINT, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        print(f"[NLI Check API] Full API Response: {response_data}")

        # Extract the text response
        generated_text = ""
        if "completion_message" in response_data and isinstance(response_data["completion_message"], dict):
            content = response_data["completion_message"].get("content")
            if isinstance(content, dict): generated_text = content.get("text", "").strip()
        elif "choices" in response_data and len(response_data["choices"]) > 0:
            message = response_data["choices"][0].get("message", {})
            generated_text = message.get("content", "").strip()
        else:
            print(f"[NLI Check API] Warning: Could not find expected text in response structure.")

        nli_output_text_raw = generated_text # Store raw output
        print(f"[NLI Check API] Raw API output: '{nli_output_text_raw}'")

        # --- *** NEW PARSING LOGIC *** ---
        if nli_output_text_raw:
            # Check for keywords in order of precedence (Contradiction > Entailment > Neutral)
            # Use case-insensitive search within the raw text
            raw_upper = nli_output_text_raw.upper() # Search in uppercase

            # Look for definitive statements near the end if possible (like "The best answer is: **WORD**")
            # More robust: search the whole text
            if "CONTRADICTION" in raw_upper or "CONTRADICTS" in raw_upper :
                 nli_parsed_judgment = "CONTRADICTION"
            elif "ENTAILMENT" in raw_upper or "ENTAILED" in raw_upper or "SUPPORTED" in raw_upper : # Added SUPPORTED based on model output
                 nli_parsed_judgment = "ENTAILMENT"
            elif "NEUTRAL" in raw_upper:
                 nli_parsed_judgment = "NEUTRAL"
            else:
                 print(f"[NLI Check API] Warning: Could not find valid judgment keyword in output.")
                 nli_parsed_judgment = "API_ERROR" # Fallback if no keyword found
        else:
             print("[NLI Check API] Warning: API returned empty text.")
             nli_parsed_judgment = "API_ERROR"
        # --- *** END NEW PARSING LOGIC *** ---

    # ... (keep error handling: Timeout, RequestException, etc.) ...
    except requests.exceptions.RequestException as e:
        print(f"[NLI Check API] Error: API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None: print(f"Response status: {e.response.status_code}")
        nli_parsed_judgment = "API_ERROR"
    except Exception as e:
         print(f"[NLI Check API] Error processing API response: {e}")
         nli_parsed_judgment = "API_ERROR"


    print(f"[NLI Check API] Final Judgment: {nli_parsed_judgment}")
    return nli_parsed_judgment # Return the parsed judgment