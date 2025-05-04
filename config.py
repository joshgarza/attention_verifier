import torch

# --- Model Configuration ---
# The exact model ID you downloaded
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
# --- Hardware Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# You might need specific GPU ID if multiple are present: "cuda:0"
print(f"Using device: {DEVICE}")

# --- Attention Processing Configuration ---
# Example: How many top evidence spans to retrieve
TOP_K_EVIDENCE = 5

# --- Other Paths ---
# CACHE_DIR = "/path/to/your/cache" # Optional: If you want to specify model cache location