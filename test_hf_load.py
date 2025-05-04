import torch
import config
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup
model_name = config.MODEL_ID
device = torch.device("cuda:0")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move model to GPU and enable gradient checkpointing
model.to(device)
model.gradient_checkpointing_enable()

# Input preparation
input_text = "Your input text here"
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs.input_ids.to(device)

# Forward pass with attention output
with torch.cuda.amp.autocast():
    outputs = model(input_ids, output_attentions=True)
    attention_weights = outputs.attentions

# Do something with attention_weights
print(attention_weights)