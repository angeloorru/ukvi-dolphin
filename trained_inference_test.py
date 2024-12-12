import torch
from transformers import LlamaForCausalLM, AutoTokenizer
from peft import PeftModel

# Specify device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the LoRA adapter configuration
peft_model_id = "trained_model/lora_model"
model_name = "cognitivecomputations/dolphin-2.9.4-llama3.1-8b"

# Load the base model
base_model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
).to(device)

# Apply the LoRA adapters to the base model
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

# Set pad_token to eos_token without adding new tokens
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

# Update the model's configuration
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Prepare the input in the same format as training
#input_text = "How can I make a freedom of information request in the UK?"
#input_text = "Can you tell me how to report an immigration crime?"
#input_text = "How can I apply for a visa to come to the UK as a student?"
#input_text = "How can I apply for an electronic travel authorisation in the UK?"
#input_text = "Can you explain how to apply for EU settlement scheme family permit in the UK?"
#input_text = "How can I get a student visa in the UK?"
#input_text = "How can I bring pets to the UK from EU?"
#input_text = "What is the Windrush Compensation Scheme?"
#input_text = "What is the life in the UK test?"
#input_text = "What is an eVisa and can I use it?"
input_text = "How can I report tax fraud to HMRC?"


inputs = tokenizer(
    input_text,
    return_tensors="pt",
    padding=True,
    truncation=True
).to(device)

generation_kwargs = {
    "min_new_tokens": 50,
    "max_new_tokens": 512,
    "do_sample": True,
    "top_p": 0.95,
    "temperature": 0.7,
    "repetition_penalty": 1.2,
    "no_repeat_ngram_size": 3,
    "early_stopping": False,
    "length_penalty": 1.0
}



# Generate output
with torch.no_grad():
    generated_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        **generation_kwargs
    )

# Decode the output
output_text = tokenizer.decode(
    generated_ids[0],
    skip_special_tokens=True,
)

if output_text.startswith(input_text):
    output_text = output_text[len(input_text):].strip()

print(f"\nInput:\n{input_text}\n")
print(f"Output:\n{output_text}\n")
