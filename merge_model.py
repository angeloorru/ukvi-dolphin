from peft import PeftModel
from transformers import LlamaForCausalLM

# Load base model
base_model = LlamaForCausalLM.from_pretrained("cognitivecomputations/dolphin-2.9.4-llama3.1-8b")

# Load LoRA-adapted model
lora_model = PeftModel.from_pretrained(base_model, "trained_model/lora_model")

# Merge LoRA into base model
merged_model = lora_model.merge_and_unload()

# Save the fully merged model
merged_model.save_pretrained("./merged_model")
