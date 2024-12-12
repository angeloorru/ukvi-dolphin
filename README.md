# ukvi-dolphin
dolphin-llama LLM inference trained with UK Visa and Immigration data 


#### 1. Create a python virtual env:
Python V3.10
`python3 -m venv .env_name`
`source .env_name/bin/activate`

#### 2. Run:
Run below only if on AMD GPU
`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6`

if not on AMD GPU run `pip install torch`. Details here: https://pytorch.org/get-started/locally/#linux-installation

and then:

`pip install -r requirements.txt`

#### 3. Run the model through LM Studio
LM Studio page: https://lmstudio.ai/


Before running the trained inference in LM Studio we need to generate a .gguf file.
This is done in few steps:

#### 4. Merge the LoRA Adapters with Dolphin-llama base model:
- Run the python script: `python merge_model.py`. This will generate a new directory.
- Go to Huggingface website: https://huggingface.co/cognitivecomputations/dolphin-2.9.4-llama3.1-8b/tree/main
- Download the following files and put them in a directory called `base_model`:
    - config.json 
    - generation_config.json 
    - model-00001-of-00004.safetensors
    - model-00002-of-00004.safetensors
    - model-00003-of-00004.safetensors
    - model-00004-of-00004.safetensors
    - model.safetensors.index.json
    - special_tokens_map.json
    - tokenizer.json
    - trainer_state.json
    - tokenizer_config.json
- Copy the following files from the base_model dir into the newly created merged_model dir:
    - special_tokens_map.json 
    - tokenizer.json
    - trainer_state.json
    - tokenizer_config.json


#### 5. To generate a gguf file use the script available here:
- Clone the repository: https://github.com/ggerganov/llama.cpp
- Run this command to create a .gguf file with the base dolphin llama and the LoRA adapters with the UKVI data:
`python3 convert_hf_to_gguf.py   --outfile llama-ukvi-model.gguf   --outtype f16   ./merged_model`

Wait for the script to complete, and you should see the generated llama-ukvi-model.gguf file

Import the new gguf file into LM Studio to test the dolphin-llama inference trained with UKVI data.

You can generate also a gguf file with the base dolphin-llama model to compare it against the one 
trained with UKVI data. 

For this, you can use this command:  `python3 convert_hf_to_gguf.py   --outfile dolphin-llama.gguf   --outtype f16   ./base_model`


*Note: Step 5 needs to be done on the llama.cpp repo. Copy the merged_model and base_model into the repo to execute the 
python scripts correctly.

#### 6. To test the LoRA adapters you can run the inference_test_training.py script:
`python trained_inference_test.py`


### Alternatives to LM Studio
- Oobabooga Web UI
- KoboldAI
- GPT4All
- Hugging Face Transformers

You may need to adjust certain parameters depending on the capabilities of your hardware.