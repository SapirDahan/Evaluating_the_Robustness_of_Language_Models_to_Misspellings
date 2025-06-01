import os
import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import dispatch_model

# Hugging Face token (optional: only needed for gated models)
token = None  # replace with your token if needed

# Inference config
BATCH_SIZE = 8
MAX_NEW_TOKENS = 64

tokenizer_cache = {}
model_cache = {}

# Define per-model quantization preference
QUANTIZATION_MODELS = {
    "Salesforce/codegen-350M-mono": "8bit",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "4bit",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": "4bit",
    "mistralai/Mistral-7B-v0.1": "4bit",
    "HuggingFaceH4/zephyr-7b-beta": "4bit",
    "openchat/openchat-3.5-0106": "4bit",
    "CohereForAI/c4ai-command-r-v01": "4bit",
    "tiiuae/falcon-7b-instruct": "4bit",
    "EleutherAI/gpt-j-6B": "4bit",
    "mosaicml/mpt-7b-instruct": "4bit",
    "mosaicml/mpt-7b-chat": "4bit",
    "mosaicml/mpt-1b-redpajama-200b": "8bit",
    "mosaicml/mpt-1b-chat": "8bit",
    "EleutherAI/pythia-1.4b": "8bit",
    "EleutherAI/pythia-2.8b": "8bit",
    "EleutherAI/gpt-neo-2.7B": "4bit",
    "EleutherAI/gpt-neo-125M": "8bit",
    "openai-community/gpt2": "8bit",
    "EleutherAI/pythia-70m": "8bit",
    "EleutherAI/pythia-160m": "8bit"
}

def load_model_and_tokenizer(model_name):
    if model_name not in tokenizer_cache:
        print(f"\n🔄 Loading model: {model_name}")
        try:
            tokenizer_args = {
                "trust_remote_code": True,
                "use_safetensors": True
            }
            model_args = {
                "trust_remote_code": True,
                "use_safetensors": True,
                "device_map": "auto",
            }
            if token is not None:
                tokenizer_args["token"] = token
                model_args["token"] = token

            # Select quantization based on model
            quant_mode = QUANTIZATION_MODELS.get(model_name, "8bit")  # Default to 8bit if not listed
            if quant_mode == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model_args["quantization_config"] = bnb_config
            else:
                model_args["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                **tokenizer_args
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_args
            )

            # Dispatch model properly
            model = dispatch_model(model, device_map="auto")
            model.eval()

            if model.config.is_encoder_decoder:
                tokenizer.padding_side = "right"
            else:
                tokenizer.padding_side = "left"

            tokenizer_cache[model_name] = tokenizer
            model_cache[model_name] = model
        except Exception as e:
            print(f"❌ Could not load {model_name}: {e}")
            return None, None
    return model_cache[model_name], tokenizer_cache[model_name]

def generate_batch(model, tokenizer, prompts, max_new_tokens):
    try:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [remove_prompt(p, o) for p, o in zip(prompts, decoded)]
    except Exception as e:
        return [f"[ERROR: {e}]" for _ in prompts]

def remove_prompt(prompt, output):
    # Remove repeated prompt from beginning of output
    if output.startswith(prompt):
        return output[len(prompt):].strip()
    # Try soft match (when there's minor tokenizer shift)
    if prompt in output:
        return output.split(prompt, 1)[-1].strip()
    return output.strip()

model_names = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
    "mistralai/Mistral-7B-v0.1",
    "HuggingFaceH4/zephyr-7b-beta",
    "openchat/openchat-3.5-0106",
    "CohereForAI/c4ai-command-r-v01",
    "tiiuae/falcon-7b-instruct",
    "EleutherAI/gpt-j-6B",
    "mosaicml/mpt-7b-instruct",
    "mosaicml/mpt-7b-chat",
    "mosaicml/mpt-1b-redpajama-200b",
    "mosaicml/mpt-1b-chat",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/gpt-neo-2.7B",
    "Salesforce/codegen-350M-mono",
    "EleutherAI/gpt-neo-125M",
    "openai-community/gpt2",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m"
]

device = 0 if torch.cuda.is_available() else -1
print(f"\n👨‍💻 Running inference on {'GPU' if device == 0 else 'CPU'}")

for model_name in model_names:
    model_dir = os.path.join("model_outputs", model_name.replace("/", "_").lower())
    if not os.path.exists(model_dir):
        print(f"⚠ Skipping {model_name} — output directory not found.")
        continue

    model, tokenizer = load_model_and_tokenizer(model_name)
    if model is None:
        continue

    json_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".json"))
    for file in json_files:
        path = os.path.join(model_dir, file)
        print(f"⚙ Processing {file}...")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to open {file}: {e}")
            continue

        for qid, entries in tqdm(data.items(), desc=f"{model_name} - {file}"):
            if qid == "model_size":
                continue

            batch = []
            batch_ids = []

            for idx, entry in enumerate(entries):
                if entry["output"] == "PLACEHOLDER":
                    batch.append(entry["input"])
                    batch_ids.append(idx)

                    # When batch full or end of list
                    if len(batch) == BATCH_SIZE or idx == len(entries) - 1:
                        outputs = generate_batch(model, tokenizer, batch, MAX_NEW_TOKENS)
                        for i, out in enumerate(outputs):
                            entries[batch_ids[i]]["output"] = out
                        batch, batch_ids = [], []

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Failed to save {file}: {e}")

print("\n✅ Inference complete.")