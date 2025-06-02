import os
import json
import torch
import gc
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import dispatch_model

# Hugging Face token (optional: only needed for gated models)
token = "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"

# Inference config
BATCH_SIZE = 16
MAX_NEW_TOKENS = 32

tokenizer_cache = {}
model_cache = {}

# Define small models (no quantization)
SMALL_MODELS = [
    "Salesforce/codegen-350M-mono",
    "EleutherAI/gpt-neo-125M",
    "openai-community/gpt2",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    #"mosaicml/mpt-1b-redpajama-200b",
    "EleutherAI/pythia-1.4b"
]

# Define per-model quantization preference for large models
QUANTIZATION_MODELS = {
    "mosaicml/mpt-1b-redpajama-200b": "8bit",
    # "mosaicml/mpt-1b-chat": "8bit",  # Commented - invalid model
    "EleutherAI/pythia-1.4b": "8bit",
    "HuggingFaceH4/zephyr-7b-beta": "8bit",
}

def load_model_and_tokenizer(model_name):
    SPECIAL_MODELS = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
        "HuggingFaceH4/zephyr-7b-beta",
        "openchat/openchat-3.5-0106"
    ]

    FALCON_MODELS = [
        "tiiuae/falcon-7b-instruct"
    ]

    GPTJ_MODELS = [
        "EleutherAI/gpt-j-6B"
    ]

    MPT_MODELS = [
        "mosaicml/mpt-7b-instruct",
        "mosaicml/mpt-7b-chat"
    ]

    if model_name not in tokenizer_cache:
        print(f"\n🔄 Loading model: {model_name}")
        try:
            tokenizer_args = {
                "use_safetensors": True,
                "token": token  # Always pass the token
            }

            if model_name in SMALL_MODELS:
                model_args = {
                    "use_safetensors": True,
                    "token": token
                }
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model_args = {
                    "use_safetensors": True,
                    "device_map": "auto",
                    "quantization_config": bnb_config,
                    "token": token
                }
                # Only add trust_remote_code if needed
                if (model_name not in FALCON_MODELS and model_name not in MPT_MODELS) or model_name == "mosaicml/mpt-1b-redpajama-200b":
                    model_args["trust_remote_code"] = True

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                **tokenizer_args
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model according to type
            if model_name in SPECIAL_MODELS:
                if "openchat" in model_name.lower():
                    from transformers import LlamaForCausalLM
                    model = LlamaForCausalLM.from_pretrained(
                        model_name,
                        **model_args
                    )
                else:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        **model_args
                    )
            elif model_name in FALCON_MODELS:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_args
                )
            elif model_name in GPTJ_MODELS:
                from transformers import GPTJForCausalLM
                model = GPTJForCausalLM.from_pretrained(
                    model_name,
                    **model_args
                )
            elif model_name in MPT_MODELS:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_args
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_args
                )

            if model_name in SMALL_MODELS:
                pass  # No dispatch needed
            else:
                pass  # Already auto-dispatched

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
    # "mistralai/Mistral-7B-v0.1",
    # "HuggingFaceH4/zephyr-7b-beta",
    # "openchat/openchat-3.5-0106", # done until here
    "mosaicml/mpt-7b-instruct",
    "mosaicml/mpt-7b-chat",
    "tiiuae/falcon-7b-instruct", #slow
    # "mosaicml/mpt-1b-redpajama-200b", # done
    # "EleutherAI/pythia-1.4b", # done
    # "EleutherAI/pythia-2.8b", # done from here until down
    # "EleutherAI/gpt-neo-2.7B",
    # "Salesforce/codegen-350M-mono",
    # "EleutherAI/gpt-neo-125M",
    # "openai-community/gpt2",
    # "EleutherAI/pythia-70m",
    # "EleutherAI/pythia-160m"
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
