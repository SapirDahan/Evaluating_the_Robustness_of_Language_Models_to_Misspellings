import os
import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline, BitsAndBytesConfig

token = "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"  # Replace with your real token or use env var

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

tokenizer_cache = {}
model_cache = {}

def load_model(model_name):
    if model_name not in tokenizer_cache:
        print(f"🔄 Loading tokenizer and model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            token=token
        )
        generator = TextGenerationPipeline(model=model, tokenizer=tokenizer)
        tokenizer_cache[model_name] = tokenizer
        model_cache[model_name] = generator
    return model_cache[model_name]

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
print(f"\n🖥 Running inference on {'GPU' if device == 0 else 'CPU'}")

for model in model_names:
    model_dir = os.path.join("model_outputs", model.replace("/", "_").lower())
    if not os.path.exists(model_dir):
        print(f"⚠ Skipping {model} — output directory not found.")
        continue

    print(f"\n🔄 Loading model: {model}")
    try:
        generator = load_model(model)
    except Exception as e:
        print(f"❌ Could not load {model}: {e}")
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

        for qid, entries in tqdm(data.items(), desc=f"{model} - {file}"):
            if qid == "model_size":
                continue
            for entry in entries:
                if entry["output"] == "PLACEHOLDER":
                    try:
                        result = generator(entry["input"], max_new_tokens=64, do_sample=False)[0]["generated_text"]
                        entry["output"] = result
                    except Exception as e:
                        entry["output"] = f"[ERROR: {e}]"

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Failed to save {file}: {e}")

print("\n✅ Inference complete.")
