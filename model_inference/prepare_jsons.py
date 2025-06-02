import os
import json
import pandas as pd
from collections import defaultdict

csv_path = os.path.join("..", "data", "augmented_questions.csv")
output_root = "model_outputs"

model_names = [
    "mistralai/Mistral-7B-v0.1",
    "HuggingFaceH4/zephyr-7b-beta",
    "openchat/openchat-3.5-0106",
    "tiiuae/falcon-7b-instruct",
    "EleutherAI/gpt-j-6B",
    "mosaicml/mpt-7b-instruct",
    "mosaicml/mpt-7b-chat",
    "mosaicml/mpt-1b-redpajama-200b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/gpt-neo-2.7B",
    "Salesforce/codegen-350M-mono",
    "EleutherAI/gpt-neo-125M",
    "openai-community/gpt2",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m"
]

parameter_counts = {
    "mistralai/Mistral-7B-v0.1": 7.3e9,
    "HuggingFaceH4/zephyr-7b-beta": 7e9,
    "openchat/openchat-3.5-0106": 7e9,
    "tiiuae/falcon-7b-instruct": 7e9,
    "mosaicml/mpt-7b-instruct": 6.7e9,
    "mosaicml/mpt-7b-chat": 6.7e9,
    "mosaicml/mpt-1b-redpajama-200b": 1.3e9,
    "EleutherAI/pythia-1.4b": 1.4e9,
    "EleutherAI/pythia-2.8b": 2.8e9,
    "EleutherAI/gpt-neo-2.7B": 2.7e9,
    "Salesforce/codegen-350M-mono": 0.35e9,
    "EleutherAI/gpt-neo-125M": 0.125e9,
    "openai-community/gpt2": 0.124e9,
    "EleutherAI/pythia-70m": 0.07e9,
    "EleutherAI/pythia-160m": 0.16e9
}

df = pd.read_csv(csv_path)
required_columns = {"original_question", "variant_question", "error_count"}
assert required_columns.issubset(df.columns), "Missing required columns in CSV."

df["question_id"] = df.groupby("original_question").ngroup()
df["model_output"] = "PLACEHOLDER"

grouped = defaultdict(lambda: {"orig": None, "variants": defaultdict(list)})
for _, row in df.iterrows():
    qid = str(row["question_id"])
    if row["error_count"] == 0:
        grouped[qid]["orig"] = {
            "id": f"{qid}_orig",
            "input": row["original_question"],
            "output": "PLACEHOLDER"
        }
    else:
        grouped[qid]["variants"][row["error_count"]].append({
            "id": f"{qid}_var{len(grouped[qid]['variants'][row['error_count']]) + 1}",
            "input": row["variant_question"],
            "output": "PLACEHOLDER"
        })

for model in model_names:
    print(f"Creating JSON for: {model}")
    model_dir = os.path.join(output_root, model.replace("/", "_").lower())
    os.makedirs(model_dir, exist_ok=True)
    model_size = int(parameter_counts.get(model, -1))

    for err_count in sorted(df["error_count"].unique()):
        if err_count == 0:
            continue
        output_data = {"model_size": model_size}
        for qid, content in grouped.items():
            if err_count in content["variants"]:
                output_data[qid] = [content["orig"]] + content["variants"][err_count]

        sorted_data = {"model_size": model_size}
        for qid in sorted(output_data.keys(), key=lambda x: int(x) if x != "model_size" else -1):
            if qid != "model_size":
                sorted_data[qid] = output_data[qid]

        with open(os.path.join(model_dir, f"{err_count}.json"), "w", encoding="utf-8") as f:
            json.dump(sorted_data, f, indent=2, ensure_ascii=False)

print("\n✅ All JSON files generated.")
