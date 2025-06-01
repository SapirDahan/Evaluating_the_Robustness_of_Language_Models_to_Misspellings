import os
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from itertools import combinations

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load sentence embedding model
model = SentenceTransformer("all-mpnet-base-v2", device=DEVICE)
model.max_seq_length = 256  # Optional: speed boost if outputs are short

# Paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
output_dir = os.path.join(project_root, "model_inference", "model_outputs")
save_path = os.path.join(project_root, "evaluation", "similarity_vector_results.json")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

results = []

# Loop over model folders
model_dirs = sorted([d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))])
for model_dir in tqdm(model_dirs, desc="🔍 Evaluating models"):
    model_path = os.path.join(output_dir, model_dir)
    outputs_by_error = {}
    model_size = "unknown"

    # Load all error-level files (e.g., 0.json, 1.json, ...)
    for file in sorted(os.listdir(model_path)):
        if not file.endswith(".json"):
            continue

        error_count = int(file.split(".")[0])
        file_path = os.path.join(model_path, file)
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Extract model size if available
        if isinstance(data, dict) and "model_size" in data:
            model_size = data["model_size"]
            del data["model_size"]

        for qid, entries in data.items():
            outputs = [e["output"] for e in entries if not e["id"].endswith("_orig")]
            if len(outputs) > 1:
                outputs_by_error.setdefault(error_count, []).append(outputs)

    # Compute similarities
    for err_count, answer_lists in tqdm(outputs_by_error.items(), desc=f"→ {model_dir}", leave=False):
        similarities = []
        for outputs in answer_lists:
            embeddings = model.encode(outputs, convert_to_tensor=True, device=DEVICE)
            sim_matrix = util.cos_sim(embeddings, embeddings) * 100

            for i, j in combinations(range(len(outputs)), 2):
                similarities.append(round(sim_matrix[i][j].item(), 2))

        results.append({
            "model": model_dir.replace("_", "/"),
            "model_size": model_size,
            "error_count": err_count,
            "similarities": similarities
        })

# Save to JSON
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print(f"✅ Done! Results saved to: {save_path}")