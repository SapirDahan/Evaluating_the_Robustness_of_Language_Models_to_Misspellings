import torch
import pandas as pd
import os
import random
from tqdm import tqdm

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
questions_path = os.path.join(base_dir, 'Data', 'questions.csv')
misspellings_path = os.path.join(base_dir, 'Data', 'misspellings.csv')
output_path = os.path.join(base_dir, 'Data', 'augmented_questions.csv')

#  Load CSV
questions_df = pd.read_csv(questions_path)
misspellings_df = pd.read_csv(misspellings_path)

# Parse misspellings into a dict
misspellings_dict = {}
for _, row in misspellings_df.iterrows():
    word = row['correct_word'].lower()
    miss = [m.strip() for m in row['misspellings'].split(',') if m.strip()]
    if miss:
        misspellings_dict[word] = miss

# Build vocab and mappings
vocab = set()
for q in questions_df['question']:
    vocab.update(q.lower().split())
for w, ms in misspellings_dict.items():
    vocab.add(w)
    vocab.update(ms)

word2idx = {w: i for i, w in enumerate(sorted(vocab))}
idx2word = {i: w for w, i in word2idx.items()}

# ───── Case-aware replacer ───── #
def adjust_case(orig: str, new: str) -> str:
    if orig.istitle():   return new.capitalize()
    if orig.isupper():   return new.upper()
    if orig.islower():   return new.lower()
    return new

# Tokenize / Detokenize functions
def tokenize(sent: str) -> list[int]:
    return [word2idx[w.lower()] for w in sent.split()]

def detokenize(idxs: list[int], orig_sent: str) -> str:
    orig_words = orig_sent.split()
    out_words = []
    for i, idx in enumerate(idxs):
        w_new = idx2word[idx]
        out_words.append(adjust_case(orig_words[i], w_new))
    return " ".join(out_words)

# Generate variants, leveraging GPU for token ops
def generate_variants_gpu(sentence: str, max_errors: int = 10, variants_per_error: int = 10):
    orig_words  = sentence.split()
    lower_words = [w.lower() for w in orig_words]
    valid_pos   = [i for i, w in enumerate(lower_words) if w in misspellings_dict]
    if not valid_pos:
        return [(sentence, 0)]

    # push token IDs to device once
    orig_ids = torch.tensor(tokenize(sentence), device=device)
    valid_pos_t = torch.tensor(valid_pos, device=device)

    variants = [(sentence, 0)]

    for err_cnt in range(1, max_errors + 1):
        generated = 0
        attempts  = 0
        max_attempts = variants_per_error * 10

        if len(valid_pos) < err_cnt:
            break

        while generated < variants_per_error and attempts < max_attempts:
            attempts += 1

            # pick err_cnt distinct positions on GPU
            perm = torch.randperm(len(valid_pos_t), device=device)
            chosen = valid_pos_t[perm[:err_cnt]].tolist()

            new_ids = orig_ids.clone()  # still on GPU

            # for each chosen position, pick a random misspelling
            for pos in chosen:
                correct = lower_words[pos]
                miss_list = misspellings_dict.get(correct, [])
                if not miss_list:
                    continue
                m = random.choice(miss_list)
                new_ids[pos] = word2idx[m]

            # pull back to CPU for detokenization
            new_ids_cpu = new_ids.cpu().tolist()
            new_sent   = detokenize(new_ids_cpu, sentence)
            variants.append((new_sent, err_cnt))
            generated += 1

    return variants

# Main augmentation loop
augmented = []
for q in tqdm(questions_df['question'], desc="Augmenting"):
    for var, errs in generate_variants_gpu(q, max_errors=10, variants_per_error=10):
        augmented.append({
            'original_question': q,
            'variant_question': var,
            'error_count': errs
        })

# Save augmented dataset
aug_df = pd.DataFrame(augmented)
aug_df.to_csv(output_path, index=False)
print(f"✓ Saved augmented dataset to: {output_path}")
