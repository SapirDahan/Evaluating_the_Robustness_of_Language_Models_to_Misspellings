import torch
import pandas as pd
import os
import itertools
from collections import defaultdict
from tqdm import tqdm

# ───── Configuration ───── #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───── Paths ───── #
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
questions_path = os.path.join(base_dir, 'data', 'questions.csv')
misspellings_path = os.path.join(base_dir, 'data', 'misspellings.csv')
output_path = os.path.join(base_dir, 'data', 'augmented_questions.csv')

# ───── Load CSVs ───── #
questions_df = pd.read_csv(questions_path)
misspellings_df = pd.read_csv(misspellings_path)

# ───── Parse misspellings into a dictionary ───── #
misspellings_dict = {}
for _, row in misspellings_df.iterrows():
    word = row['correct_word'].lower()
    misspells = row['misspellings'].strip().split(',')
    misspellings_dict[word] = [w.strip() for w in misspells]

# ───── Build Vocabulary ───── #
vocab = set()
for q in questions_df['question']:
    vocab.update(q.lower().split())
for w, ms in misspellings_dict.items():
    vocab.add(w)
    vocab.update(ms)

# Word <-> index mappings
word2idx = {word: idx for idx, word in enumerate(sorted(vocab))}
idx2word = {idx: word for word, idx in word2idx.items()}

# ───── Utilities ───── #
def adjust_case(original, new):
    if original.istitle():
        return new.capitalize()
    elif original.isupper():
        return new.upper()
    elif original.islower():
        return new.lower()
    return new

def tokenize(sentence):
    return [word2idx[word.lower()] for word in sentence.split()]

def detokenize(indexes, original_sentence):
    original_words = original_sentence.split()
    words = [adjust_case(original_words[i], idx2word[idx]) for i, idx in enumerate(indexes)]
    return " ".join(words)

# ───── Generate Variants on GPU ───── #
def generate_variants_gpu(sentence, max_errors=2, max_variants=1000):
    original_words = sentence.split()
    indices = torch.tensor(tokenize(sentence), device=device)

    replaceable = []
    for i, word in enumerate(original_words):
        lw = word.lower()
        if lw in misspellings_dict:
            replacements = [word2idx[mw] for mw in misspellings_dict[lw] if mw in word2idx]
            if replacements:
                replaceable.append((i, torch.tensor(replacements, device=device)))

    if len(replaceable) == 0:
        return [(sentence, 0)]

    variants = [(indices.clone(), 0)]
    total = 1

    max_errors = min(max_errors, len(replaceable))

    for error_count in range(1, max_errors + 1):
        for combo in itertools.combinations(replaceable, error_count):
            positions, choices = zip(*combo)
            for new_words in itertools.product(*choices):
                new_indices = indices.clone()
                for pos, new_word in zip(positions, new_words):
                    new_indices[pos] = new_word
                variants.append((new_indices, error_count))
                total += 1
                if total >= max_variants:
                    break
            if total >= max_variants:
                break
        if total >= max_variants:
            break

    return [(detokenize(v.tolist(), sentence), e) for v, e in variants]

# ───── Generate Full Dataset ───── #
augmented_variants = []
for sentence in tqdm(questions_df['question'], desc="Generating variants"):
    variants = generate_variants_gpu(sentence, max_errors=10)
    for variant, err in variants:
        augmented_variants.append({
            'original_question': sentence,
            'variant_question': variant,
            'error_count': err
        })

# ───── Save Output ───── #
augmented_df = pd.DataFrame(augmented_variants)
augmented_df.to_csv(output_path, index=False)
print(f"✓ Saved augmented dataset to: {output_path}")