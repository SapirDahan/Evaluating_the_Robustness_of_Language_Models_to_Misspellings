import itertools


def generate_augmented_variants(sentence, misspellings_dict, max_errors=10):
    """
    Generate augmented variants of a sentence by introducing misspellings.

    Returns a list of tuples: (augmented_sentence, error_count).
    - The original sentence is always included (error_count 0).
    - For error counts 1 to max_errors, only words that are found in misspellings_dict (case-insensitive) are replaced.

    Capitalization Handling:
      - If the original word is in title case (first letter uppercase), each candidate replacement is converted to title case.
    """
    words = sentence.split()

    # Identify positions where the lowercase word exists in our dictionary.
    candidate_indices = [i for i, word in enumerate(words) if word.lower() in misspellings_dict]
    effective_max_errors = min(max_errors, len(candidate_indices))

    variants = [(sentence, 0)]

    # Debug: Show which words are eligible for replacement.
    print("DEBUG - Processing sentence:", sentence)
    print("DEBUG - Candidate indices for replacement:", candidate_indices)

    for error_count in range(1, effective_max_errors + 1):
        for indices in itertools.combinations(candidate_indices, error_count):
            replacement_options = []
            for i in indices:
                key = words[i].lower()
                candidates = misspellings_dict.get(key, [])
                # If the original word starts with an uppercase letter, capitalize each candidate.
                if words[i][0].isupper():
                    candidates = [cand.capitalize() for cand in candidates]
                replacement_options.append(candidates)
                # Debug: Print the candidates after processing for the word at index i.
                print(f"DEBUG - For word '{words[i]}' candidates after capitalization:", candidates)
            # Generate all combinations of replacements.
            for replacements in itertools.product(*replacement_options):
                new_words = words.copy()
                for j, idx in enumerate(indices):
                    new_words[idx] = replacements[j]
                variant_sentence = " ".join(new_words)
                variants.append((variant_sentence, error_count))
    return variants