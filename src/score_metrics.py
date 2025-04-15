import difflib
import numpy as np
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load a SentenceTransformer model (this may take some time)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def difflib_similarity(text1, text2):
    """
    Compute similarity using difflib's SequenceMatcher.
    Returns a score between 0 and 100.
    """
    ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    return ratio * 100

def embedding_similarity(text1, text2):
    """
    Compute cosine similarity between sentence embeddings.
    Returns a score between 0 and 100.
    """
    embeddings = embedding_model.encode([text1, text2], convert_to_tensor=True)
    cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    normalized = ((cosine_sim + 1) / 2) * 100  # Normalize from [-1, 1] to [0, 100]
    return normalized

def bleu_score(reference, hypothesis):
    """
    Compute BLEU score between the reference and hypothesis texts.
    Returns a score between 0 and 100.
    """
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    smoothing = SmoothingFunction().method1
    score = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)
    return score * 100

def jaccard_similarity(text1, text2):
    """
    Compute the Jaccard similarity between two texts based on token overlap.
    Returns a score between 0 and 100.
    """
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    union = tokens1.union(tokens2)
    if not union:
        return 0
    intersection = tokens1.intersection(tokens2)
    return (len(intersection) / len(union)) * 100