from transformers import pipeline
import torch

def load_model_pipeline(model_name):
    """
    Load a HuggingFace text-generation pipeline for the specified model.
    Uses GPU if available.
    """
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline("text-generation", model=model_name, device=device)
    return pipe

def query_model(model_pipeline, input_text, max_length=50):
    """
    Query the provided model pipeline with the given input_text.
    Returns the generated text.
    """
    outputs = model_pipeline(input_text, max_length=max_length, num_return_sequences=1)
    return outputs[0]['generated_text']