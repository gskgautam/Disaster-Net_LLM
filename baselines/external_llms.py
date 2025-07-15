import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import numpy as np


def run_cllmate(texts, model_path='cllmate-base', device=None):
    """Run CLLMate baseline for text classification."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()


def run_llava(texts, images, model_path='llava-base', device=None):
    """Run LLaVA baseline for vision-language classification. Requires LLaVA library."""
    try:
        # Hypothetical import; replace with actual LLaVA import if available
        from llava.model import LlavaModel, LlavaProcessor
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        processor = LlavaProcessor.from_pretrained(model_path)
        model = LlavaModel.from_pretrained(model_path).to(device)
        # Preprocess images and texts
        inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()
    except ImportError:
        raise ImportError("LLaVA library is not installed. Please install it to use this baseline.")


def run_top1sim(texts, images, model_path='clip-base', device=None):
    """Run Top-1 Sim baseline using CLIP embeddings for similarity-based classification."""
    from transformers import CLIPProcessor, CLIPModel
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path).to(device)
    # Preprocess
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    # Move all tensor inputs to device
    for k in inputs:
        if torch.is_tensor(inputs[k]):
            inputs[k] = inputs[k].to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Compute similarity between text and image embeddings
        logits_per_image = outputs.logits_per_image  # [batch, batch]
        probs = torch.softmax(logits_per_image, dim=1)
        # For each image, get the top-1 most similar text
        top1_indices = probs.argmax(dim=1)
    return top1_indices.cpu().numpy() 