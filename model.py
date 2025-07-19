from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import numpy as np
import os

saved_model_path = os.path.join(os.path.dirname(__file__), "saved_model")
try:
    tokenizer = BertTokenizer.from_pretrained(saved_model_path)
    model = BertForSequenceClassification.from_pretrained(saved_model_path)
    model.eval()
except Exception as e:
    print(f"[ERROR] Failed to load model or tokenizer: {e}")
    tokenizer = None
    model = None

class_names = ['Fake', 'Real']

def predict_proba(texts):
    if model is None or tokenizer is None:
        raise RuntimeError("Model or tokenizer not loaded.")
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
    return probs.cpu().numpy()

def predict_news(text):
    if model is None or tokenizer is None:
        raise RuntimeError("Model or tokenizer not loaded.")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)
        label = "Real" if pred.item() == 1 else "Fake"

    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba, num_features=5)
    explanation = exp.as_list()

    return label, confidence.item(), explanation