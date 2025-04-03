import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Specify the path to your locally saved trained model
model_path = "trained_sarcasm_model"

# Load the tokenizer and model from the local directory
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

def detect_sarcasm(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # Get model predictions
    outputs = model(**inputs)
    # Apply softmax to obtain probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    # The probability for class '1' (sarcasm) is at index 1
    sarcasm_score = probs[0][1].item()
    # Return True if sarcasm probability is greater than 50%
    return sarcasm_score > 0.5

