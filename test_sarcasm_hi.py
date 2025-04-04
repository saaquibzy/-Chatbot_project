import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load model and tokenizer
model_path = r"C:\Chatbot_Project\trained_sarcasm_model"  # Use raw string (r"") or double backslashes (\\)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Test sentences
test_texts = [
    "Oh sure, because I *love* doing extra work on weekends.",
    "Yeah, right, this is exactly what I wanted!",
    "This is the best day ever!",
    "The sun is shining and the birds are singing."
]

# Run predictions
for text in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sarcastic_prob = probabilities[0][1].item()
    not_sarcastic_prob = probabilities[0][0].item()

    print(f"Text: {text} -> Not Sarcastic: {not_sarcastic_prob:.4f}, Sarcastic: {sarcastic_prob:.4f}")
