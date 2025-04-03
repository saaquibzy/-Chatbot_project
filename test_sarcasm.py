from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model_path = r"C:\Chatbot_Project\distilbert-base-uncased"  # Use the correct directory

# Load tokenizer and model manually
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create a pipeline
sarcasm_detector = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Test the model
print(sarcasm_detector("Oh great, another Monday!"))
