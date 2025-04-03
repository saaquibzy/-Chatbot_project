import torch
import pandas as pd
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizerFast
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

# Load and preprocess sarcasm dataset (Replace with actual dataset path or Hugging Face dataset)
def load_data():
    df = pd.read_json("sarcasm_dataset.json")  # Ensure you have a dataset
    df = df[['text', 'label']]  # Columns: text (sentence), label (0 = not sarcastic, 1 = sarcastic)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    train_dataset = Dataset.from_dict({'text': train_texts.tolist(), 'label': train_labels.tolist()})
    val_dataset = Dataset.from_dict({'text': val_texts.tolist(), 'label': val_labels.tolist()})
    return train_dataset, val_dataset

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset, val_dataset = load_data()
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./sarcasm_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./sarcasm_model")
tokenizer.save_pretrained("./sarcasm_model")

print("Training complete! Model saved at ./sarcasm_model")
