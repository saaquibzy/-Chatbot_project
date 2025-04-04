import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load the dataset
file_path = "C:/Chatbot_Project/cleaned_sarcasm_tweets.csv"

try:
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# Debugging: Check dataset structure
print("📊 Dataset shape:", df.shape)  # Should print (rows, columns)
print("📝 First few rows of dataset:\n", df.head())  # Print first 5 rows

# Check and clean column names
df.columns = df.columns.str.strip()
print("🔎 Columns in dataset:", df.columns.tolist())

# Ensure required columns exist
if "tweet" not in df.columns or "sarcasm" not in df.columns:
    print("❌ Error: Dataset does not contain required columns 'tweet' and 'sarcasm'.")
    exit()

# Check unique values in the 'sarcasm' column before mapping
print("🔍 Unique sarcasm labels before mapping:", df['sarcasm'].unique())

# Convert 'sarcasm' column from 'yes'/'no' to 1/0 and handle inconsistencies
df['sarcasm'] = df['sarcasm'].astype(str).str.strip().str.lower()
# Check unique values before mapping
print("🔍 Unique sarcasm labels before processing:", df['sarcasm'].unique())

# Ensure 'sarcasm' is in integer format (only if needed)
df['sarcasm'] = pd.to_numeric(df['sarcasm'], errors='coerce')

# Drop rows with missing sarcasm values
df = df.dropna(subset=['sarcasm'])

# Convert sarcasm column to integer
df['sarcasm'] = df['sarcasm'].astype(int)

# Check unique values after processing
print("✅ Unique sarcasm labels after processing:", df['sarcasm'].unique())


# Drop rows with missing labels
df = df.dropna(subset=['sarcasm'])

# Convert to integer
df['sarcasm'] = df['sarcasm'].astype(int)

# Check unique values after mapping
print("✅ Unique sarcasm labels after mapping:", df['sarcasm'].unique())

# Check for missing values
print("📉 Missing values before cleanup:\n", df.isnull().sum())

# Drop rows with missing tweets
df = df.dropna(subset=['tweet'])
df['tweet'] = df['tweet'].astype(str)  # Ensure all tweets are strings

# Final dataset check
print("📊 Dataset shape after removing NaNs:", df.shape)
print("📌 Number of samples after cleaning:", len(df))
print("🔍 Sample data after cleaning:\n", df.head())

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=128, return_tensors="pt")
        
        # Ensure labels are integers
        labels = pd.to_numeric(labels, errors='coerce').fillna(0).astype(int)
        
        # Debugging: Print unique labels
        print("📌 Unique labels in dataset:", set(labels))

        self.labels = torch.tensor(labels.tolist(), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Split data into train and test
try:
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['tweet'], df['sarcasm'], test_size=0.2, random_state=42
    )
    print("✅ Data split successful.")
except Exception as e:
    print(f"❌ Error during train-test split: {e}")
    exit()

train_dataset = SarcasmDataset(train_texts, train_labels)
test_dataset = SarcasmDataset(test_texts, test_labels)

# Load DistilBERT model with 2 output labels (Sarcasm: Yes/No)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="C:/Chatbot_Project/results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="C:/Chatbot_Project/logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
try:
    print("🚀 Starting training...")
    trainer.train()
    print("✅ Training complete!")
except Exception as e:
    print(f"❌ Error during training: {e}")
    exit()

# Save the trained model
model.save_pretrained("C:/Chatbot_Project/trained_sarcasm_model")
tokenizer.save_pretrained("C:/Chatbot_Project/trained_sarcasm_model")

print("✅ Model saved successfully!")
