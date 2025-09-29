import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# --- 1. Configuration ---
MODEL_NAME = "OceanOmics/eDNABERT-S_16S"
NUM_LABELS = None  
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "./eDNABERT_finetuned"
DATA_PATH = "your_edna_data.csv" 
SEQUENCE_COLUMN = "sequence"
LABEL_COLUMN = "label"

# --- 2. Load and Prepare Data ---
df = pd.read_csv(DATA_PATH)

# Encode labels
label_encoder = LabelEncoder()
df[LABEL_COLUMN + "_encoded"] = label_encoder.fit_transform(df[LABEL_COLUMN])
NUM_LABELS = len(label_encoder.classes_)
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
label2id = {label: i for i, label in enumerate(label_encoder.classes_)}

train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# --- 3. Custom Dataset Class ---
class EDNADataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_len=128):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = str(self.sequences[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- 4. Load Tokenizer and Model ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

# --- 5. Create Datasets ---
train_dataset = EDNADataset(
    train_df[SEQUENCE_COLUMN].tolist(),
    train_df[LABEL_COLUMN + "_encoded"].tolist(),
    tokenizer
)
val_dataset = EDNADataset(
    val_df[SEQUENCE_COLUMN].tolist(),
    val_df[LABEL_COLUMN + "_encoded"].tolist(),
    tokenizer
)

# --- 6. Define Metrics ---
metric = evaluate.load("f1") # You can also use "accuracy", "precision", "recall"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average="weighted")

# --- 7. Configure Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none", 
    logging_dir='./logs',
    logging_steps=100,
)

# --- 8. Create Trainer and Train ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()
print("Training complete!")

# --- 9. Save the fine-tuned model ---
trainer.save_model(OUTPUT_DIR + "/final_model")
tokenizer.save_pretrained(OUTPUT_DIR + "/final_model")
label_encoder_path = OUTPUT_DIR + "/final_model/label_encoder.pkl"
import pickle
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"Model and tokenizer saved to {OUTPUT_DIR}/final_model")
print(f"Label encoder saved to {label_encoder_path}")



 from transformers import pipeline
 loaded_model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR + "/final_model")
 loaded_tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR + "/final_model")
 with open(label_encoder_path, 'rb') as f:
    loaded_label_encoder = pickle.load(f)

 classifier = pipeline("sentiment-analysis", model=loaded_model, tokenizer=loaded_tokenizer)

