import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Paths
DATA_DIR = 'd:/hack4health/Hack4Health/round1'
VAL_PATH = os.path.join(DATA_DIR, 'val.xlsx')
LABEL_MAP_PATH = os.path.join(DATA_DIR, 'label_map.json')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
REPORTS_DIR = os.path.join(DATA_DIR, 'reports')

# Load Data
print("Loading data...")
val_df = pd.read_excel(VAL_PATH)

with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
    
num_classes = len(label_map)
class_names = list(label_map.keys())

# Dataset Class
class PoetryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("Initializing tokenizer...")
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

val_dataset = PoetryDataset(
    val_df['cleaned_poem'].values,
    val_df['label_id'].values,
    tokenizer
)

val_loader = DataLoader(val_dataset, batch_size=32)

# Load Model
print("Loading model...")
model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=num_classes,
    problem_type="single_label_classification"
)
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'xlmr_fixed_model.bin')))
model = model.to(device)
model.eval()

# Get Predictions
print("Running inference...")
all_preds = []
all_probs = []

with torch.no_grad():
    for d in tqdm(val_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        
        _, preds = torch.max(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(torch.max(probs, dim=1)[0].cpu().numpy()) # Max probability (Confidence)

# 1. Prediction Distribution Plot
print("Generating Prediction Distribution Plot...")
plt.figure(figsize=(20, 10))
sns.countplot(x=all_preds)
plt.title('Predicted Class Distribution (Validation Set)')
plt.xlabel('Class ID')
plt.ylabel('Count')
plt.xticks(tuple(label_map.values()), list(label_map.keys()), rotation=90)
plt.tight_layout()
dist_path = os.path.join(REPORTS_DIR, 'inference_prediction_distribution.png')
plt.savefig(dist_path)
print(f"Saved: {dist_path}")

# 2. Confidence Histogram
print("Generating Confidence Histogram...")
plt.figure(figsize=(12, 6))
sns.histplot(all_probs, bins=20, kde=True)
plt.title('Prediction Confidence Distribution')
plt.xlabel('Confidence (Max Probability)')
plt.ylabel('Count')
plt.axvline(np.mean(all_probs), color='r', linestyle='--', label=f'Mean Confidence: {np.mean(all_probs):.2f}')
plt.legend()
plt.tight_layout()
conf_path = os.path.join(REPORTS_DIR, 'inference_confidence_histogram.png')
plt.savefig(conf_path)
print(f"Saved: {conf_path}")

print("Inference plots generated successfully.")
