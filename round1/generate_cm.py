import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from sklearn.metrics import confusion_matrix
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

val_loader = DataLoader(val_dataset, batch_size=16)

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
print("Getting predictions...")
all_preds = []
all_targets = []

with torch.no_grad():
    for d in tqdm(val_loader):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Generate Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(24, 20)) # Large size for 46 classes
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=list(label_map.keys()),
            yticklabels=list(label_map.keys()))
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix - Baseline XLM-R (No Weights)')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

save_path = os.path.join(REPORTS_DIR, 'transformer_confusion_matrix.png')
plt.savefig(save_path)
print(f"Confusion matrix saved to {save_path}")
