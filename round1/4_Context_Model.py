import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from collections import defaultdict

# Setup Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Paths
DATA_DIR = 'd:/hack4health/Hack4Health/round1'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.xlsx')
VAL_PATH = os.path.join(DATA_DIR, 'val.xlsx')
LABEL_MAP_PATH = os.path.join(DATA_DIR, 'label_map.json')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
REPORTS_DIR = os.path.join(DATA_DIR, 'reports')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# 1. Load Data
print("Loading data...")
train_df = pd.read_excel(TRAIN_PATH)
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

train_dataset = PoetryDataset(
    train_df['cleaned_poem'].values,
    train_df['label_id'].values,
    tokenizer
)

val_dataset = PoetryDataset(
    val_df['cleaned_poem'].values,
    val_df['label_id'].values,
    tokenizer
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# FIX 1 & 2: Use XLMRobertaForSequenceClassification, NO CUSTOM HEAD
print("Initializing model (XLMRobertaForSequenceClassification)...")
model = XLMRobertaForSequenceClassification.from_pretrained(
    "xlm-roberta-base",
    num_labels=num_classes,
    problem_type="single_label_classification"
)
model = model.to(device)

# Training Setup (FIX 4)
EPOCHS = 3
optimizer = AdamW(model.parameters(), lr=2e-5) # Fix 3: No scheduler mentioned in strict fix, LR 2e-5

# FIX 3: No Class Weights initially
loss_fn = nn.CrossEntropyLoss() 

def train_epoch(model, data_loader, optimizer, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    all_preds_epoch = []
    
    for d in tqdm(data_loader, desc="Training"):
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["labels"].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=targets
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        all_preds_epoch.extend(preds.cpu().numpy())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        
    # FIX 5: Sanity Check
    unique_preds, counts = np.unique(all_preds_epoch, return_counts=True)
    print(f"\n[SANITY CHECK] Unique Predictions in this epoch: {len(unique_preds)}")
    print(f"Prediction Counts (Top 10): {dict(zip(unique_preds[:10], counts[:10]))} ...")
    
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=targets
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, np.mean(losses)

# Training Loop
best_f1 = 0 
history = {
    'train_acc': [],
    'train_loss': [],
    'val_acc': [],
    'val_loss': []
}

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    
    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        device,
        len(train_df)
    )
    
    print(f'Train loss {train_loss} accuracy {train_acc}')
    
    val_acc, val_loss = eval_model(
        model,
        val_loader,
        device,
        len(val_df)
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    
    # Store history
    history['train_acc'].append(train_acc.item() if torch.is_tensor(train_acc) else train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc.item() if torch.is_tensor(val_acc) else val_acc)
    history['val_loss'].append(val_loss)
    
    # Macro F1 Check
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for d in val_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    print(f'Val Macro F1: {val_f1}')
    
    if val_f1 > best_f1:
        torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'xlmr_fixed_model.bin'))
        best_f1 = val_f1
        print("Saved best model.")
        
# Save History
with open(os.path.join(REPORTS_DIR, 'training_history.json'), 'w') as f:
    json.dump(history, f, indent=4)

# Plot Training Curves
print("Generating training plots...")
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history['train_acc'], label='Train Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'training_curves.png'))
print(f"Plots saved to {os.path.join(REPORTS_DIR, 'training_curves.png')}")

# Final Evaluation
print("Loading best model for evaluation...")
model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'xlmr_fixed_model.bin')))
model = model.to(device)

def get_predictions(model, data_loader):
    model = model.eval()
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            
            predictions.extend(preds)
            real_values.extend(targets)
            
    predictions = torch.stack(predictions).cpu()
    real_values = torch.stack(real_values).cpu()
    return predictions, real_values

y_pred, y_test = get_predictions(model, val_loader)

# Classification Report
print(classification_report(y_test, y_pred, target_names=list(label_map.keys())))

# Save Metrics
report = classification_report(y_test, y_pred, target_names=list(label_map.keys()), output_dict=True)
with open(os.path.join(REPORTS_DIR, 'transformer_metrics.json'), 'w') as f:
    json.dump(report, f, indent=4)
    
print("Evaluation complete. Results saved.")
