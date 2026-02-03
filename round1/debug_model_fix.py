import os
import torch
import torch.nn as nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from torch.optim import AdamW
import pandas as pd
import numpy as np

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

DATA_DIR = 'd:/hack4health/Hack4Health/round1'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.xlsx')

# Load tiny subset
df = pd.read_excel(TRAIN_PATH).head(16)
texts = df['cleaned_poem'].values
labels = torch.tensor(df['label_id'].values, dtype=torch.long).to(device)

# Tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
encoding = tokenizer(
    list(texts),
    padding='max_length',
    truncation=True,
    max_length=128,
    return_tensors='pt'
)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

print(f"Input IDs shape: {input_ids.shape}")
print(f"Input IDs sample: {input_ids[0, :10]}") # Check if meaningful tokens exist

# Model
class SimpleXLMR(nn.Module):
    def __init__(self, n_classes=46):
        super(SimpleXLMR, self).__init__()
        self.bert = XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.out(pooled)

model = SimpleXLMR().to(device)

# Freeze BERT (Fix 4 condition)
for param in model.bert.parameters():
    param.requires_grad = False

optimizer = AdamW(model.parameters(), lr=2e-5) # Fix 3 condition
loss_fn = nn.CrossEntropyLoss()

print("\n--- Before Step ---")
print(f"Head Weight Norm: {model.out.weight.norm().item()}")
print(f"Head Bias Norm: {model.out.bias.norm().item()}")

# Forward
outputs = model(input_ids, attention_mask)
loss = loss_fn(outputs, labels)
print(f"Loss: {loss.item()}")

# Backward
loss.backward()

print("\n--- Gradients ---")
if model.out.weight.grad is not None:
    print(f"Head Weight Grad Norm: {model.out.weight.grad.norm().item()}")
else:
    print("Head Weight Grad: None")

if model.bert.encoder.layer[0].output.dense.weight.grad is not None:
    print(f"BERT Layer 0 Grad Norm: {model.bert.encoder.layer[0].output.dense.weight.grad.norm().item()}")
else:
    print("BERT Layer 0 Grad: None (Expected due to freeze)")

# Step
optimizer.step()

print("\n--- After Step ---")
print(f"Head Weight Norm: {model.out.weight.norm().item()}")
print(f"Head Bias Norm: {model.out.bias.norm().item()}")
