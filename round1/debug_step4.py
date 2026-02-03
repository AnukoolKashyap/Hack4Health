import pandas as pd
import numpy as np
import json
import os
from collections import Counter

DATA_DIR = 'd:/hack4health/Hack4Health/round1'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.xlsx')
VAL_PATH = os.path.join(DATA_DIR, 'val.xlsx')
LABEL_MAP_PATH = os.path.join(DATA_DIR, 'label_map.json')
REPORTS_DIR = os.path.join(DATA_DIR, 'reports')

# 1. Load Data
print("Loading data...")
train_df = pd.read_excel(TRAIN_PATH)
val_df = pd.read_excel(VAL_PATH)

# Context Scoring Logic
CONTEXT_KEYWORDS = {
    "devotion": ["bhakti", "god", "lord", "worship", "faith", "prayer", "divine", "soul", "sacred", "temple"],
    "morality": ["dharma", "duty", "truth", "right", "wrong", "virtue", "justice", "sin", "ethics", "moral"],
    "nature": ["sun", "moon", "star", "sky", "cloud", "rain", "wind", "river", "ocean", "sea"],
    "symbolism": ["dream", "shadow", "mystery", "light", "dark", "illusion", "reflection", "metaphor"]
}

def compute_context_scores(text):
    text = str(text).lower()
    scores = []
    for context, keywords in CONTEXT_KEYWORDS.items():
        match_count = sum(1 for k in keywords if k in text)
        score = min(1.0, match_count / 2.0) 
        scores.append(score)
    return scores

# Redirect output to file
with open('debug_output.txt', 'w') as f:
    f.write(f"Train samples: {len(train_df)}\n")
    f.write(f"Val samples: {len(val_df)}\n\n")
    f.write("Top 5 Train Classes:\n")
    f.write(str(train_df['label_id'].value_counts().head(5)) + "\n\n")
    
    sample_scores = [compute_context_scores(t) for t in train_df['cleaned_poem'].head(50)]
    nonzero_scores = [s for s in sample_scores if sum(s) > 0]
    f.write(f"Out of 50 samples, {len(nonzero_scores)} have non-zero context scores.\n")
    f.write(f"Sample scores: {nonzero_scores[:5]}\n\n")

    with open(os.path.join(REPORTS_DIR, 'transformer_metrics.json'), 'r') as json_f:
        metrics = json.load(json_f)
    f.write("Performance Summary:\n")
    f.write(f"Accuracy: {metrics['accuracy']}\n")
    f.write(f"Macro F1: {metrics['macro avg']['f1-score']}\n")
    active_classes = [k for k, v in metrics.items() if isinstance(v, dict) and v['f1-score'] > 0 and k not in ['macro avg', 'weighted avg']]
    f.write(f"Classes with F1 > 0: {active_classes}\n")
    
print("Debug info written to debug_output.txt")
