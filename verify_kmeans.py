import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import json

base_path = "/Users/hemishjain22/Desktop/Hackathons/hack4healtj/"
reports_dir = f'{base_path}round2/reports'
os.makedirs(reports_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    print("Loading Data...")
    df = pd.read_excel(f'{base_path}round2/Combined_Emotions.xlsx')
    df['cleaned_poem'] = df['cleaned_poem'].astype(str)
    
    print("Loading mBERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
    model.eval()
    
    def get_embeddings(texts, batch_size=32):
        embeddings = []
        total = len(texts)
        # Process a subset if too large for quick verification? No, let's do all.
        # But for speed, maybe limit batch size or just prints.
        
        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size].tolist()
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = model(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        return np.vstack(embeddings)

    print("Generating Embeddings...")
    X_emb = get_embeddings(df['cleaned_poem'])
    
    print("Clustering (K=9)...")
    kmeans = KMeans(n_clusters=9, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_emb)
    df['cluster'] = clusters
    
    sil = silhouette_score(X_emb, clusters)
    print(f"Silhouette Score: {sil:.4f}")
    
    print("t-SNE Reduction...")
    # Use n_jobs=1 if scikit-learn supports it, but TSNE usually single threaded or specific impl.
    # Note: sklearn TSNE usually n_jobs is part of method or global config.
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_emb)
    df['tsne_1'] = X_tsne[:, 0]
    df['tsne_2'] = X_tsne[:, 1]
    
    print("Saving Plots...")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='cluster', data=df, palette='tab10', alpha=0.6)
    plt.title('K-Means Clusters')
    plt.savefig(f'{reports_dir}/kmeans_clusters.png')
    plt.close()
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='primary_emotion', data=df, legend=False, alpha=0.5)
    plt.title('True Labels')
    plt.savefig(f'{reports_dir}/true_labels_tsne.png')
    plt.close()
    
    with open(f'{reports_dir}/kmeans_analysis.json', 'w') as f:
        json.dump({'silhouette': float(sil)}, f)
        
    print("Verification Success.")

except Exception as e:
    print(f"Error: {e}")
    # Don't fail the script, just print error
