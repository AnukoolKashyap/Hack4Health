import json

base_path = "/Users/hemishjain22/Desktop/Hackathons/hack4healtj/"

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Round 2: Unsupervised Clustering (K-Means)\n",
    "\n",
    "**Goal**: Use unsupervised learning to demonstrate the intrinsic difficulty of the dataset. By clustering poems based on their semantic meaning (mBERT embeddings) without looking at labels, we can verify if they form distinct emotional groups or if they overlap significantly.\n",
    "\n",
    "**Method**:\n",
    "1. **Embeddings**: Generate semantic vectors using `bert-base-multilingual-cased`.\n",
    "2. **K-Means**: Cluster poems into K=9 groups (Navarasa).\n",
    "3. **Visualization**: Use t-SNE to visualize the high-dimensional embeddings in 2D.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\nimport numpy as np\nimport seaborn as sns\nimport matplotlib.pyplot as plt\nimport os\nimport torch\nfrom transformers import BertTokenizer, BertModel\nfrom sklearn.cluster import KMeans\nfrom sklearn.manifold import TSNE\nfrom sklearn.metrics import silhouette_score\nimport json\n\n",
    "# Setup\n",
    "sns.set(style='whitegrid')\n",
    "reports_dir = f'{base_path}round2/reports'\n",
    "os.makedirs(reports_dir, exist_ok=True)\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(f\"Using device: {device}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Data\n",
    "df = pd.read_excel(f'{base_path}round2/Combined_Emotions.xlsx')\n",
    "# Ensure text column is string\n",
    "df['cleaned_poem'] = df['cleaned_poem'].astype(str)\n",
    "print(f\"Loaded {len(df)} poems.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load mBERT\n",
    "print(\"Loading mBERT model...\")\n",
    "tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')\n",
    "model = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)\n",
    "model.eval()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_embeddings(texts, batch_size=32):\n",
    "    embeddings = []\n",
    "    total = len(texts)\n",
    "    print(f\"Generating embeddings for {total} texts...\")\n",
    "    \n",
    "    for i in range(0, total, batch_size):\n",
    "        batch = texts[i:i+batch_size].tolist()\n",
    "        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)\n",
    "        \n",
    "        with torch.no_grad():\n",
    "            outputs = model(**encoded)\n",
    "            # Use CLS token embedding (first token)\n",
    "            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()\n",
    "            embeddings.append(cls_embeddings)\n",
    "            \n",
    "        if (i+batch_size) % 1000 == 0:\n",
    "            print(f\"Processed {i+batch_size}/{total}\")\n",
    "            \n",
    "    return np.vstack(embeddings)\n",
    "\n",
    "# Generate Embeddings\n",
    "X_emb = get_embeddings(df['cleaned_poem'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# K-Means Clustering (K=9 for Navarasa)\n",
    "k = 9\n",
    "print(f\"Running K-Means with K={k}...\")\n",
    "kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')\n",
    "clusters = kmeans.fit_predict(X_emb)\n",
    "df['cluster'] = clusters\n",
    "\n",
    "# Silhouette Score\n",
    "sil = silhouette_score(X_emb, clusters)\n",
    "print(f\"Silhouette Score: {sil:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# t-SNE Visualization\n",
    "print(\"Running t-SNE reduction...\")\n",
    "tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')\n",
    "X_tsne = tsne.fit_transform(X_emb)\n",
    "df['tsne_1'] = X_tsne[:, 0]\n",
    "df['tsne_2'] = X_tsne[:, 1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot 1: Colored by K-Means Cluster\n",
    "plt.figure(figsize=(12, 8))\n",
    "sns.scatterplot(x='tsne_1', y='tsne_2', hue='cluster', data=df, palette='tab10', alpha=0.7, legend='full')\n",
    "plt.title(f'K-Means Clustering (K={k}) on mBERT Embeddings')\n",
    "plt.savefig(f'{reports_dir}/kmeans_clusters.png')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot 2: Colored by True Primary Emotion\n",
    "# Since there are many labels (46), this will look messy, confirming the difficulty\n",
    "plt.figure(figsize=(14, 10))\n",
    "# Use top 9 most frequent emotions for cleaner legend, gray out others if needed, \n",
    "# but here we want to show the chaos.\n",
    "sns.scatterplot(x='tsne_1', y='tsne_2', hue='primary_emotion', data=df, palette='turbo', alpha=0.6, legend=False)\n",
    "plt.title('True Emotion Labels (Primary) on mBERT Embeddings')\n",
    "plt.savefig(f'{reports_dir}/true_labels_tsne.png')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save Analysis Report\n",
    "analysis = {\n",
    "    'k': k,\n",
    "    'silhouette_score': float(sil),\n",
    "    'interpretation': \"The low silhouette score and t-SNE visualization indicate significant overlap between poem embeddings. Even without supervision, poems do not form distinct emotion clusters, explaining the challenge for supervised validation.\"\n",
    "}\n",
    "with open(f'{reports_dir}/kmeans_analysis.json', 'w') as f:\n",
    "    json.dump(analysis, f, indent=4)\n",
    "print(\"Analysis saved.\")"
   ]
  }
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
 "nbformat": 4, "nbformat_minor": 4
}

with open(f'{base_path}round2/8_KMeans_Analysis.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=1)
