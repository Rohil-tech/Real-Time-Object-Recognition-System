"""
Rohil Kulshreshtha
February 21, 2026
CS 5330 - PR-CV - Assignment 3

Visualize embeddings in 2D using PCA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def visualize_embeddings(
    emb_csv='data/embedding_database.csv',
    obj_csv='data/object_database.csv',
    output_dir='visualization/results'
):
    os.makedirs(output_dir, exist_ok=True)

    # Load CNN embeddings (512-D ResNet18 features)
    # NOTE: The 'label' column in embedding_database.csv is corrupted (stores a
    # float instead of a string label). We load true string labels from
    # object_database.csv, which has matching row order.
    emb_df = pd.read_csv(emb_csv)
    obj_df = pd.read_csv(obj_csv, dtype={'label': str})

    labels = obj_df['label'].values
    embedding_cols = [col for col in emb_df.columns if col.startswith('emb')]
    embeddings = emb_df[embedding_cols].values

    assert len(labels) == len(embeddings), (
        f"Row count mismatch: embeddings={len(embeddings)}, labels={len(labels)}"
    )

    unique_labels = sorted(set(labels))

    print(f"Loaded {len(labels)} embeddings")
    print(f"Objects: {unique_labels}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # PCA projection onto top 2 eigenvectors
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    print(f"\nPCA: PC1={pca.explained_variance_ratio_[0]:.2%}, PC2={pca.explained_variance_ratio_[1]:.2%}\n")

    # Plot
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors[i]],
            label=label,
            s=150,
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )

    plt.xlabel('Principal Component 1', fontsize=14, fontweight='bold')
    plt.ylabel('Principal Component 2', fontsize=14, fontweight='bold')
    plt.title('2D Embedding Visualization (ResNet18)', fontsize=16, fontweight='bold')
    plt.legend(title='Objects', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'embedding_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

    print("\n=== Cluster Statistics ===")
    for label in unique_labels:
        mask = labels == label
        points = embeddings_2d[mask]
        count = int(mask.sum())
        center = points.mean(axis=0)
        spread = points.std(axis=0).mean()
        print(f"{label:15}: {count:2d} samples, center=({center[0]:6.1f}, {center[1]:6.1f}), spread={spread:4.1f}")

if __name__ == "__main__":
    visualize_embeddings()