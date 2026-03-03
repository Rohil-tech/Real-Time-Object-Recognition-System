"""
Rohil Kulshreshtha
February 21, 2026
CS 5330 - PR-CV - Assignment 3

Visualize hand-crafted features in 2D using PCA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def visualize_handcrafted(
    obj_csv='data/object_database.csv',
    output_dir='visualization/results'
):
    os.makedirs(output_dir, exist_ok=True)

    # Load hand-crafted features
    df = pd.read_csv(obj_csv, dtype={'label': str})

    labels = df['label'].values
    feature_cols = ['percentFilled', 'aspectRatio', 'huMoment1', 'huMoment2']
    features = df[feature_cols].values

    unique_labels = sorted(set(labels))

    print(f"Loaded {len(labels)} samples")
    print(f"Objects: {unique_labels}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Features: {feature_cols}")

    # Standardize before PCA — important for hand-crafted features since
    # percentFilled, aspectRatio, and huMoments are on very different scales
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # PCA projection onto top 2 eigenvectors
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)

    pc1_var = pca.explained_variance_ratio_[0]
    pc2_var = pca.explained_variance_ratio_[1]
    print(f"\nPCA: PC1={pc1_var:.2%}, PC2={pc2_var:.2%}, Combined={pc1_var+pc2_var:.2%}\n")

    # Print feature loadings so we know what each PC represents
    print("=== Feature Loadings (eigenvectors) ===")
    for i, component in enumerate(pca.components_):
        print(f"PC{i+1}: " + ", ".join(
            f"{feat}={val:+.3f}" for feat, val in zip(feature_cols, component)
        ))

    # Plot
    plt.figure(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=label,
            s=150,
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )

    plt.xlabel(f'Principal Component 1 ({pc1_var:.1%} variance)', fontsize=14, fontweight='bold')
    plt.ylabel(f'Principal Component 2 ({pc2_var:.1%} variance)', fontsize=14, fontweight='bold')
    plt.title('2D Hand-Crafted Feature Visualization (PCA)', fontsize=16, fontweight='bold')
    plt.legend(title='Objects', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'handcrafted_visualization.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    print("\n=== Cluster Statistics ===")
    for label in unique_labels:
        mask = labels == label
        points = features_2d[mask]
        count = int(mask.sum())
        center = points.mean(axis=0)
        spread = points.std(axis=0).mean()
        print(f"{label:15}: {count:2d} samples, center=({center[0]:6.2f}, {center[1]:6.2f}), spread={spread:.3f}")

if __name__ == "__main__":
    visualize_handcrafted()