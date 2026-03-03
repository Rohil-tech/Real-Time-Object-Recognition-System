"""
Rohil Kulshreshtha
February 21, 2026
CS 5330 - PR-CV - Assignment 3

Master script to run all visualizations in one go
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from visualize_embedding import visualize_embeddings
from visualize_handcrafted import visualize_handcrafted

if __name__ == "__main__":
    print("=" * 55)
    print("  CS 5330 - Assignment 3 - Embedding Visualizations")
    print("=" * 55)

    print("\n[1/2] CNN Embeddings (ResNet18) - PCA Visualization")
    print("-" * 55)
    visualize_embeddings()

    print("\n[2/2] Hand-Crafted Features - PCA Visualization")
    print("-" * 55)
    visualize_handcrafted()

    print("\n" + "=" * 55)
    print("  All visualizations complete!")
    print("  Results saved to: visualization/results/")
    print("=" * 55)