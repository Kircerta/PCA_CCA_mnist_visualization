import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import os

DATA_PATH = "mnist.csv"


def visualize_eigen_evidence(path):
    # 1. Load & Preprocess
    if not os.path.exists(path): return
    df = pd.read_csv(path)
    X = df.iloc[:, 1:].values

    # 必须保留 VarianceThreshold 对象以进行 inverse_transform
    selector = VarianceThreshold(threshold=0)
    X_filtered = selector.fit_transform(X)

    # Standardize
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_filtered)

    # 2. Fit PCA
    pca = PCA(n_components=3)
    pca.fit(X_std)

    # 3. Recover the Eigen-images (Loadings)
    # Inverse transform to map back to 784 pixels (including the dead ones)
    components_full = selector.inverse_transform(pca.components_)

    # 4. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Titles representing the "Hypothesis" we want to test
    titles = [
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} Var)\nExpectation: Ink Density",
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} Var)\nExpectation: Slant/Thickness",
        f"PC3 ({pca.explained_variance_ratio_[2]:.1%} Var)\nExpectation: Topology (The Loop?)"
    ]

    for i, ax in enumerate(axes):
        # Reshape to 28x28 image
        img = components_full[i].reshape(28, 28)

        # Plot heatmap
        im = ax.imshow(img, cmap='seismic', vmin=-0.15, vmax=0.15)
        ax.set_title(titles[i], fontsize=12)
        ax.axis('off')

        # Add contours to help see the shape
        ax.contour(img, levels=[0], colors='black', linewidths=0.5, alpha=0.5)

    plt.suptitle("Mathematical Evidence: The Top 3 Eigen-digits", fontsize=16)
    plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    plt.show()


if __name__ == "__main__":
    visualize_eigen_evidence(DATA_PATH)