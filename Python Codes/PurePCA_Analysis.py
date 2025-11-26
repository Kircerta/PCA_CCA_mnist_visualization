import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import os

# Configuration
DATA_PATH = "mnist.csv"
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def load_data(path):
    """
    Loads MNIST data.
    Expects format: label, pixel0, pixel1, ...
    """
    if not os.path.exists(path):
        print(f"Error: {path} not found. Please ensure the CSV file is in the working directory.")
        # Fallback for testing if file missing (Optional, using sklearn)
        # from sklearn.datasets import fetch_openml
        # print("Fetching MNIST from OpenML (fallback)...")
        # mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        # return mnist.data[:10000], mnist.target[:10000].astype(int) # Sample for speed
        return None, None

    df = pd.read_csv(path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    print(f"Data Loaded: N={X.shape[0]}, D={X.shape[1]}")
    return X, y


def perform_pca_analysis(X, y):
    print("\n--- Starting PCA Analysis ---")

    # 1. Critical Step: Remove Zero-Variance Pixels
    # Many pixels in MNIST are always black. They add no information but computational cost.
    selector = VarianceThreshold(threshold=0)
    X_filtered = selector.fit_transform(X)
    print(f"PCA Preprocessing: Feature space reduced from {X.shape[1]} to {X_filtered.shape[1]} (Removed dead pixels)")

    # 2. Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # 3. Fit PCA
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(X_scaled)

    # --- Visualization 1: Scree Plot ---
    plt.figure(figsize=(10, 6))
    var_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, 51), var_ratio, 'b-o', markersize=4, linewidth=1.5)

    # Add an annotation for 90% variance if reached
    idx_90 = np.argmax(var_ratio >= 0.9)
    if var_ratio[idx_90] >= 0.9:
        plt.axvline(x=idx_90 + 1, color='r', linestyle='--', alpha=0.7)
        plt.text(idx_90 + 3, 0.88, f'90% Variance @ {idx_90 + 1} Comps', color='r')

    plt.title('PCA Scree Plot: Cumulative Variance Explained')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance')
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- Visualization 2: 2D Projection ---
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.6, s=10)
    plt.colorbar(scatter, label='Digit Class')
    plt.title('PCA 2D Projection (PC1 vs PC2)')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.show()

    # --- Visualization 3: Eigen-digits ---
    # We need to map back to 28x28.
    # Since we removed pixels, we need to inverse transform the components to visualize them correctly.
    full_components = selector.inverse_transform(pca.components_)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i, ax in enumerate(axes.flat):
        # Reshape to image
        img = full_components[i].reshape(28, 28)
        im = ax.imshow(img, cmap='seismic', vmin=-0.15, vmax=0.15)
        ax.set_title(f"PC {i + 1}")
        ax.axis('off')
    plt.suptitle("Eigen-digits (Top 10 Principal Components)")
    plt.show()


def perform_cca_analysis(X, y):
    print("\n--- Starting CCA Analysis (Left vs. Right Split) ---")

    # 1. Split Data (Geometric Split)
    n = X.shape[0]
    # Reshape to (N, 28, 28) then split
    X_img = X.reshape(n, 28, 28)
    X_left_raw = X_img[:, :, :14].reshape(n, -1)
    X_right_raw = X_img[:, :, 14:].reshape(n, -1)

    # 2. Critical Step: Remove Zero-Variance Pixels INDEPENDENTLY per view
    # This prevents singularity in CCA's covariance inversion
    sel_l = VarianceThreshold(threshold=0)
    sel_r = VarianceThreshold(threshold=0)

    X_left = sel_l.fit_transform(X_left_raw)
    X_right = sel_r.fit_transform(X_right_raw)

    print(f"Left View Features: {X_left_raw.shape[1]} -> {X_left.shape[1]}")
    print(f"Right View Features: {X_right_raw.shape[1]} -> {X_right.shape[1]}")

    # 3. Standardize
    X_l_std = StandardScaler().fit_transform(X_left)
    X_r_std = StandardScaler().fit_transform(X_right)

    # 4. Run CCA (Global Fit)
    # We fit on all data to find the general "bilateral dependency" axis
    cca = CCA(n_components=1)
    U, V = cca.fit_transform(X_l_std, X_r_std)

    # 5. Calculate Correlations by Digit Class
    corrs = []
    digits = range(10)

    print("\nClass-Specific Canonical Correlations:")
    print("-" * 40)
    print(f"{'Digit':<10} | {'Correlation':<15}")
    print("-" * 40)

    for d in digits:
        idx = (y == d)
        if np.sum(idx) > 1:  # Need at least 2 samples for correlation
            # Calculate Pearson correlation of canonical variates for this class
            c = np.corrcoef(U[idx, 0], V[idx, 0])[0, 1]
            corrs.append(c)
            print(f"{d:<10} | {c:.4f}")
        else:
            corrs.append(0)

    # 6. Visualization
    plt.figure(figsize=(10, 6))
    # Use a color palette that highlights lower correlations
    colors = ['#d62728' if x < 0.6 else 'steelblue' for x in corrs]

    bars = plt.bar(digits, corrs, color=colors, alpha=0.8)
    plt.axhline(y=np.mean(corrs), color='gray', linestyle='--', label=f'Avg Corr ({np.mean(corrs):.2f})')

    plt.title('Statistical Dependency: Left vs Right Half (CCA)')
    plt.xlabel('Digit Class')
    plt.ylabel('Canonical Correlation Coefficient')
    plt.ylim(0, 1.1)
    plt.xticks(digits)
    plt.legend()

    # Add value labels
    for bar, v in zip(bars, corrs):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}",
                 ha='center', va='bottom', fontweight='bold')

    plt.show()

    # Optional: Scatter plot for Digit 1 vs Digit 0 (To visualize the "Sparsity" argument)
    # Plotting the Canonical Variates U vs V for Digit 0 and 1
    plt.figure(figsize=(8, 8))
    for d in [0, 1]:
        idx = (y == d)
        plt.scatter(U[idx, 0], V[idx, 0], label=f'Digit {d}', alpha=0.5, s=10)
    plt.xlabel('Left View Variate (U)')
    plt.ylabel('Right View Variate (V)')
    plt.title('CCA Latent Space: Digit 0 vs Digit 1')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    X, y = load_data(DATA_PATH)
    if X is not None:
        perform_pca_analysis(X, y)
        perform_cca_analysis(X, y)