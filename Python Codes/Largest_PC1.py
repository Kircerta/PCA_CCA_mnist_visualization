import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


def analyze_6_vs_9_stability(filepath='mnist.csv'):
    # 1. Load Data
    print("Loading Data...")
    try:
        df = pd.read_csv(filepath)
        if 'label' in str(df.columns[0]).lower():
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
        else:
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
    except:
        print("Error: csv not found.")
        return

    # Constants
    DIGITS = [6, 9]
    PCA_VARIANTS = [20, 50, 100]  # 我们测试三个不同的 PCA 维度
    CCA_COMPONENTS = 1  # 目前只看第一主成分，但要对比稳定性

    fig, axes = plt.subplots(len(DIGITS), len(PCA_VARIANTS), figsize=(15, 8))

    print(f"Comparing Digits {DIGITS} across PCA dimensions {PCA_VARIANTS}...")

    for row_idx, digit in enumerate(DIGITS):
        indices = np.where(y == digit)[0]
        # 采样最多 1000 个以保证速度，但足够统计显著
        if len(indices) > 1000: indices = indices[:1000]

        X_digit = X[indices]
        n = X_digit.shape[0]

        # Spatial Split
        X_img = X_digit.reshape(n, 28, 28)
        X_left = X_img[:, :, :14].reshape(n, -1)
        X_right = X_img[:, :, 14:].reshape(n, -1)

        # Standardize (Fixed)
        scaler_l = StandardScaler()
        scaler_r = StandardScaler()
        X_l_std = scaler_l.fit_transform(X_left)
        X_r_std = scaler_r.fit_transform(X_right)

        for col_idx, n_pca in enumerate(PCA_VARIANTS):
            ax = axes[row_idx, col_idx]

            # --- PCA Variation ---
            pca_l = PCA(n_components=n_pca, random_state=42)
            pca_r = PCA(n_components=n_pca, random_state=42)
            X_l_pca = pca_l.fit_transform(X_l_std)
            X_r_pca = pca_r.fit_transform(X_r_std)

            # --- CCA ---
            cca = CCA(n_components=1, max_iter=2000)
            cca.fit(X_l_pca, X_r_pca)

            # Get Correlation
            U, V = cca.transform(X_l_pca, X_r_pca)
            corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

            # Reconstruct Pattern
            w_l = cca.x_weights_[:, 0]
            w_r = cca.y_weights_[:, 0]
            pattern_l = np.dot(w_l, pca_l.components_).reshape(28, 14)
            pattern_r = np.dot(w_r, pca_r.components_).reshape(28, 14)

            # Stitch
            full_img = np.zeros((28, 30))
            full_img[:, :14] = pattern_l
            full_img[:, 16:] = pattern_r

            # Sign Flip Check (Heuristic)
            # 为了方便视觉对比，我们强制让图像中最强的点是红色的（正）
            # 这可以消除数学上的符号翻转带来的视觉干扰
            if np.abs(np.min(full_img)) > np.abs(np.max(full_img)):
                full_img = -full_img

            # Visualize
            limit = np.max(np.abs(full_img)) * 0.8
            ax.imshow(full_img, cmap='RdBu_r', vmin=-limit, vmax=limit)

            if row_idx == 0:
                ax.set_title(f"PCA Components: {n_pca}", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(f"Digit {digit}\n(n={n})", fontsize=14, fontweight='bold')

            ax.text(0.5, -0.1, f"Corr: {corr:.3f}", transform=ax.transAxes, ha='center')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Stability Analysis: Effect of PCA Dimension on Structural Dependency (6 vs 9)", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_6_vs_9_stability()