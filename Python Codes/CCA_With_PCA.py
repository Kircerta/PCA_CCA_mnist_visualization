import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import os

# Configuration
DATA_PATH = "mnist.csv"
# 设置绘图风格
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)


def compare_four_splits_with_pca(path):
    print(f"Loading data from {path}...")
    if not os.path.exists(path):
        print("Error: File not found. Please check DATA_PATH.")
        return

    # 1. 加载数据
    df = pd.read_csv(path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    n = X.shape[0]

    # Reshape
    X_img = X.reshape(n, 28, 28)

    # --- 构造对角线掩码 ---
    rows, cols = np.indices((28, 28))

    # Main Diagonal (\)
    mask_main_upper = (rows <= cols)
    mask_main_lower = (rows > cols)

    # Anti Diagonal (/)
    mask_anti_upper = (rows + cols <= 27)
    mask_anti_lower = (rows + cols > 27)

    X_main_upper = X_img[:, mask_main_upper]
    X_main_lower = X_img[:, mask_main_lower]
    X_anti_upper = X_img[:, mask_anti_upper]
    X_anti_lower = X_img[:, mask_anti_lower]

    # --- 定义 4 种切分 ---
    splits = {
        'L/R (Vertical)': (
            X_img[:, :, :14].reshape(n, -1),
            X_img[:, :, 14:].reshape(n, -1)
        ),
        'T/B (Horizontal)': (
            X_img[:, :14, :].reshape(n, -1),
            X_img[:, 14:, :].reshape(n, -1)
        ),
        'Main Diag (\\)': (
            X_main_upper,
            X_main_lower
        ),
        'Anti Diag (/)': (
            X_anti_upper,
            X_anti_lower
        )
    }

    results = {'Digit': [], 'Split Type': [], 'Correlation': [], 'Unexplained Variance': []}

    print("\n--- Running Global CCA (pre-processed by PCA-80) ---")

    for name, (view1_raw, view2_raw) in splits.items():
        print(f"Processing {name} split...")

        # Step A: 基础清理 (Zero Variance)
        sel1 = VarianceThreshold(0)
        sel2 = VarianceThreshold(0)

        try:
            v1 = sel1.fit_transform(view1_raw)
            v2 = sel2.fit_transform(view2_raw)
        except ValueError:
            print(f"Skipping {name}: Data constant or empty.")
            continue

        # Step B: 标准化 (Standardization)
        # PCA 对缩放非常敏感，必须先标准化
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        v1 = scaler1.fit_transform(v1)
        v2 = scaler2.fit_transform(v2)

        # Step C: PCA 降维 (Extract Top 80 PCs)
        # 我们对 View 1 和 View 2 分别做 PCA
        n_pca = 5

        # 安全检查：如果特征数本身少于80，就取特征数最大值
        n_c1 = min(n_pca, v1.shape[1])
        n_c2 = min(n_pca, v2.shape[1])

        pca1 = PCA(n_components=n_c1, random_state=42)
        pca2 = PCA(n_components=n_c2, random_state=42)

        v1_pca = pca1.fit_transform(v1)
        v2_pca = pca2.fit_transform(v2)

        # print(f"  > PCA shape: {v1_pca.shape} | {v2_pca.shape}")

        # Step D: Global CCA
        cca = CCA(n_components=1)
        try:
            U, V = cca.fit_transform(v1_pca, v2_pca)
        except Exception as e:
            print(f"CCA Failed for {name}: {e}")
            continue

        # Step E: Subgroup Analysis
        for d in range(10):
            idx = (y == d)
            if np.sum(idx) > 1:
                u_sub = U[idx, 0]
                v_sub = V[idx, 0]

                if np.std(u_sub) == 0 or np.std(v_sub) == 0:
                    corr = 0.0
                else:
                    corr = np.corrcoef(u_sub, v_sub)[0, 1]

                unexplained = 1 - corr ** 2

                results['Digit'].append(d)
                results['Split Type'].append(name)
                results['Correlation'].append(corr)
                results['Unexplained Variance'].append(unexplained)

    res_df = pd.DataFrame(results)

    # --- Visualization ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 16))

    # Plot 1: Correlation
    ax1 = sns.barplot(data=res_df, x='Digit', y='Correlation', hue='Split Type', ax=axes[0], palette="viridis")
    axes[0].set_title("Structural Redundancy (PCA-80 -> Global CCA): 4 Split Types")
    axes[0].set_ylim(0, 1.25)
    axes[0].legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    # 添加数值标签
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.3f', padding=5, fontsize=9, rotation=90)

    # Plot 2: Unexplained Variance
    ax2 = sns.barplot(data=res_df, x='Digit', y='Unexplained Variance', hue='Split Type', ax=axes[1], palette="rocket")
    axes[1].set_title("Structural Uncertainty (1 - r^2) with PCA Denoising")
    axes[1].set_ylabel("Unexplained Variance")
    axes[1].set_ylim(0, 1.25)
    axes[1].legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.3f', padding=5, fontsize=9, rotation=90)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_four_splits_with_pca(DATA_PATH)