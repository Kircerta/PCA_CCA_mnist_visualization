import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_scree_analysis(filepath='mnist.csv', variance_threshold=0.90):
    print(f"Loading data for Scree Analysis from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        # 兼容性处理：寻找label列
        if 'label' in str(df.columns[0]).lower():
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
        else:
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
    except Exception as e:
        print(f"Error: {e}")
        return None

    digits = range(10)
    needed_components = []

    # 设置画布
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    print("\n--- Sample Count & Component Analysis ---")
    print(f"{'Digit':<6} | {'Count':<8} | {'PCs for ' + str(int(variance_threshold * 100)) + '% Var'}")
    print("-" * 45)

    for i, digit in enumerate(digits):
        # 1. 筛选数据
        X_digit = X[y == digit]
        count = X_digit.shape[0]

        # 2. 标准化 (重要: PCA对Scale敏感)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_digit)
        X_std = np.nan_to_num(X_std)  # 处理全0列导致的NaN

        # 3. Full PCA
        pca = PCA(n_components=100, svd_solver='full')  # 只看前100个通常足够做scree plot
        pca.fit(X_std)

        # 4. 计算累积方差
        cumsum = np.cumsum(pca.explained_variance_ratio_)

        # 5. 寻找阈值点 (Elbow point logic based on threshold)
        # argmax 返回第一个 True 的索引
        k = np.argmax(cumsum >= variance_threshold) + 1
        needed_components.append(k)

        print(f"{digit:<6} | {count:<8} | {k}")

        # 6. 绘图 Scree Plot
        ax = axes[i]
        ax.plot(cumsum, linewidth=2, color='blue')
        ax.axhline(y=variance_threshold, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=k, color='r', linestyle='--', alpha=0.5)
        ax.set_title(f"Digit {digit} (n={count})\nNeed k={k}", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if i >= 5: ax.set_xlabel("Number of Components")
        if i % 5 == 0: ax.set_ylabel("Cumulative Variance")

    # 推荐全局 K
    global_k_max = max(needed_components)
    global_k_mean = int(np.mean(needed_components))

    plt.suptitle(f"Scree Plots Analysis: Finding Optimal k for {int(variance_threshold * 100)}% Variance", fontsize=16)
    plt.tight_layout()
    plt.show()

    print("-" * 45)
    print(
        f"Recommendation: To satisfy {int(variance_threshold * 100)}% variance for ALL digits, use k = {global_k_max}")
    print(f"Conservative choice (Mean): k = {global_k_mean}")

    return global_k_max


if __name__ == "__main__":
    optimal_k = run_scree_analysis()