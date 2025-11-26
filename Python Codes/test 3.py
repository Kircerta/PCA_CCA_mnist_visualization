import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


def compare_whole_pca_vs_cca_fixed(filepath='mnist.csv'):
    print("----------------------------------------------------------------")
    print("COMPARING WHOLE DIGIT PCA vs. SPLIT CCA (ROBUST VERSION)")
    print("----------------------------------------------------------------")

    try:
        df = pd.read_csv(filepath)
        if 'label' in str(df.columns[0]).lower():
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
        else:
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
    except Exception as e:
        print(f"Error: {e}")
        return

    TARGET_DIGITS = range(10)
    IMG_SIZE = 28
    CUT_COL = 14
    PCA_N_LOCAL = 30

    fig, axes = plt.subplots(10, 3, figsize=(12, 25))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    axes[0, 0].set_title("Whole PCA: PC1\n(The 'Body')", fontsize=12, fontweight='bold')
    axes[0, 1].set_title("Whole PCA: PC2\n(The 'Style/Tilt')", fontsize=12, fontweight='bold')
    axes[0, 2].set_title("CCA Pattern\n(The 'Bridge')", fontsize=12, fontweight='bold')

    for i, digit in enumerate(TARGET_DIGITS):
        print(f"Processing Digit {digit}...")

        # --- A. Data Prep ---
        indices = np.where(y == digit)[0]
        X_digit = X[indices]
        n_samples = X_digit.shape[0]

        if n_samples < 50:
            print(f"  Skipping {digit} (Insufficient samples)")
            continue

        # --- B. 计算 Whole PCA (Fix applied here) ---
        scaler_whole = StandardScaler()
        X_whole_std = scaler_whole.fit_transform(X_digit)

        # FIX 1: 清洗 NaN (将除以0产生的 NaN 变回 0)
        X_whole_std = np.nan_to_num(X_whole_std)

        # FIX 2: 使用 'full' solver 避免随机算法在数值不稳定时崩溃
        pca_whole = PCA(n_components=5, svd_solver='full')
        pca_whole.fit(X_whole_std)

        whole_pc1 = pca_whole.components_[0].reshape(28, 28)
        whole_pc2 = pca_whole.components_[1].reshape(28, 28)

        # --- C. 计算 CCA ---
        X_img = X_digit.reshape(n_samples, IMG_SIZE, IMG_SIZE)
        X_left = X_img[:, :, :CUT_COL].reshape(n_samples, -1)
        X_right = X_img[:, :, CUT_COL:].reshape(n_samples, -1)

        scaler_l = StandardScaler()
        scaler_r = StandardScaler()
        X_l_std = scaler_l.fit_transform(X_left)
        X_r_std = scaler_r.fit_transform(X_right)

        # FIX 1: 清洗 NaN
        X_l_std = np.nan_to_num(X_l_std)
        X_r_std = np.nan_to_num(X_r_std)

        # FIX 2: 使用 'full' solver
        pca_l = PCA(n_components=PCA_N_LOCAL, svd_solver='full', random_state=42)
        pca_r = PCA(n_components=PCA_N_LOCAL, svd_solver='full', random_state=42)

        X_l_pca = pca_l.fit_transform(X_l_std)
        X_r_pca = pca_r.fit_transform(X_r_std)

        cca = CCA(n_components=1)
        try:
            cca.fit(X_l_pca, X_r_pca)

            w_l = cca.x_weights_[:, 0]
            w_r = cca.y_weights_[:, 0]
            pattern_l = np.dot(w_l, pca_l.components_).reshape(28, 14)
            pattern_r = np.dot(w_r, pca_r.components_).reshape(28, 14)

            cca_img = np.zeros((28, 28))
            cca_img[:, :14] = pattern_l
            cca_img[:, 14:] = pattern_r
        except Exception as e:
            print(f"  CCA Error for digit {digit}: {e}")
            cca_img = np.zeros((28, 28))  # Fallback if CCA fails

        # --- D. Visualization ---
        def plot_heatmap(ax, data, label=None):
            # 防止全0数据导致的绘图错误
            if np.all(data == 0):
                v_max = 1
            else:
                v_max = np.max(np.abs(data))

            ax.imshow(data, cmap='RdBu_r', vmin=-v_max, vmax=v_max)
            if label:
                ax.text(-5, 14, label, fontsize=14, fontweight='bold', va='center')
            ax.axis('off')

        plot_heatmap(axes[i, 0], whole_pc1, label=str(digit))
        plot_heatmap(axes[i, 1], whole_pc2)
        plot_heatmap(axes[i, 2], cca_img)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_whole_pca_vs_cca_fixed()