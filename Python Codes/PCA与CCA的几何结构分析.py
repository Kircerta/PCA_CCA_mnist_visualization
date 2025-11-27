import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


def visualize_digit_geometry(filepath='mnist.csv', n_components=30):
    """
    n_components: 来自上一步 Scree Plot 的建议值 (例如 30)
    """
    print("----------------------------------------------------------------")
    print(f"RUNNING GEOMETRY ANALYSIS with k={n_components}")
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

    # 每一行一个数字，3列分别是 PC1, PC2, CCA-Map
    fig, axes = plt.subplots(10, 3, figsize=(9, 25))
    plt.subplots_adjust(hspace=0.4, wspace=0.1)

    # 设置列标题
    cols = ["Global PC1\n(Mean Shape)", "Global PC2\n(Primary Variation)", f"CCA Pair 1\n(L-R Correlation)"]
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=12, fontweight='bold')

    for i, digit in enumerate(TARGET_DIGITS):
        # --- 数据准备 ---
        X_digit = X[y == digit]
        n_samples = X_digit.shape[0]

        # 1. 全局 PCA (Whole Image Analysis)
        # 用来观察在这个数字里，最主要的变量是“写法倾斜”、“粗细”还是“圈的大小”
        scaler_whole = StandardScaler()
        X_whole_std = np.nan_to_num(scaler_whole.fit_transform(X_digit))

        pca_whole = PCA(n_components=5, svd_solver='full')
        pca_whole.fit(X_whole_std)

        whole_pc1 = pca_whole.components_[0].reshape(IMG_SIZE, IMG_SIZE)
        whole_pc2 = pca_whole.components_[1].reshape(IMG_SIZE, IMG_SIZE)
        var_pc1 = pca_whole.explained_variance_ratio_[0]
        var_pc2 = pca_whole.explained_variance_ratio_[1]

        # 2. 分割 CCA (Split Analysis)
        # 现将图像reshape并分割
        X_img = X_digit.reshape(n_samples, IMG_SIZE, IMG_SIZE)
        X_left = X_img[:, :, :CUT_COL].reshape(n_samples, -1)
        X_right = X_img[:, :, CUT_COL:].reshape(n_samples, -1)

        # 对左右分别标准化 + PCA降维
        scaler_l = StandardScaler()
        scaler_r = StandardScaler()
        X_l_std = np.nan_to_num(scaler_l.fit_transform(X_left))
        X_r_std = np.nan_to_num(scaler_r.fit_transform(X_right))

        pca_l = PCA(n_components=n_components, svd_solver='full', random_state=42)
        pca_r = PCA(n_components=n_components, svd_solver='full', random_state=42)

        X_l_pca = pca_l.fit_transform(X_l_std)
        X_r_pca = pca_r.fit_transform(X_r_std)

        # 运行 CCA
        cca = CCA(n_components=1)
        try:
            cca.fit(X_l_pca, X_r_pca)

            # 计算 Canonical Correlation Coefficient (r)
            U, V = cca.transform(X_l_pca, X_r_pca)
            corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

            # 逆向投影权重以可视化
            w_l = cca.x_weights_[:, 0]
            w_r = cca.y_weights_[:, 0]
            pattern_l = np.dot(w_l, pca_l.components_).reshape(28, 14)
            pattern_r = np.dot(w_r, pca_r.components_).reshape(28, 14)

            # 拼接 CCA Map
            cca_img = np.zeros((28, 28))
            cca_img[:, :14] = pattern_l
            cca_img[:, 14:] = pattern_r

        except Exception as e:
            print(f"CCA Error for {digit}: {e}")
            corr = 0
            cca_img = np.zeros((28, 28))

        # --- 绘图逻辑 ---
        def plot_heatmap(ax, data, title_text=None, show_val=None):
            v_max = np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else 1
            im = ax.imshow(data, cmap='RdBu_r', vmin=-v_max, vmax=v_max)
            ax.axis('off')
            if show_val:
                ax.text(0, 26, show_val, color='black', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, pad=1))
            if title_text and i == 0:  # 仅在第一行额外强调
                pass

                # 左图：Whole PC1

        plot_heatmap(axes[i, 0], whole_pc1, show_val=f"Var: {var_pc1:.1%}")
        axes[i, 0].text(-5, 14, str(digit), fontsize=14, fontweight='bold', va='center')  # Digit Label

        # 中图：Whole PC2
        plot_heatmap(axes[i, 1], whole_pc2, show_val=f"Var: {var_pc2:.1%}")

        # 右图：CCA Pattern
        plot_heatmap(axes[i, 2], cca_img, show_val=f"ρ: {corr:.2f}")

    plt.suptitle(f"MNIST Geometry: Whole PCA vs Split CCA (k={n_components})", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Leave space for title
    plt.show()


if __name__ == "__main__":
    # 这里假设 Step 1 算出来的是 30，你可以手动修改这个值
    visualize_digit_geometry(n_components=5)