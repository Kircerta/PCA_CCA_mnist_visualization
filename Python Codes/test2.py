import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler


def analyze_local_mnist_geometry(filepath='mnist.csv'):
    # ---------------------------------------------------------
    # 1. Load Local Data
    # ---------------------------------------------------------
    print("----------------------------------------------------------------")
    print(f"1. Loading Local Data from {filepath}...")

    try:
        df = pd.read_csv(filepath)
        print(f"   File Loaded. Raw Shape: {df.shape}")

        # 自动识别 Label 列 (假设第一列是 label，或者列名包含 'label')
        # 这是一个鲁棒性检查，防止csv格式微小差异
        if 'label' in str(df.columns[0]).lower():
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
        else:
            # Fallback: 假设第一列就是 Label
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values

        print(f"   X Shape: {X.shape}, y Shape: {y.shape}")

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found in current directory.")
        return
    except Exception as e:
        print(f"Error processing data: {e}")
        return

    # ---------------------------------------------------------
    # 2. Configuration & Setup
    # ---------------------------------------------------------
    TARGET_DIGITS = range(10)  # Analyze 0 to 9
    IMG_SIZE = 28
    CUT_COL = 14  # Split at column 14
    PCA_N = 30  # Reduced dimension

    # Visualization Grid: 2 rows x 5 columns
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    # ---------------------------------------------------------
    # 3. Statistical Analysis Loop
    # ---------------------------------------------------------
    print("\n----------------------------------------------------------------")
    print("2. Running PCA-CCA Pipeline (Local Data)")
    print(
        f"{'Digit':<6} | {'N_Samples':<10} | {'PCA_Var_L':<10} | {'PCA_Var_R':<10} | {'CCA_Corr':<10} | {'Status':<10}")
    print("-" * 75)

    for i, digit in enumerate(TARGET_DIGITS):
        ax = axes[i]

        # --- A. Filter Data ---
        indices = np.where(y == digit)[0]
        X_digit = X[indices]
        n_samples = X_digit.shape[0]

        # Safety Check: 如果样本量太少，无法做 50 维的 PCA
        if n_samples < PCA_N + 5:
            print(f"{digit:<6} | {n_samples:<10} | {'FAIL':<10} | {'Insufficient Samples'}")
            ax.text(0.5, 0.5, "Insufficient Data", ha='center', va='center')
            ax.axis('off')
            continue

        # --- B. Spatial Splitting ---
        # Reshape to (N, 28, 28) then split
        # 注意：这里假设 CSV 是 flatten 过的 (N, 784)
        X_img = X_digit.reshape(n_samples, IMG_SIZE, IMG_SIZE)
        X_left = X_img[:, :, :CUT_COL].reshape(n_samples, -1)  # Left Half
        X_right = X_img[:, :, CUT_COL:].reshape(n_samples, -1)  # Right Half

        # --- C. Preprocessing (Standardize -> PCA) ---
        scaler_l = StandardScaler()
        scaler_r = StandardScaler()
        # Fit transform
        X_l_std = scaler_l.fit_transform(X_left)
        X_r_std = scaler_r.fit_transform(X_right)

        pca_l = PCA(n_components=PCA_N, random_state=42)
        pca_r = PCA(n_components=PCA_N, random_state=42)

        X_l_pca = pca_l.fit_transform(X_l_std)
        X_r_pca = pca_r.fit_transform(X_r_std)

        # Explained Variance (Diagnosis)
        var_l = np.sum(pca_l.explained_variance_ratio_)
        var_r = np.sum(pca_r.explained_variance_ratio_)

        # --- D. CCA (Canonical Correlation) ---
        cca = CCA(n_components=1)
        try:
            cca.fit(X_l_pca, X_r_pca)

            # 1. Calculate Correlation
            U, V = cca.transform(X_l_pca, X_r_pca)
            corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

            # 2. Extract Weights & Inverse Project
            w_l_pca = cca.x_weights_[:, 0]
            w_r_pca = cca.y_weights_[:, 0]

            # Project: (1, 50) @ (50, 392) -> (1, 392)
            pattern_l = np.dot(w_l_pca, pca_l.components_).reshape(28, 14)
            pattern_r = np.dot(w_r_pca, pca_r.components_).reshape(28, 14)

            # --- E. Print Stats ---
            print(f"{digit:<6} | {n_samples:<10} | {var_l:.4f}     | {var_r:.4f}     | {corr:.4f}     | {'OK'}")

            # --- F. Visualize ---
            # Stitch image
            full_img = np.zeros((28, 30))
            full_img[:, :14] = pattern_l
            full_img[:, 16:] = pattern_r  # Gap

            # Normalize for visualization (centered at 0)
            v_max = np.max(np.abs(full_img)) * 0.8

            ax.imshow(full_img, cmap='RdBu_r', vmin=-v_max, vmax=v_max)
            ax.set_title(f"Digit {digit} ($r={corr:.3f}$)", fontsize=11, fontweight='bold')
            ax.axis('off')

        except Exception as e:
            print(f"{digit:<6} | {n_samples:<10} | Error: {e}")
            ax.set_title(f"Error Digit {digit}")

    plt.suptitle("CCA Dependency Maps (Local Data, n=5000)\nRed = Positive Coupling, Blue = Negative Coupling",
                 fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    analyze_local_mnist_geometry()