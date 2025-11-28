import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# --- 1. 全局配置与基础工具 ---
DATA_PATH = "mnist.csv"
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
if not os.path.exists("plots"):
    os.makedirs("plots")


def load_data_global(path, sample_size=None):
    """
    统一的数据加载入口
    """
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None, None

    # 避免重复打印，只有第一次加载或耗时操作才打印
    df = pd.read_csv(path)

    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)

    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values  # 保持原始数值，后续再归一化

    print(f"Data Loaded. Shape: {X.shape}")
    return X, y


def get_image_splits(X_input, mode='LR', img_size=28):
    """
    【核心工具】统一处理图像切分逻辑
    mode: 'LR' (左右切), 'TB' (上下切), '4Way' (上下左右)
    """
    n_samples = X_input.shape[0]
    cut = img_size // 2
    X_img = X_input.reshape(n_samples, img_size, img_size)

    splits = {}

    if mode in ['LR', '4Way']:
        splits['Left'] = X_img[:, :, :cut].reshape(n_samples, -1)
        splits['Right'] = X_img[:, :, cut:].reshape(n_samples, -1)

    if mode in ['TB', '4Way']:
        splits['Top'] = X_img[:, :cut, :].reshape(n_samples, -1)
        splits['Bottom'] = X_img[:, cut:, :].reshape(n_samples, -1)

    return splits


def compute_pca_cca_pipeline(X1, X2, n_components):
    """
    【核心工具】统一执行 PCA -> CCA 流程
    返回: 权重, 相关系数, PCA模型, CCA模型
    """
    try:
        # 1. PCA
        pca1 = PCA(n_components=n_components, random_state=42)
        pca2 = PCA(n_components=n_components, random_state=42)
        X1_pca = pca1.fit_transform(X1)
        X2_pca = pca2.fit_transform(X2)

        # 2. CCA
        cca = CCA(n_components=1)
        cca.fit(X1_pca, X2_pca)
        U, V = cca.transform(X1_pca, X2_pca)
        corr = np.corrcoef(U[:, 0], V[:, 0])[0, 1]

        return {
            'corr': corr,
            'pca1': pca1, 'pca2': pca2,
            'cca': cca,
            'w1': cca.x_weights_[:, 0],  # 对应 X1 的权重
            'w2': cca.y_weights_[:, 0]  # 对应 X2 的权重
        }
    except Exception as e:
        print(f"Pipeline Error: {e}")
        return None


# --- 2. 具体的分析与绘图任务 ---

def visualize_cca_geometry(X, y, n_components=5, mode='LR'):
    """
    合并了原先的 visualize_pure_cca_mode (LR) 和 run_top_bottom_experiment (TB)
    只需指定 mode='LR' 或 mode='TB'
    """
    print(f"\n--- Running CCA Geometry Analysis (Mode: {mode}, PCs={n_components}) ---")

    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.flatten()
    cmap = 'seismic'

    # 确定切分的 Key
    key1, key2 = ('Left', 'Right') if mode == 'LR' else ('Top', 'Bottom')

    img_size = 28
    cut = 14

    for digit in range(10):
        # 1. 准备数据
        X_digit = X[y == digit] / 255.0
        splits = get_image_splits(X_digit, mode=mode)

        # 2. 运行计算管道
        res = compute_pca_cca_pipeline(splits[key1], splits[key2], n_components)

        cca_img = np.zeros((img_size, img_size))
        corr = 0

        if res:
            corr = res['corr']
            # 逆向投影 (Back-projection)
            # pattern = Weight * PCA_Components
            pat1 = np.dot(res['w1'], res['pca1'].components_)
            pat2 = np.dot(res['w2'], res['pca2'].components_)

            # 还原形状并拼接
            if mode == 'LR':
                cca_img[:, :cut] = pat1.reshape(28, 14)
                cca_img[:, cut:] = pat2.reshape(28, 14)
            else:  # TB
                cca_img[:cut, :] = pat1.reshape(14, 28)
                cca_img[cut:, :] = pat2.reshape(14, 28)

        # 3. 绘图
        ax = axes[digit]
        v_max = np.max(np.abs(cca_img)) if np.max(np.abs(cca_img)) > 0 else 1
        ax.imshow(cca_img, cmap=cmap, vmin=-v_max, vmax=v_max)
        ax.axis('off')
        ax.set_title(f"Digit {digit}\nρ = {corr:.3f}", fontsize=12, fontweight='bold')
        ax.text(0.5, -0.1, f"Mode: {mode}", transform=ax.transAxes, ha='center', fontsize=9, color='gray')

    plt.suptitle(f"CCA Geometry: {mode} Correlation Patterns (Top {n_components} PCs)", fontsize=16)
    plt.tight_layout()
    plt.show()


def verify_variance_coverage(X, y, n_components=15):
    """
    检查 PCA 解释的方差比例
    """
    print(f"\n--- Variance Verification (Top {n_components} PCs) ---")
    data = []

    for digit in range(10):
        X_digit = X[y == digit] / 255.0
        splits = get_image_splits(X_digit, mode='4Way')  # 一次切分得到4份

        row = {"Digit": digit}
        for key in ['Left', 'Right', 'Top', 'Bottom']:
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(splits[key])
            row[f"{key}_Var"] = np.sum(pca.explained_variance_ratio_)

        data.append(row)

    df = pd.DataFrame(data)
    print(df.set_index("Digit").applymap(lambda x: f"{x:.1%}").to_markdown())
    print(f"Average Variance (Left): {df['Left_Var'].mean():.1%}")


def inspect_cca_weights_unified(X, y, target_digit, n_components=5, mode='TB'):
    """
    合并了原先的 inspect_cca_weights (TB) 和 inspect_cca_weights_lr (LR)
    """
    print(f"\n=== CCA Weight Inspection: Digit {target_digit}, Mode {mode} ===")

    # 1. 数据与切分
    X_digit = X[y == target_digit] / 255.0
    splits = get_image_splits(X_digit, mode=mode)
    key1, key2 = ('Left', 'Right') if mode == 'LR' else ('Top', 'Bottom')

    # 2. 计算
    res = compute_pca_cca_pipeline(splits[key1], splits[key2], n_components)
    if not res: return

    w1, w2 = res['w1'], res['w2']

    # 3. 打印简报
    dom_idx1 = np.argmax(np.abs(w1))
    print(f"Dominant PC in {key1}: PC{dom_idx1 + 1} (Weight: {w1[dom_idx1]:.4f})")

    # 4. 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x_indices = np.arange(1, n_components + 1)

    for i, (weights, title, color_logic) in enumerate([
        (w1, f"{key1} Side Weights", True),
        (w2, f"{key2} Side Weights", True)
    ]):
        colors = ['red' if x >= 0 else 'blue' for x in weights]
        axes[i].bar(x_indices, weights, color=colors, alpha=0.7)
        axes[i].set_title(title, fontsize=14)
        axes[i].set_xticks(x_indices)
        axes[i].set_xticklabels([f"PC{k}" for k in x_indices])
        axes[i].grid(axis='y', linestyle='--')

    plt.suptitle(f"CCA Weight Distribution ({mode} Split)", fontsize=16)
    plt.tight_layout()
    plt.show()


def visualize_pc_matrix_unified(X, y, target_digit, n_comps=5):
    """
    展示某个数字的 Top/Bottom/Left/Right 前 n 个 PC 的特征图
    """
    print(f"\n=== Eigen-Image Matrix for Digit '{target_digit}' ===")

    X_digit = X[y == target_digit] / 255.0
    splits = get_image_splits(X_digit, mode='4Way')

    # 定义显示顺序和 reshape 形状
    config = [
        ('Top', splits['Top'], (14, 28)),
        ('Bottom', splits['Bottom'], (14, 28)),
        ('Left', splits['Left'], (28, 14)),
        ('Right', splits['Right'], (28, 14))
    ]

    fig, axes = plt.subplots(4, n_comps, figsize=(3 * n_comps, 10))
    cmap = 'seismic'

    for row_idx, (name, data, shape) in enumerate(config):
        pca = PCA(n_components=n_comps, random_state=42).fit(data)

        for col_idx in range(n_comps):
            ax = axes[row_idx, col_idx]
            pc_img = pca.components_[col_idx].reshape(shape)
            v_max = np.max(np.abs(pc_img))

            ax.imshow(pc_img, cmap=cmap, vmin=-v_max, vmax=v_max)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0: ax.set_title(f"PC {col_idx + 1}", fontweight='bold')
            if col_idx == 0: ax.set_ylabel(name, fontsize=14, fontweight='bold', rotation=90)

    plt.suptitle(f"Geometry Modes: Digit {target_digit} (Top {n_comps} PCs)", fontsize=16, y=0.96)
    plt.tight_layout()
    plt.show()


def run_hd_reconstruction_demo(X, y):
    """
    左半边 -> 右半边 线性回归还原演示
    """
    print(f"\n=== Running HD Reconstruction Demo ===")
    n_samples = X.shape[0]

    # 1. 切分 (使用 helper)
    splits = get_image_splits(X / 255.0, mode='LR')  # 注意这里传入归一化数据
    X_left, X_right = splits['Left'], splits['Right']

    # 2. 标准化 (Standardization for PCA/Regression)
    scaler_l, scaler_r = StandardScaler(), StandardScaler()
    X_l_std = scaler_l.fit_transform(X_left)
    X_r_std = scaler_r.fit_transform(X_right)

    # 3. PCA & Regression
    n_components = 5
    pca_l = PCA(n_components=n_components, random_state=42)
    pca_r = PCA(n_components=n_components, random_state=42)
    Z_l = pca_l.fit_transform(X_l_std)
    Z_r = pca_r.fit_transform(X_r_std)

    mapper = LinearRegression().fit(Z_l, Z_r)
    print(f"R^2 Score (Left->Right): {mapper.score(Z_l, Z_r):.4f}")

    # 4. 随机抽样还原
    idx = np.random.randint(0, n_samples)

    # 预测过程
    z_l_sample = Z_l[idx].reshape(1, -1)
    z_r_pred = mapper.predict(z_l_sample)
    x_r_rec = scaler_r.inverse_transform(pca_r.inverse_transform(z_r_pred))

    # 拼接显示
    img_orig = (X[idx] / 255.0).reshape(28, 28)
    img_rec_right = x_r_rec.reshape(28, 14)
    img_full_rec = np.hstack((img_orig[:, :14], img_rec_right))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img_orig, cmap='gray');
    ax[0].set_title(f"Original (Digit {y[idx]})")
    ax[1].imshow(img_full_rec, cmap='gray');
    ax[1].set_title("Reconstructed from Left")
    for a in ax: a.axis('off')
    plt.show()


# --- 3. 主流程调用函数 ---
def run_full_analysis(n_components, target_digit):
    # 1. 加载数据 (只做一次)
    X, y = load_data_global(DATA_PATH, sample_size=5000)
    if X is None: return

    # 2. 几何关联分析 (Left-Right) - 对应原 visualize_pure_cca_mode
    visualize_cca_geometry(X, y, n_components, mode='LR')

    # 3. 几何关联分析 (Top-Bottom) - 对应原 run_top_bottom_experiment
    visualize_cca_geometry(X, y, n_components, mode='TB')

    # 4. 权重检查 - 对应原 inspect_cca_weights
    inspect_cca_weights_unified(X, y, target_digit, mode='TB')

    inspect_cca_weights_unified(X, y, target_digit, mode='LR')

    # 5. 特征矩阵 - 对应原 visualize_pc_matrix...
    visualize_pc_matrix_unified(X, y, target_digit)

    # 6. 方差验证 - 对应原 verify_variance_coverage
    verify_variance_coverage(X, y, n_components)

    # 7. 还原演示
    run_hd_reconstruction_demo(X, y)

run_full_analysis(n_components = 5,target_digit = 2)