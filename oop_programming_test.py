import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

DATA_PATH = "mnist.csv"
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
if not os.path.exists("plots"):
    os.makedirs("plots")

# Functions

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

def visualize_cca_geometry(X, y, n_components, mode='LR'):
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

def verify_variance_coverage(X, y, n_components):
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

def inspect_cca_weights_unified(X, y, target_digit, n_components, mode='TB'):
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

def inspect_cca_weights_numeric_output(X, y, target_digit, n_components):
    print(f"\n=== CCA Weight Inspection for Digit {target_digit} (Top-Bottom Split) ===")

    # --- 1. 数据准备 ---
    X_digit = X[y == target_digit]
    X_norm = X_digit / 255.0
    n_samples = X_norm.shape[0]

    IMG_SIZE = 28
    CUT_ROW = 14

    X_img = X_norm.reshape(n_samples, IMG_SIZE, IMG_SIZE)
    X_top = X_img[:, :CUT_ROW, :].reshape(n_samples, -1)
    X_bottom = X_img[:, CUT_ROW:, :].reshape(n_samples, -1)

    # --- 2. PCA 降维 ---
    pca_t = PCA(n_components=n_components, random_state=42)
    pca_b = PCA(n_components=n_components, random_state=42)

    X_t_pca = pca_t.fit_transform(X_top)
    X_b_pca = pca_b.fit_transform(X_bottom)

    # --- 3. CCA 核心 ---
    cca = CCA(n_components=1)
    cca.fit(X_t_pca, X_b_pca)

    # [!] 获取权重 [!]
    # x_weights_ 对应 X_top 的权重
    # y_weights_ 对应 X_bottom 的权重
    # shape is (n_features, n_components), we want the first component
    w_top = cca.x_weights_[:, 0]
    w_bottom = cca.y_weights_[:, 0]

    # --- 4. 打印数值表格 ---
    print("\n--- CCA Weight Distribution ---")
    df_weights = pd.DataFrame({
        "PC_Index": [f"PC{i + 1}" for i in range(n_components)],
        "Top_Weight": w_top,
        "Bottom_Weight": w_bottom,
        "Abs_Top_Strength": np.abs(w_top)  # 用于排序或观察绝对强度
    })

    # 打印原始数值
    print(df_weights.to_markdown(index=False, floatfmt=".4f"))

    # 找出那个“捣乱”的主导成分
    dominant_idx = np.argmax(np.abs(w_top))
    print(
        f"\n[Observation] The dominant component in Top half is PC{dominant_idx + 1} (Weight: {w_top[dominant_idx]:.4f})")

def visualize_pc_matrix_unified(X, y, target_digit, n_components):
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

    fig, axes = plt.subplots(4, n_components, figsize=(3 * n_components, 10))
    cmap = 'seismic'

    for row_idx, (name, data, shape) in enumerate(config):
        pca = PCA(n_components=n_components, random_state=42).fit(data)

        for col_idx in range(n_components):
            ax = axes[row_idx, col_idx]
            pc_img = pca.components_[col_idx].reshape(shape)
            v_max = np.max(np.abs(pc_img))

            ax.imshow(pc_img, cmap=cmap, vmin=-v_max, vmax=v_max)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0: ax.set_title(f"PC {col_idx + 1}", fontweight='bold')
            if col_idx == 0: ax.set_ylabel(name, fontsize=14, fontweight='bold', rotation=90)

    plt.suptitle(f"Geometry Modes: Digit {target_digit} (Top {n_components} PCs)", fontsize=16, y=0.96)
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

def display_top_bottom_split(X, y, target_digit):

    IMG_SIZE = 28
    CUT_ROW = 14  # 水平切割点：0-13行为上部，14-27行为下部

    # ---------------------------------------------------------
    # [SANITY CHECK] 视觉验证切片逻辑
    # ---------------------------------------------------------
    # 取出一个数字 0 的样本来看看
    sample_digit = X[y == target_digit][0].reshape(IMG_SIZE, IMG_SIZE) / 255.0

    # 模拟切片
    sample_top = sample_digit[:CUT_ROW, :]
    sample_bottom = sample_digit[CUT_ROW:, :]

    # 绘图验证
    fig_check, ax_check = plt.subplots(1, 3, figsize=(10, 3))
    ax_check[0].imshow(sample_digit, cmap='gray')
    ax_check[0].set_title("Original (Digit 0)")

    ax_check[1].imshow(sample_top, cmap='gray')
    ax_check[1].set_title("Top Slice (Rows 0-13)")

    ax_check[2].imshow(sample_bottom, cmap='gray')
    ax_check[2].set_title("Bottom Slice (Rows 14-27)")

    plt.suptitle(" ", fontsize=14, color='red')
    plt.show()
    # ---------------------------------------------------------

def visualize_target_digit_pc1_only(X, y, target_digit):

    X_digit = X[y == target_digit]
    X_norm = X_digit / 255.0
    n_samples = X_norm.shape[0]

    IMG_SIZE = 28
    CUT = 14

    # 还原为图像
    X_img = X_norm.reshape(n_samples, IMG_SIZE, IMG_SIZE)

    # 2. 准备四份数据
    # Top (Row 0-13)
    X_top = X_img[:, :CUT, :].reshape(n_samples, -1)
    # Bottom (Row 14-27)
    X_bottom = X_img[:, CUT:, :].reshape(n_samples, -1)
    # Left (Col 0-13)
    X_left = X_img[:, :, :CUT].reshape(n_samples, -1)
    # Right (Col 14-27)
    X_right = X_img[:, :, CUT:].reshape(n_samples, -1)

    # 3. 训练 4 个独立的 PCA (只取 PC1)
    pca_t = PCA(n_components=1, random_state=42).fit(X_top)
    pca_b = PCA(n_components=1, random_state=42).fit(X_bottom)
    pca_l = PCA(n_components=1, random_state=42).fit(X_left)
    pca_r = PCA(n_components=1, random_state=42).fit(X_right)

    # 4. 绘图 (1行 4列)
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    # 配置列表
    # (Title, Component, ReshapeSize)
    plots_config = [
        ("Top Split PC1\n(Translation?)", pca_t.components_[0], (14, 28)),
        ("Bottom Split PC1\n(Translation?)", pca_b.components_[0], (14, 28)),
        ("Left Split PC1\n(Shear/Tilt?)", pca_l.components_[0], (28, 14)),
        ("Right Split PC1\n(Shear/Tilt?)", pca_r.components_[0], (28, 14))
    ]

    cmap = 'seismic'

    for ax, (title, component, shape) in zip(axes, plots_config):
        # Reshape
        pc_img = component.reshape(shape)

        # 归一化显示范围，增强对比
        v_max = np.max(np.abs(pc_img))

        im = ax.imshow(pc_img, cmap=cmap, vmin=-v_max, vmax=v_max)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

        # 添加边框以示区分
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(1)

    # 添加 Colorbar
    cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    fig.colorbar(im, cax=cbar_ax, label='Pixel Weight (Red+, Blue-)')
    plt.subplots_adjust(wspace=0.3)
    plt.show()

def display_number(target_digit):

    df = pd.read_csv('mnist.csv')

    # Select 10 Random Sample of the Given Number, Change target number after "=="
    # igit_2_data = df[df.iloc[:, 0] == <Insert Your Number Here>]

    digit_2_data = df[df.iloc[:, 0] == target_digit]

    # 3. 随机抽取 10 个样本
    # random_state=42 保证每次抽取的样本一致，如果想看不同的可以去掉它
    samples = digit_2_data.sample(10, random_state=42)

    # 4. 可视化
    fig, axes = plt.subplots(1, 10, figsize=(15, 2)) # 创建 1行10列 的画布

    for i, (index, row) in enumerate(samples.iterrows()):
        # 提取像素数据：去掉第一列 label，剩下的部分转换为 numpy 数组
        pixel_values = row.iloc[1:].values

        # 将 Flattened (1, 784) 数据 Reshape 为 (28, 28)
        image_matrix = pixel_values.reshape(28, 28)

        # 绘图
        axes[i].imshow(image_matrix, cmap='gray') # 使用灰度图显示
        axes[i].axis('off') # 关闭坐标轴刻度
        axes[i].set_title(f"Sample {i+1}")

    plt.suptitle("Random Samples of Digit '2'", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_4way_scree(X, n_components):
    """
    绘制 Top/Bottom/Left/Right 四种切分方式的 PCA Scree Plot
    用于论证保留多少个主成分是合适的
    """
    print(f"\n--- Generating 4-Way Scree Plots (Top {n_components} PCs) ---")

    # 1. 获取四种切分数据 (使用全部数据或大样本)
    # 注意：这里我们传入归一化后的数据 / 255.0
    splits = get_image_splits(X / 255.0, mode='4Way')

    # 2. 设置绘图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    pc_nums = np.arange(1, n_components + 1)

    # 定义顺序和颜色
    config = [
        ('Top Half', splits['Top'], '#4c72b0'),
        ('Bottom Half', splits['Bottom'], '#dd8452'),
        ('Left Half', splits['Left'], '#55a868'),
        ('Right Half', splits['Right'], '#c44e52')
    ]

    for i, (name, X_part, color) in enumerate(config):
        # 运行 PCA
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_part)
        var_ratio = pca.explained_variance_ratio_

        # 绘图
        ax = axes[i]
        ax.plot(pc_nums, var_ratio, marker='o', linestyle='-', color=color, linewidth=2, markersize=6)

        # 标注前 3 个点的数值
        for j, val in enumerate(var_ratio[:3]):
            ax.text(pc_nums[j], val + 0.005, f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(f'{name} - Explained Variance', fontsize=14, fontweight='bold')
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Variance Ratio')
        ax.set_xticks(pc_nums[::2])
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(f'PCA Scree Plots: Variance Retention by Spatial Partition', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def plot_correlation_summary_bars(X, y, n_components):
    """
    计算并绘制 0-9 所有数字在 LR (左右) 和 UD (上下) 模式下的 CCA 相关性柱状图
    """
    print(f"\n--- Generating Correlation Summary Barplots (PCA k={n_components}) ---")

    results = []

    # 1. 遍历计算所有数字的相关性
    for digit in range(10):
        X_digit = X[y == digit] / 255.0

        # 避免样本过少
        if X_digit.shape[0] < n_components + 1:
            continue

        # 切分
        splits = get_image_splits(X_digit, mode='4Way')

        # 计算 LR 相关性
        res_lr = compute_pca_cca_pipeline(splits['Left'], splits['Right'], n_components)
        corr_lr = res_lr['corr'] if res_lr else 0

        # 计算 UD 相关性
        res_ud = compute_pca_cca_pipeline(splits['Top'], splits['Bottom'], n_components)
        corr_ud = res_ud['corr'] if res_ud else 0

        results.append({
            'Digit': digit,
            'LR_Correlation': corr_lr,
            'UD_Correlation': corr_ud
        })

    df_results = pd.DataFrame(results)

    # 2. 绘图 (两个子图)
    # 定义配置：(数据列, 标题, 颜色)
    plot_configs = [
        ('LR_Correlation', 'Left-Right Correlation', '#87CEEB'),  # SkyBlue
        ('UD_Correlation', 'Up-Down Correlation', '#F08080')  # LightCoral
    ]

    for col, title, color_code in plot_configs:
        plt.figure(figsize=(8, 4))

        # 绘制柱状图
        sns.barplot(x='Digit', y=col, data=df_results, color=color_code, alpha=0.9, edgecolor=".2")

        plt.title(f'{title} (PCA k={n_components})', fontweight='bold')
        plt.ylabel('Canonical Correlation')
        plt.ylim(0, 1.05)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # 在柱子上标数值
        for index, row in df_results.iterrows():
            plt.text(row.name, row[col] + 0.02, f'{row[col]:.2f}', color='black', ha="center", fontsize=10)

        plt.tight_layout()
        plt.show()

def run_full_analysis(n_components, target_digit):
    # Utility to Run Everything
    X, y = load_data_global(DATA_PATH, sample_size=5000)
    if X is None: return

    display_top_bottom_split(X, y, target_digit)

    visualize_target_digit_pc1_only(X, y, target_digit)

    # 2. Left/Right CCA Plot
    visualize_cca_geometry(X, y, n_components, mode='LR')

    # 3. Top/Bottom CCA Plot
    visualize_cca_geometry(X, y, n_components, mode='TB')

    # 4. Top/Bottom CCA Weight Barplot
    inspect_cca_weights_unified(X, y, target_digit, mode='TB')

    # 5. Left/Right CCA Weight Barplot
    inspect_cca_weights_unified(X, y, target_digit, mode='LR')

    # 6. Display PCs'Heatmap Visualizations of All Four Splits
    visualize_pc_matrix_unified(X, y, target_digit)

    # 7. Scree Plot and Variance Explained Output in Table

    verify_variance_coverage(X, y, n_components)
    plot_4way_scree(X, n_components=20)

    # 8. CCA weight for each number output
    inspect_cca_weights_numeric_output(X, y, target_digit, n_components)

    # 9. Correlation Barplot
    plot_correlation_summary_bars(X, y, n_components)

    # 10. Linear Regression to Restore Number
    run_hd_reconstruction_demo(X, y)

# --- RUN EVERYTHING ---

run_full_analysis(n_components = 5,target_digit = 2)