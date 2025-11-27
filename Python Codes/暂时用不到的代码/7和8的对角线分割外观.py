import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
import os


# 1. 加载数据
def load_data():
    # 尝试加载本地文件，如果不行则下载
    # 为了演示方便，这里直接写你常用的逻辑
    path = "mnist.csv"
    if os.path.exists(path):
        print(f"Loading from {path}...")
        df = pd.read_csv(path)
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
    else:
        print("Local file not found, fetching from OpenML...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
        X = X[:5000]  # 只取前5000个加速
        y = y[:5000].astype(int)

    return X, y


def visualize_diagonal_anatomy(X, y):
    # 筛选出 6 和 9
    digits_of_interest = [7, 0]

    # 定义对角线 Mask (左上 -> 右下)
    # Row index (y轴) 从 0(上) 到 27(下)
    # Col index (x轴) 从 0(左) 到 27(右)
    rows, cols = np.indices((28, 28))

    # Upper Triangle: Row <= Col (右上部分，包含对角线)
    mask_upper = (rows <= cols)
    # Lower Triangle: Row > Col (左下部分)
    mask_lower = (rows > cols)

    plt.figure(figsize=(12, 8))

    # 为每个数字选 5 个样本展示
    samples_per_digit = 5

    for i, digit in enumerate(digits_of_interest):
        # 获取该数字的所有样本
        digit_samples = X[y == digit]

        # 随机选几个
        # np.random.seed(42)
        indices = np.random.choice(len(digit_samples), samples_per_digit, replace=False)
        selected_imgs = digit_samples[indices].reshape(-1, 28, 28)

        for j in range(samples_per_digit):
            img = selected_imgs[j]

            # 制造切割后的图像 (视觉上置零)
            img_upper = img.copy()
            img_upper[~mask_upper] = 0  # 把非上三角的部分变黑

            img_lower = img.copy()
            img_lower[~mask_lower] = 0  # 把非下三角的部分变黑

            # --- 绘图 ---
            # 行：每个样本一行
            # 列：原图 | 上三角(右上) | 下三角(左下)

            # 计算子图位置
            # i*samples*3 是大组的偏移，j*3 是小组的偏移
            base_idx = i * (samples_per_digit * 3) + j * 3 + 1

            # 1. Original
            plt.subplot(len(digits_of_interest) * samples_per_digit, 3, base_idx)
            plt.imshow(img, cmap='gray_r')  # 反色，白底黑字更好看清楚结构
            plt.axis('off')
            if j == 0:
                plt.title(f"Digit {digit}: Original", fontsize=10, fontweight='bold')

            # 2. Upper Triangle (右上)
            plt.subplot(len(digits_of_interest) * samples_per_digit, 3, base_idx + 1)
            plt.imshow(img_upper, cmap='Reds')  # 用红色显示切片
            # 画出对角线辅助视线
            plt.plot([0, 27], [0, 27], 'k--', linewidth=0.5)
            plt.axis('off')
            if j == 0:
                plt.title("Upper/Right Partition", fontsize=10)

            # 3. Lower Triangle (左下)
            plt.subplot(len(digits_of_interest) * samples_per_digit, 3, base_idx + 2)
            plt.imshow(img_lower, cmap='Blues')  # 用蓝色显示切片
            plt.plot([0, 27], [0, 27], 'k--', linewidth=0.5)
            plt.axis('off')
            if j == 0:
                plt.title("Lower/Left Partition", fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y = load_data()
    if X is not None:
        visualize_diagonal_anatomy(X, y)