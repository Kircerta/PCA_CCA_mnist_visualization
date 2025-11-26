import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def visualize_6_vs_9_variance(filepath='mnist.csv'):
    # 1. 加载数据
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        # 自动判断是否有 header
        if 'label' in str(df.columns[0]).lower() or 'label' in str(df.columns).lower():
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
        else:
            y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. 提取 6 和 9 的所有样本
    indices_6 = np.where(y == 6)[0]
    indices_9 = np.where(y == 9)[0]

    # 3. 随机抽取 10 个样本 (如果样本不够则取全部)
    n_samples = 10
    samples_6 = X[np.random.choice(indices_6, n_samples, replace=False)]
    samples_9 = X[np.random.choice(indices_9, n_samples, replace=False)]

    # 4. 绘图设置
    fig, axes = plt.subplots(2, 10, figsize=(16, 4))
    plt.subplots_adjust(hspace=0.3)

    # 绘制第一行：数字 6
    for i in range(10):
        img = samples_6[i].reshape(28, 28)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Digit 6\n(Closed Loop)", loc='left', fontsize=14, color='blue', fontweight='bold')

    # 绘制第二行：数字 9
    for i in range(10):
        img = samples_9[i].reshape(28, 28)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Digit 9\n(Open/Free Tail)", loc='left', fontsize=14, color='red', fontweight='bold')

    # 添加整体标题
    plt.suptitle(f"Kinematic Constraints Check: '6' (Closed) vs '9' (Open)\nObserve the variance in the tails of '9'",
                 fontsize=16)

    plt.show()


# 运行
if __name__ == "__main__":
    visualize_6_vs_9_variance('mnist.csv')