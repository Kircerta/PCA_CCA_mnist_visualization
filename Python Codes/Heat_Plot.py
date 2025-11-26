import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_theme(style="white")


def visualize_digit_structure(csv_path):
    try:
        # 1. 读取数据
        df = pd.read_csv(csv_path)
        y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values

        # ---------------------------------------------------------
        # 实验 A: "平均脸" 对比 (The Average Digit)
        # ---------------------------------------------------------
        # 既然 CCA 是基于整体统计特性的，看“平均值”最能反映该数字的通用结构
        digits_to_show = [0, 1, 2, 8]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i, digit in enumerate(digits_to_show):
            # 取出该数字的所有样本，计算像素平均值
            X_digit = X[y == digit]
            mean_img = X_digit.mean(axis=0).reshape(28, 28)

            axes[i].imshow(mean_img, cmap='jet')  # 使用热力图(jet)更能看清笔画分布强度
            axes[i].set_title(f"Average Digit '{digit}'")
            axes[i].axis('off')
        plt.suptitle("Experiment A: Average Pixel Intensity (Heatmap)", fontsize=16)
        plt.show()

        # ---------------------------------------------------------
        # 实验 B: 左右切分特写 (Left vs Right Split)
        # ---------------------------------------------------------
        # 重点对比 "1" (低相关) 和 "0" (高相关)
        target_digits = [1, 0]

        fig, axes = plt.subplots(2, 6, figsize=(12, 5))

        for row_idx, digit in enumerate(target_digits):
            # 随机选 2 个该数字的样本
            indices = np.random.choice(np.where(y == digit)[0], 2, replace=False)

            for col_idx, idx in enumerate(indices):
                img = X[idx].reshape(28, 28)

                # 左半边 (前14列)
                left_half = img[:, :14]
                # 右半边 (后14列)
                right_half = img[:, 14:]

                # 为了画图方便，补齐空白以便显示
                empty = np.zeros((28, 14))

                # 画完整图
                ax_orig = axes[row_idx, col_idx * 3]
                ax_orig.imshow(img, cmap='gray_r')
                ax_orig.set_title(f"Digit {digit} (Full)")
                ax_orig.axis('off')

                # 画左半边 (右边补黑)
                ax_left = axes[row_idx, col_idx * 3 + 1]
                ax_left.imshow(np.hstack((left_half, empty)), cmap='gray_r')
                ax_left.set_title("Left Part")
                ax_left.axis('off')

                # 画右半边 (左边补黑)
                ax_right = axes[row_idx, col_idx * 3 + 2]
                ax_right.imshow(np.hstack((empty, right_half)), cmap='gray_r')
                ax_right.set_title("Right Part")
                ax_right.axis('off')

        plt.suptitle("Experiment B: Visualizing the Split (Why '1' is different?)", fontsize=16)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Error: mnist.csv file not found.")


if __name__ == "__main__":
    visualize_digit_structure("mnist.csv")
