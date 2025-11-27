import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Read CSV file
df = pd.read_csv("mnist.csv")
# Select the first image
row = 10
label = int(df.loc[row, "label"])
vec = df.loc[row, df.columns[1:]].to_numpy(dtype=np.uint8)
# Convert to 28x28 matrix (row-major)
img = vec.reshape(28, 28)
# Display
plt.imshow(img, cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()