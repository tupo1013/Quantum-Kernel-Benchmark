import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_xor_data(n_samples=100, noise=0.1):
    data_set = np.random.rand(n_samples, 2) * 2 - 1
    label_set = np.logical_xor(data_set[:, 0] > 0, data_set[:, 1] > 0).astype(int)
    data_set += np.random.normal(0, noise, data_set.shape)
    return data_set, label_set

def engineer_split_set(data_set, label_set):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_set)
    data_train, data_test, label_train, label_test = train_test_split(
        data_scaled, label_set, test_size=0.3, random_state=42
    )
    return data_train, data_test, label_train, label_test

# X, y = generate_xor_data(n_samples=200, noise = 0.0)

# # Plot
# plt.figure(figsize=(6,6))
# plt.scatter(X[y==0, 0], X[y==0, 1], c="blue", label="Class 0", alpha=0.7, edgecolors="k")
# plt.scatter(X[y==1, 0], X[y==1, 1], c="red", label="Class 1", alpha=0.7, edgecolors="k")
# plt.title("XOR Dataset")
# plt.xlabel("X1")
# plt.ylabel("X2")
# plt.legend()
# plt.grid(True)
# plt.show()