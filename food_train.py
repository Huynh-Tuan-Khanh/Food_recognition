import os
import cv2
import numpy as np
import pickle
import h5py
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# === Load dữ liệu ảnh ===
data_dir = r"Train"
images, labels = [], []

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (128, 128))   # Resize đồng bộ
        images.append(img.flatten())
        labels.append(folder)

images = np.array(images, dtype=np.float32) / 255.0   # chuẩn hóa [0,1]
labels = np.array(labels)

# === Mã hoá nhãn ===
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)
y = np.eye(num_classes)[labels_encoded]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)

# === ANN structure ===
input_size = X_train.shape[1]   # 128*128 = 16384
hidden_size = 256
output_size = num_classes

np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

def relu(x): return np.maximum(0, x)
def relu_deriv(x): return (x > 0).astype(float)
def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# === Train loop với mini-batch ===
epochs = 40
batch_size = 64
lr = 0.01

for epoch in range(epochs):
    # Shuffle dữ liệu mỗi epoch
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward
        Z1 = X_batch @ W1 + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2 + b2
        A2 = softmax(Z2)

        # Backprop
        dZ2 = A2 - y_batch
        dW2 = A1.T @ dZ2 / X_batch.shape[0]
        db2 = np.mean(dZ2, axis=0, keepdims=True)

        dA1 = dZ2 @ W2.T
        dZ1 = dA1 * relu_deriv(Z1)
        dW1 = X_batch.T @ dZ1 / X_batch.shape[0]
        db1 = np.mean(dZ1, axis=0, keepdims=True)

        # Update
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

    # Accuracy train cuối mỗi epoch
    Z1 = X_train @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    pred_train = np.argmax(A2, axis=1)
    acc_train = np.mean(pred_train == np.argmax(y_train, axis=1))

    print(f"Epoch {epoch+1}/{epochs} - Acc: {acc_train:.4f}")

# === Evaluate on test set ===
Z1 = X_test @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
A2 = softmax(Z2)
pred_test = np.argmax(A2, axis=1)
acc_test = np.mean(pred_test == np.argmax(y_test, axis=1))
print(f"✅ Test Accuracy: {acc_test*100:.2f}%")

# === Save ANN to .h5 ===
with h5py.File("vietnamese_food.h5", "w") as f:
    f.create_dataset("W1", data=W1)
    f.create_dataset("b1", data=b1)
    f.create_dataset("W2", data=W2)
    f.create_dataset("b2", data=b2)

print("💾 Đã lưu model ANN thành vietnamese_food.h5")

# Save encoder
with open("label_encoder_food.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("💾 Đã lưu label encoder thành label_encoder_food.pkl")
