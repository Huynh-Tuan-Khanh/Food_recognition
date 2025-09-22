import os
import cv2
import numpy as np
import pickle
import h5py
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# === Load dữ liệu ảnh ===
data_dir = r"Anh/data_mono"
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
        img = cv2.resize(img, (100, 100))
        images.append(img.flatten())
        labels.append(folder)

images = np.array(images) / 255.0
labels = np.array(labels)

# Mã hoá nhãn
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)
y = np.eye(num_classes)[labels_encoded]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)

# === ANN structure ===
input_size = X_train.shape[1]   # 100*100 = 10000
hidden_size = 128
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

# === Train loop ===
epochs = 13
lr = 0.01
for epoch in range(epochs):
    # Forward
    Z1 = X_train @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # Loss
    loss = -np.mean(np.sum(y_train * np.log(A2 + 1e-9), axis=1))

    # Backprop
    dZ2 = A2 - y_train
    dW2 = A1.T @ dZ2 / X_train.shape[0]
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = X_train.T @ dZ1 / X_train.shape[0]
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    # Accuracy train
    pred_train = np.argmax(A2, axis=1)
    acc_train = np.mean(pred_train == np.argmax(y_train, axis=1))
    print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Acc: {acc_train:.4f}")

# === Evaluate on test set ===
Z1 = X_test @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
A2 = softmax(Z2)
pred_test = np.argmax(A2, axis=1)
acc_test = np.mean(pred_test == np.argmax(y_test, axis=1))
print(f"✅ Test Accuracy: {acc_test*100:.2f}%")

# === Save ANN to .h5 ===
with h5py.File("face_recognition_ann.h5", "w") as f:
    f.create_dataset("W1", data=W1)
    f.create_dataset("b1", data=b1)
    f.create_dataset("W2", data=W2)
    f.create_dataset("b2", data=b2)

print("💾 Đã lưu model ANN thành vietnamese_food.h5")

# Save encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("💾 Đã lưu label encoder thành label_encoder.pkl")
