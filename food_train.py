import os
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

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
        images.append(img)
        labels.append(folder)

images = np.array(images, dtype=np.float32) / 255.0   # chuẩn hóa [0,1]
labels = np.array(labels)

# === Mã hoá nhãn ===
encoder = LabelEncoder()
labels_encoded = encoder.fit_transform(labels)
num_classes = len(encoder.classes_)
y = to_categorical(labels_encoded, num_classes=num_classes)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)

# === ANN với Keras Sequential ===
model = Sequential([
    Flatten(input_shape=(128, 128)),      # input: ảnh 128x128 -> vector 16384
    Dense(256, activation='relu'),
    Dropout(0.3),                         # thêm dropout để tránh overfitting
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# === Train ===
history = model.fit(X_train, y_train, 
                    epochs=40, 
                    batch_size=64,
                    validation_data=(X_test, y_test))

# === Evaluate ===
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {acc*100:.2f}%")

# === Save model & encoder ===
model.save("vietnamese_food.h5")
print("💾 Đã lưu model ANN thành vietnamese_food.h5")

with open("label_encoder_food.pkl", "wb") as f:
    pickle.dump(encoder, f)
print("💾 Đã lưu label encoder thành label_encoder_food.pkl")
