import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import h5py
import pickle

# ==== Load model ANN đã train (2 lớp) ====
with h5py.File("vietnamese_food.h5", "r") as f:
    print("Keys trong file h5:", list(f.keys()))  # Debug
    W1 = f["W1"][:]; b1 = f["b1"][:]
    W2 = f["W2"][:]; b2 = f["b2"][:]

with open("label_encoder_food.pkl", "rb") as f:
    encoder = pickle.load(f)
    print("Classes:", encoder.classes_)  # Debug

# Hàm dự đoán
def predict_food(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (100, 100)) / 255.0   # resize giống lúc train
    x = img.flatten().reshape(1, -1)            # vector 1x10000

    # Forward ANN 2 lớp
    Z1 = x @ W1 + b1
    A1 = np.maximum(0, Z1)  # ReLU
    Z2 = A1 @ W2 + b2

    exp = np.exp(Z2 - np.max(Z2, axis=1, keepdims=True))
    A2 = exp / np.sum(exp, axis=1, keepdims=True)

    pred = np.argmax(A2, axis=1)[0]
    prob = np.max(A2)

    print("Predict index:", pred, "Prob:", prob)  # Debug
    return encoder.inverse_transform([pred])[0], prob

# ==== GUI với Tkinter ====
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.png;*.jpeg")]
    )
    if not file_path:
        return

    # Hiển thị ảnh
    img = Image.open(file_path)
    img = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    panel.config(image=img_tk)
    panel.image = img_tk

    # Dự đoán
    try:
        label, prob = predict_food(file_path)
        result_label.config(text=f"🍜 {label} ({prob*100:.2f}%)")
    except Exception as e:
        result_label.config(text=f"Lỗi: {e}")

# ==== Tạo cửa sổ chính ====
root = tk.Tk()
root.title("Nhận diện món ăn Việt Nam (ANN)")

upload_btn = tk.Button(root, text="Chọn ảnh", command=upload_image)
upload_btn.pack()

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack()

root.mainloop()
