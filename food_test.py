import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ==== Load model Keras ANN ====
model = load_model("vietnamese_food.h5")

# Load label encoder
with open("label_encoder_food.pkl", "rb") as f:
    encoder = pickle.load(f)

# Hàm dự đoán
def predict_food(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128)) / 255.0   # resize giống lúc train
    x = img.reshape(1, 128, 128)                # input cho model (Flatten ở trong model)
    
    # Dự đoán
    preds = model.predict(x)
    pred_idx = np.argmax(preds, axis=1)[0]
    prob = np.max(preds)

    return encoder.inverse_transform([pred_idx])[0], prob

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
root.title("Nhận diện món ăn Việt Nam (ANN - Keras)")

upload_btn = tk.Button(root, text="Chọn ảnh", command=upload_image)
upload_btn.pack()

panel = tk.Label(root)
panel.pack()

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.pack()

root.mainloop()
