import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load trained model
model = tf.keras.models.load_model("emotion_model.h5")
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']  # Update based on your dataset

def preprocess_image(path_or_array):
    try:
        if isinstance(path_or_array, str):
            img = cv2.imread(path_or_array, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.cvtColor(path_or_array, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (48, 48))
        img = img.astype('float32') / 255.0
        return img.reshape(1, 48, 48, 1)
    except Exception as e:
        messagebox.showerror("Preprocessing Error", str(e))
        return None

def show_predictions(prediction):
    top_indices = prediction.argsort()[-3:][::-1]
    result_text = "\n".join([f"{class_names[i]}: {round(prediction[i]*100, 2)}%" for i in top_indices])
    result_label.config(text=result_text)

    # Bar chart
    fig = plt.Figure(figsize=(4,2.2), dpi=100)
    ax = fig.add_subplot(111)
    ax.bar(class_names, prediction * 100, color='skyblue')
    ax.set_ylim(0, 100)
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Emotion Probabilities")

    for widget in chart_frame.winfo_children():
        widget.destroy()

    chart = FigureCanvasTkAgg(fig, master=chart_frame)
    chart.draw()
    chart.get_tk_widget().pack()

def predict_from_file():
    path = filedialog.askopenfilename()
    if not path:
        return
    img_array = preprocess_image(path)
    if img_array is not None:
        prediction = model.predict(img_array)[0]
        show_predictions(prediction)

        img = Image.open(path).resize((140, 140))
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img

def capture_from_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Webcam Error", "Cannot access webcam")
        return

    ret, frame = cap.read()
    cap.release()

    if ret:
        img_array = preprocess_image(frame)
        if img_array is not None:
            prediction = model.predict(img_array)[0]
            show_predictions(prediction)

            img = cv2.resize(frame, (140, 140))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)

            image_label.config(image=img)
            image_label.image = img

# GUI
root = tk.Tk()
root.title("Facial Emotion Classifier")
root.geometry("500x600")
root.config(bg="#f0f2f5")

tk.Label(root, text="Facial Emotion Recognition", font=("Arial", 16, "bold"), bg="#f0f2f5", fg="#333").pack(pady=15)

tk.Button(root, text="Upload Face Image", command=predict_from_file, font=("Arial", 12), bg="#4caf50", fg="white", width=20).pack(pady=5)
tk.Button(root, text="Capture from Webcam", command=capture_from_webcam, font=("Arial", 12), bg="#2196f3", fg="white", width=20).pack(pady=5)

image_label = tk.Label(root, bg="#f0f2f5")
image_label.pack(pady=10)

result_label = tk.Label(root, text="", font=("Arial", 12), fg="darkblue", bg="#f0f2f5")
result_label.pack(pady=10)

chart_frame = tk.Frame(root, bg="#f0f2f5")
chart_frame.pack(pady=10)

tk.Label(root, text="Model: emotion_model.h5", font=("Arial", 8), bg="#f0f2f5", fg="gray").pack(side=tk.BOTTOM, pady=5)

root.mainloop()
