import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import pickle

# Load model and label mapping
model = tf.keras.models.load_model("asl_mobilenetv2.h5")
with open("labels.pkl", "rb") as f:
    label_map = pickle.load(f)
labels = {v: k for k, v in label_map.items()}

IMG_SIZE = 224
sentence = ""
current_pred = [""]

# Text-to-speech engine
engine = pyttsx3.init()

# MediaPipe hand detection setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Preprocess cropped hand image
def preprocess_hand_image(img):
    h, w, _ = img.shape
    size = max(h, w)
    black_bg = np.zeros((size, size, 3), dtype=np.uint8)  # Black background
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    black_bg[y_offset:y_offset+h, x_offset:x_offset+w] = img
    resized = cv2.resize(black_bg, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized.astype(np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 3)

# Speak the current sentence
def speak():
    engine.say(sentence_output.get())
    engine.runAndWait()

# Predict function (runs every 10ms)
def predict_frame():
    ret, frame = cap.read()
    if not ret:
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    display_frame = np.zeros_like(frame)  # Black background

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape

        x_min, y_min = w, h
        x_max = y_max = 0

        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        margin = 20
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        hand_img = frame[y_min:y_max, x_min:x_max]
        if hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
            label_var.set("Invalid crop")
            return

        input_img = preprocess_hand_image(hand_img)
        pred = model.predict(input_img, verbose=0)[0]
        prob = np.max(pred)
        pred_label = labels[np.argmax(pred)]

        label_var.set(f"Prediction: {pred_label} ({prob:.2f})")
        current_pred[0] = pred_label

        # Show hand only on black background
        display_frame[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    else:
        label_var.set("No hand detected")

    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)))
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.img_tk = img_tk
    root.after(10, predict_frame)

# Confirm predicted character and add to sentence
def confirm_character():
    global sentence
    sentence += current_pred[0]
    sentence_output.set(sentence)

# GUI setup
root = tk.Tk()
root.title("ASL Real-Time Detector (MobileNetV2)")

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

label_var = tk.StringVar()
tk.Label(root, textvariable=label_var, font=("Helvetica", 14)).pack(pady=10)

tk.Button(root, text="✅ Confirm Character", command=confirm_character, width=20).pack(pady=5)

sentence_output = tk.StringVar()
tk.Label(root, textvariable=sentence_output, font=("Helvetica", 16), fg="blue").pack(pady=10)

tk.Button(root, text="🔊 Speak Sentence", command=speak, width=20).pack(pady=5)

# Start webcam and detection loop
cap = cv2.VideoCapture(0)
predict_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
