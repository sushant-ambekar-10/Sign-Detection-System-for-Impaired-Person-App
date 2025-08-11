# # ========== GUI SETUP ==========

# import cv2
# import numpy as np
# import tensorflow as tf
# import pyttsx3
# import tkinter as tk
# from PIL import Image, ImageTk
# import mediapipe as mp
# import pickle
# import os
# import time

# # Load the trained model
# model = tf.keras.models.load_model("asl_mobilenet_best.h5")

# # Load labels
# if os.path.exists("labels.pkl"):
#     with open("labels.pkl", "rb") as f:
#         label_map = pickle.load(f)
#     labels = {v: k for k, v in label_map.items()}
# else:
#     labels = {i: str(i) for i in range(36)}

# IMG_SIZE = 128
# sentence = ""
# current_pred = [""]
# engine = pyttsx3.init()

# # Mediapipe setup
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
# mp_draw = mp.solutions.drawing_utils

# # Timing
# start_time = None
# capture_duration = 2
# current_highest_label = ""
# current_highest_prob = 0

# # Preprocess image
# def preprocess_hand_image(img):
#     h, w, _ = img.shape
#     size = max(h, w)
#     white_bg = np.ones((size, size, 3), dtype=np.uint8) * 255
#     x_offset = (size - w) // 2
#     y_offset = (size - h) // 2
#     white_bg[y_offset:y_offset+h, x_offset:x_offset+w] = img
#     resized = cv2.resize(white_bg, (IMG_SIZE, IMG_SIZE))
#     normalized = resized / 255.0
#     return normalized.astype(np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 3)

# def speak():
#     engine.say(sentence_output.get())
#     engine.runAndWait()

# def confirm_character():
#     global sentence
#     sentence += current_pred[0]
#     sentence_output.set(sentence)

# def add_space():
#     global sentence
#     sentence += " "
#     sentence_output.set(sentence)

# def remove_character():
#     global sentence
#     if sentence:
#         sentence = sentence[:-1]
#         sentence_output.set(sentence)

# def predict_frame():
#     global start_time, current_highest_label, current_highest_prob

#     ret, frame = cap.read()
#     if not ret:
#         root.after(30, predict_frame)
#         return

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
#     display_frame = np.ones_like(frame) * 255

#     if results.multi_hand_landmarks:
#         hand_landmarks = results.multi_hand_landmarks[0]
#         h, w, _ = frame.shape

#         x_min, y_min = w, h
#         x_max = y_max = 0
#         for lm in hand_landmarks.landmark:
#             x, y = int(lm.x * w), int(lm.y * h)
#             x_min = min(x_min, x)
#             y_min = min(y_min, y)
#             x_max = max(x_max, x)
#             y_max = max(y_max, y)

#         margin = 20
#         x_min = max(0, x_min - margin)
#         y_min = max(0, y_min - margin)
#         x_max = min(w, x_max + margin)
#         y_max = min(h, y_max + margin)

#         hand_img = frame[y_min:y_max, x_min:x_max]
#         if hand_img.shape[0] == 0 or hand_img.shape[1] == 0:
#             label_var.set("Invalid crop")
#             root.after(30, predict_frame)
#             return

#         input_img = preprocess_hand_image(hand_img)
#         pred = model.predict(input_img, verbose=0)[0]
#         prob = np.max(pred)
#         pred_label = labels[np.argmax(pred)]

#         current_time = time.time()
#         if start_time is None:
#             start_time = current_time
#             current_highest_label = pred_label
#             current_highest_prob = prob
#         else:
#             if prob > current_highest_prob:
#                 current_highest_label = pred_label
#                 current_highest_prob = prob

#             if current_time - start_time >= capture_duration:
#                 label_var.set(f"Captured: {current_highest_label} ({current_highest_prob:.2f})")
#                 current_pred[0] = current_highest_label
#                 start_time = None
#                 current_highest_prob = 0
#             else:
#                 remaining_time = capture_duration - (current_time - start_time)
#                 label_var.set(f"Predicting: {pred_label} ({prob:.2f}) | {remaining_time:.1f}s left")

#         display_frame[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]
#         cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#     else:
#         label_var.set("No hand detected")
#         start_time = None

#     img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)))
#     canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
#     canvas.img_tk = img_tk
#     root.after(30, predict_frame)

# # GUI Setup
# root = tk.Tk()
# root.title("ASL Real-Time Detector (MobileNetV2 - Timer-based)")
# root.configure(bg="#f0f4f7")

# canvas = tk.Canvas(root, width=640, height=480)
# canvas.pack(pady=10)

# label_var = tk.StringVar()
# tk.Label(root, textvariable=label_var, font=("Helvetica", 14), bg="#f0f4f7", fg="#333").pack(pady=5)

# button_frame = tk.Frame(root, bg="#f0f4f7")
# button_frame.pack(pady=10)

# button_style = {
#     "font": ("Helvetica", 12),
#     "width": 20,
#     "bg": "#4CAF50",
#     "fg": "white",
#     "activebackground": "#45a049",
#     "relief": tk.RAISED
# }

# space_button_style = button_style.copy()
# space_button_style.update({"bg": "#2196F3", "activebackground": "#1e88e5"})

# remove_button_style = button_style.copy()
# remove_button_style.update({"bg": "#f44336", "activebackground": "#e53935"})

# tk.Button(button_frame, text="✅ Confirm Character", command=confirm_character, **button_style).grid(row=0, column=0, padx=5, pady=5)
# tk.Button(button_frame, text="➖ Add Space", command=add_space, **space_button_style).grid(row=0, column=1, padx=5, pady=5)
# tk.Button(button_frame, text="❌ Remove Character", command=remove_character, **remove_button_style).grid(row=0, column=2, padx=5, pady=5)

# sentence_output = tk.StringVar()
# tk.Label(root, textvariable=sentence_output, font=("Helvetica", 16), fg="#1a237e", bg="#f0f4f7").pack(pady=10)

# tk.Button(root, text="🔊 Speak Sentence", command=speak, **button_style).pack(pady=5)

# # Start webcam and GUI
# cap = cv2.VideoCapture(0)
# predict_frame()
# root.mainloop()
# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import tkinter as tk
from PIL import Image, ImageTk
import mediapipe as mp
import pickle
import os
import time
from googletrans import Translator

# Load the trained model
model = tf.keras.models.load_model("asl_mobilenet_best.h5")

# Load labels
if os.path.exists("labels.pkl"):
    with open("labels.pkl", "rb") as f:
        label_map = pickle.load(f)
    labels = {v: k for k, v in label_map.items()}
else:
    labels = {i: str(i) for i in range(36)}

IMG_SIZE = 128
sentence = ""
current_pred = [""]
engine = pyttsx3.init()
translator = Translator()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Timing
start_time = None
capture_duration = 2
current_highest_label = ""
current_highest_prob = 0

# Preprocess image
def preprocess_hand_image(img):
    h, w, _ = img.shape
    size = max(h, w)
    white_bg = np.ones((size, size, 3), dtype=np.uint8) * 255
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    white_bg[y_offset:y_offset+h, x_offset:x_offset+w] = img
    resized = cv2.resize(white_bg, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized.astype(np.float32).reshape(1, IMG_SIZE, IMG_SIZE, 3)

# Speak the sentence
def speak():
    engine.say(sentence_output.get())
    engine.runAndWait()

# Confirm character
def confirm_character():
    global sentence
    sentence += current_pred[0]
    sentence_output.set(sentence)

# Add space
def add_space():
    global sentence
    sentence += " "
    sentence_output.set(sentence)

# Remove last character
def remove_character():
    global sentence
    if sentence:
        sentence = sentence[:-1]
        sentence_output.set(sentence)

# Translate to Marathi
def translate_to_marathi():
    english_text = sentence_output.get()
    if english_text.strip() == "":
        translation_output.set("Please enter a sentence.")
        return
    try:
        translated = translator.translate(english_text, src='en', dest='mr')
        translation_output.set(translated.text)
    except Exception as e:
        translation_output.set(f"Error: {str(e)}")

# Frame prediction
def predict_frame():
    global start_time, current_highest_label, current_highest_prob

    ret, frame = cap.read()
    if not ret:
        root.after(30, predict_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    display_frame = np.ones_like(frame) * 255

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
            root.after(30, predict_frame)
            return

        input_img = preprocess_hand_image(hand_img)
        pred = model.predict(input_img, verbose=0)[0]
        prob = np.max(pred)
        pred_label = labels[np.argmax(pred)]

        current_time = time.time()
        if start_time is None:
            start_time = current_time
            current_highest_label = pred_label
            current_highest_prob = prob
        else:
            if prob > current_highest_prob:
                current_highest_label = pred_label
                current_highest_prob = prob

            if current_time - start_time >= capture_duration:
                label_var.set(f"Captured: {current_highest_label} ({current_highest_prob:.2f})")
                current_pred[0] = current_highest_label
                start_time = None
                current_highest_prob = 0
            else:
                remaining_time = capture_duration - (current_time - start_time)
                label_var.set(f"Predicting: {pred_label} ({prob:.2f}) | {remaining_time:.1f}s left")

        display_frame[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    else:
        label_var.set("No hand detected")
        start_time = None

    img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)))
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.img_tk = img_tk
    root.after(30, predict_frame)

# GUI Setup
root = tk.Tk()
root.title("ASL Real-Time Detector (MobileNetV2 + Marathi Translation)")
root.configure(bg="#f0f4f7")

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack(pady=10)

label_var = tk.StringVar()
tk.Label(root, textvariable=label_var, font=("Helvetica", 14), bg="#f0f4f7", fg="#333").pack(pady=5)

button_frame = tk.Frame(root, bg="#f0f4f7")
button_frame.pack(pady=10)

button_style = {
    "font": ("Helvetica", 12),
    "width": 20,
    "bg": "#4CAF50",
    "fg": "white",
    "activebackground": "#45a049",
    "relief": tk.RAISED
}

space_button_style = button_style.copy()
space_button_style.update({"bg": "#2196F3", "activebackground": "#1e88e5"})

remove_button_style = button_style.copy()
remove_button_style.update({"bg": "#f44336", "activebackground": "#e53935"})

tk.Button(button_frame, text="✅ Confirm Character", command=confirm_character, **button_style).grid(row=0, column=0, padx=5, pady=5)
tk.Button(button_frame, text="➖ Add Space", command=add_space, **space_button_style).grid(row=0, column=1, padx=5, pady=5)
tk.Button(button_frame, text="❌ Remove Character", command=remove_character, **remove_button_style).grid(row=0, column=2, padx=5, pady=5)

sentence_output = tk.StringVar()
tk.Label(root, textvariable=sentence_output, font=("Helvetica", 16), fg="#1a237e", bg="#f0f4f7").pack(pady=10)

tk.Button(root, text="🔊 Speak Sentence", command=speak, **button_style).pack(pady=5)

# Marathi translation
tk.Button(root, text="🌐 Translate to Marathi", command=translate_to_marathi, **button_style).pack(pady=5)

translation_output = tk.StringVar()
tk.Label(root, textvariable=translation_output, font=("Helvetica", 14), fg="#004d40", bg="#f0f4f7", wraplength=600).pack(pady=5)

# Start webcam and GUI
cap = cv2.VideoCapture(0)
predict_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()
