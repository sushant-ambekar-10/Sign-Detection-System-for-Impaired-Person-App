# # Import Libraries
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# # Step 1: Set Dataset Path
# DATASET_DIR = "asl_dataset"

# # Step 2: Load and Preprocess Dataset
# IMG_SIZE = 128  # Larger size for MobileNetV2
# X = []
# y = []
# labels = sorted(os.listdir(DATASET_DIR))
# label_dict = {label: idx for idx, label in enumerate(labels)}

# for label in labels:
#     folder_path = os.path.join(DATASET_DIR, label)
#     for img_name in os.listdir(folder_path):
#         img_path = os.path.join(folder_path, img_name)
#         try:
#             img = cv2.imread(img_path)
#             img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#             X.append(img)
#             y.append(label_dict[label])
#         except Exception as e:
#             print(f"Error loading image {img_path}: {e}")

# X = np.array(X) / 255.0  # Normalize
# y = np.array(y)

# # Step 3: Split Dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Step 4: Build Model (Transfer Learning)
# base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
# base_model.trainable = False  # Freeze base model

# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Dropout(0.3),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(len(label_dict), activation='softmax')
# ])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.summary()

# # Step 5: Callbacks
# checkpoint = ModelCheckpoint('asl_mobilenet_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
# early_stop = EarlyStopping(monitor='val_accuracy', patience=5)

# # Step 6: Train Model
# history = model.fit(X_train, y_train, epochs=30, validation_split=0.2, batch_size=32, callbacks=[checkpoint, early_stop])

# # Step 7: Evaluate Model
# model.load_weights('asl_mobilenet_best.h5')
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy * 100:.2f}%")

# # Step 8: Classification Report
# y_pred = model.predict(X_test)
# y_pred_classes = np.argmax(y_pred, axis=1)

# print("\nClassification Report:\n")
# print(classification_report(y_test, y_pred_classes, target_names=label_dict.keys()))

# # Step 9: Confusion Matrix
# cm = confusion_matrix(y_test, y_pred_classes)
# plt.figure(figsize=(12, 10))
# plt.imshow(cm, cmap='Blues')
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.colorbar()
# plt.show()

# # Step 10: Save Full Model
# model.save('asl_mobilenet_full.h5')

# # Step 11: Save Dataset (Optional)
# np.save('X.npy', X)
# np.save('y.npy', y)

# print("Dataset, model, and reports saved successfully.")


# Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Step 1: Set Dataset Path
DATASET_DIR = "asl_dataset"

# Step 2: Load and Preprocess Dataset
IMG_SIZE = 128  # Larger size for MobileNetV2
X = []
y = []
labels = sorted(os.listdir(DATASET_DIR))
label_dict = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    folder_path = os.path.join(DATASET_DIR, label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(label_dict[label])
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

X = np.array(X) / 255.0  # Normalize
y = np.array(y)

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 4: Build Model (Transfer Learning)
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(label_dict), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Step 5: Callbacks (COMMENTED to avoid retraining)
# checkpoint = ModelCheckpoint('asl_mobilenet_best.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
# early_stop = EarlyStopping(monitor='val_accuracy', patience=5)

# Step 6: Train Model (COMMENTED to skip retraining)
# history = model.fit(X_train, y_train, epochs=30, validation_split=0.2, batch_size=32, callbacks=[checkpoint, early_stop])

# Step 7: Evaluate Model
# Load the saved weights instead of retraining
model.load_weights('asl_mobilenet_best.h5')
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 8: Classification Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_classes, target_names=label_dict.keys()))

# Step 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()

# Step 10: Save Full Model
model.save('asl_mobilenet_full.h5')

# Step 11: Save Dataset (Optional)
np.save('X.npy', X)
np.save('y.npy', y)

print("Dataset, model, and reports saved successfully.")

