# Master's Level Handwriting Recognition Project - CNN Based Deep Learning Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------- Load and Prepare Data -------------------------
print("Loading training data...")
train_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/handwiriting/Train.csv')
X = train_data.drop('label', axis=1).values
y = train_data['label'].values

# Normalize pixel values
X = X.astype('float32') / 255.0
X = X.reshape(-1, 28, 28, 1)

# One-hot encode labels
y = to_categorical(y, num_classes=10)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------- Data Augmentation -------------------------
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# ------------------------- Model Architecture (CNN) -------------------------
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ------------------------- Callbacks -------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# ------------------------- Model Training -------------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr, checkpoint],
    verbose=1
)

# ------------------------- Load Best Model -------------------------
best_model = load_model('best_model.h5')

# ------------------------- Model Evaluation -------------------------
val_loss, val_acc = best_model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# ------------------------- Plot Accuracy and Loss -------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.show()

# ------------------------- Test Set Evaluation -------------------------
test_data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/handwiriting/mnist_test.csv')
X_test = test_data.drop('label', axis=1).values
y_test_true = test_data['label'].values

X_test = X_test.astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

y_test_true_categorical = to_categorical(y_test_true, num_classes=10)

test_loss, test_acc = best_model.evaluate(X_test, y_test_true_categorical)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# ------------------------- Classification Report -------------------------
y_pred_probs = best_model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("Classification Report:")
print(classification_report(y_test_true, y_pred))

# ------------------------- Confusion Matrix -------------------------
conf_matrix = confusion_matrix(y_test_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ------------------------- Visualizing Correctly Classified Images -------------------------
correct_indices = np.where(y_pred == y_test_true)[0]
print(f"Number of correctly classified images: {len(correct_indices)}")

plt.figure(figsize=(10, 10))
for i, idx in enumerate(correct_indices[:9]):  # Display first 9 correct samples
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test_true[idx]}, Pred: {y_pred[idx]}")
    plt.axis('off')
plt.suptitle("Correctly Classified Examples")
plt.tight_layout()
plt.show()

# ------------------------- Visualizing Misclassified Images -------------------------
wrong_indices = np.where(y_pred != y_test_true)[0]
print(f"Number of misclassified images: {len(wrong_indices)}")

plt.figure(figsize=(10, 10))
for i, idx in enumerate(wrong_indices[:9]):  # Display first 9 misclassified samples
    plt.subplot(3, 3, i + 1)
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test_true[idx]}, Pred: {y_pred[idx]}", color='red')
    plt.axis('off')
plt.suptitle("Misclassified Examples")
plt.tight_layout()
plt.show()

# ------------------------- Error Analysis (Misclassified Images) -------------------------
wrong = np.where(y_pred != y_test_true)[0]
print(f"Number of misclassified images: {len(wrong)}")

for i in range(5):
    idx = wrong[i]
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test_true[idx]}, Predicted: {y_pred[idx]}")
    plt.axis('off')
    plt.show()

# ------------------------- Model Saving (For Deployment) -------------------------
# Save as .h5 and SavedModel format
best_model.save('final_handwriting_model.h5')
# ------------------------- Model Saving (For Deployment) -------------------------
# Save in Keras recommended format
best_model.save('final_handwriting_model.keras')

# Optional: Export as TensorFlow SavedModel format for TFLite / TF Serving
best_model.export('final_handwriting_model_tf')

print("Model saved successfully in both formats.")


print("Model saved successfully.")

# ------------------------- Optional: Flask API Deployment Starter -------------------------
'''
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('final_handwriting_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['image']  # Expect flattened 784 pixel array
    image = np.array(data).reshape(1,28,28,1) / 255.0
    prediction = np.argmax(model.predict(image), axis=1)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
'''
