# train.py

import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras import regularizers # type: ignore

with open('cnn_landmarks.pickle', 'rb') as f:
    dataset = pickle.load(f)

X = np.array(dataset['data'])  # shape: (n, 2, 21, 2)
y = np.array(dataset['labels'])

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded)

# Build CNN model
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2, 21, 2)),
    tf.keras.layers.Conv2D(32, (1, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((1, 2)),
    tf.keras.layers.Conv2D(64, (1, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(2, 21, 2)),

    tf.keras.layers.Conv2D(32, (1, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((1, 2)),

    tf.keras.layers.Conv2D(64, (1, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation='relu', 
                          kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_test, y_test),
    batch_size=16
)

#model.save('isl_landmark_cnn.h5')
model.save('isl_landmark_cnn.keras')

# Save label encoder
with open('label_encoder.pickle', 'wb') as f:
    pickle.dump(le, f)

# Generate predictions for the test set
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels, target_names=le.classes_))

# Accuracy plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')  # saves as image for your report
plt.show()

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
