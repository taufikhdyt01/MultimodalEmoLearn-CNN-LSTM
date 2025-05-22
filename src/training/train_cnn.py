import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data yang sudah dipersiapkan
X_train_images = np.load('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/data/images/X_train_images.npy')
y_train = np.load('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/data/images/y_train_images.npy')
X_val_images = np.load('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/data/images/X_val_images.npy')
y_val = np.load('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/data/images//y_val_images.npy')
X_test_images = np.load('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/data/images/X_test_images.npy')
y_test = np.load('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/data/images/y_test_images.npy')

# Periksa label apa saja yang ada
print("Unique labels in training data:", np.unique(y_train))
print("Unique labels in validation data:", np.unique(y_val))
print("Unique labels in test data:", np.unique(y_test))

# Buat mapping dari label string ke integer
unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
label_map = {label: i for i, label in enumerate(unique_labels)}
print("Label mapping:", label_map)

# Convert string labels ke numerical menggunakan mapping
y_train_num = np.array([label_map[label] for label in y_train])
y_val_num = np.array([label_map[label] for label in y_val])
y_test_num = np.array([label_map[label] for label in y_test])

# Print contoh konversi untuk verifikasi
print("\nSample Label Conversion:")
for i in range(min(5, len(y_train))):
    print(f"Original: {y_train[i]} -> Numeric: {y_train_num[i]}")

# Hitung jumlah kelas
num_classes = len(label_map)
print(f"Number of classes: {num_classes}")

# Convert ke one-hot encoding
y_train_one_hot = to_categorical(y_train_num, num_classes)
y_val_one_hot = to_categorical(y_val_num, num_classes)
y_test_one_hot = to_categorical(y_test_num, num_classes)

# Print contoh one-hot encoding untuk verifikasi
print("\nSample One-Hot Encoding:")
for i in range(min(3, len(y_train))):
    print(f"Label: {y_train[i]} (Numeric: {y_train_num[i]}) -> One-Hot: {y_train_one_hot[i]}")

# Buat model CNN
def build_cnn_model(input_shape=(224, 224, 3), num_classes=7):
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Block 4
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        
        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Buat direktori untuk menyimpan model jika belum ada
os.makedirs('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/', exist_ok=True)

# Buat callbacks
checkpoint = ModelCheckpoint(
    'D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/cnn_model_best.h5',
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Latih model CNN
print("Building and training CNN model...")
cnn_model = build_cnn_model(num_classes=num_classes)
print(cnn_model.summary())

cnn_history = cnn_model.fit(
    X_train_images,
    y_train_one_hot,
    batch_size=32,
    epochs=50,
    validation_data=(X_val_images, y_val_one_hot),
    callbacks=[checkpoint, early_stopping]
)

# Evaluasi model
print("Evaluating CNN model...")
cnn_model = load_model('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/cnn_model_best.h5')  # Load model terbaik
test_loss, test_acc = cnn_model.evaluate(X_test_images, y_test_one_hot)
print(f"Test accuracy: {test_acc}")

# Prediksi untuk evaluasi detail
y_pred = np.argmax(cnn_model.predict(X_test_images), axis=1)
y_true = np.argmax(y_test_one_hot, axis=1)

# Tampilkan classification report
print("\nClassification Report:")
# Gunakan nama label asli untuk report
target_names = list(label_map.keys())
print(classification_report(y_true, y_pred, target_names=target_names))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/cnn_confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'])
plt.plot(cnn_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'])
plt.plot(cnn_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/cnn_training_history.png')
plt.close()

# Simpan label mapping untuk digunakan kembali
import pickle
with open('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/label_map.pkl', 'wb') as f:
    pickle.dump(label_map, f)

print("Training and evaluation completed. Results saved to D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/")