# Load data landmark yang sudah dipersiapkan
X_train_landmarks = np.load('D:/Preprocessing/LSTM_Data/X_train_landmarks.npy')
y_train = np.load('D:/Preprocessing/LSTM_Data/y_train.npy')
X_val_landmarks = np.load('D:/Preprocessing/LSTM_Data/X_val_landmarks.npy')
y_val = np.load('D:/Preprocessing/LSTM_Data/y_val.npy')
X_test_landmarks = np.load('D:/Preprocessing/LSTM_Data/X_test_landmarks.npy')
y_test = np.load('D:/Preprocessing/LSTM_Data/y_test.npy')

# Convert string labels ke numerical jika diperlukan
if isinstance(y_train[0], str):
    y_train = np.array([label_map[label] for label in y_train])
    y_val = np.array([label_map[label] for label in y_val])
    y_test = np.array([label_map[label] for label in y_test])

# Convert ke one-hot encoding
y_train_one_hot = to_categorical(y_train, num_classes)
y_val_one_hot = to_categorical(y_val, num_classes)
y_test_one_hot = to_categorical(y_test, num_classes)

# Reshape data untuk LSTM jika diperlukan
# Jika data landmark Anda sudah dalam format sequence, Anda perlu reshape
# Jika tidak, Anda bisa membuat sequence frame-by-frame atau menggunakan Dense layers
input_shape = X_train_landmarks.shape[1]

# Buat model LSTM untuk landmark
from tensorflow.keras.layers import LSTM, Bidirectional

def build_landmark_model(input_shape, num_classes=7):
    model = Sequential([
        # Reshape layer jika diperlukan (convert flat landmarks ke sequence)
        # Jika data landmark Anda adalah (batch_size, 136), perlu reshape menjadi 
        # sequence seperti (batch_size, seq_length, feature_dim)
        # Alternatifnya, gunakan Dense layers untuk flat features:
        
        Dense(128, activation='relu', input_shape=(input_shape,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Buat callbacks
checkpoint = ModelCheckpoint(
    'D:/Models/landmark_model_best.h5',
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Latih model landmark
landmark_model = build_landmark_model(input_shape, num_classes=num_classes)
landmark_history = landmark_model.fit(
    X_train_landmarks,
    y_train_one_hot,
    batch_size=32,
    epochs=100,
    validation_data=(X_val_landmarks, y_val_one_hot),
    callbacks=[checkpoint, early_stopping]
)

# Evaluasi model
landmark_model = load_model('D:/Models/landmark_model_best.h5')  # Load model terbaik
test_loss, test_acc = landmark_model.evaluate(X_test_landmarks, y_test_one_hot)
print(f"Test accuracy: {test_acc}")

# Prediksi untuk evaluasi detail
y_pred = np.argmax(landmark_model.predict(X_test_landmarks), axis=1)
y_true = np.argmax(y_test_one_hot, axis=1)

# Tampilkan classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('D:/Models/landmark_confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(landmark_history.history['accuracy'])
plt.plot(landmark_history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(landmark_history.history['loss'])
plt.plot(landmark_history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('D:/Models/landmark_training_history.png')
plt.close()