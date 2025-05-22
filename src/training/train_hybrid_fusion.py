# Load data yang sudah dipersiapkan
X_train_images = np.load('D:/Preprocessing/CNN_Data/X_train_images.npy')
X_train_landmarks = np.load('D:/Preprocessing/LSTM_Data/X_train_landmarks.npy')
y_train = np.load('D:/Preprocessing/CNN_Data/y_train_images.npy')

X_val_images = np.load('D:/Preprocessing/CNN_Data/X_val_images.npy')
X_val_landmarks = np.load('D:/Preprocessing/LSTM_Data/X_val_landmarks.npy')
y_val = np.load('D:/Preprocessing/CNN_Data/y_val_images.npy')

X_test_images = np.load('D:/Preprocessing/CNN_Data/X_test_images.npy')
X_test_landmarks = np.load('D:/Preprocessing/LSTM_Data/X_test_landmarks.npy')
y_test = np.load('D:/Preprocessing/CNN_Data/y_test_images.npy')

# Convert string labels ke numerical jika diperlukan
if isinstance(y_train[0], str):
    y_train = np.array([label_map[label] for label in y_train])
    y_val = np.array([label_map[label] for label in y_val])
    y_test = np.array([label_map[label] for label in y_test])

# Convert ke one-hot encoding
y_train_one_hot = to_categorical(y_train, num_classes)
y_val_one_hot = to_categorical(y_val, num_classes)
y_test_one_hot = to_categorical(y_test, num_classes)

# Bangun model hybrid fusion
def build_hybrid_fusion_model(input_shape_image=(224, 224, 3), input_shape_landmark=136, num_classes=7):
    # CNN Stream (untuk gambar)
    image_input = Input(shape=input_shape_image, name='image_input')
    
    # CNN layers
    x_img = Conv2D(32, (3, 3), padding='same', activation='relu')(image_input)
    x_img = BatchNormalization()(x_img)
    x_img = Conv2D(32, (3, 3), padding='same', activation='relu')(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = MaxPooling2D(pool_size=(2, 2))(x_img)
    x_img = Dropout(0.25)(x_img)
    
    x_img = Conv2D(64, (3, 3), padding='same', activation='relu')(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = Conv2D(64, (3, 3), padding='same', activation='relu')(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = MaxPooling2D(pool_size=(2, 2))(x_img)
    x_img = Dropout(0.25)(x_img)
    
    x_img = Conv2D(128, (3, 3), padding='same', activation='relu')(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = Conv2D(128, (3, 3), padding='same', activation='relu')(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = MaxPooling2D(pool_size=(2, 2))(x_img)
    x_img = Dropout(0.25)(x_img)
    
    x_img = Flatten()(x_img)
    x_img = Dense(256, activation='relu')(x_img)
    x_img = BatchNormalization()(x_img)
    x_img = Dropout(0.5)(x_img)
    
    # Landmark Stream
    landmark_input = Input(shape=(input_shape_landmark,), name='landmark_input')
    
    x_lm = Dense(128, activation='relu')(landmark_input)
    x_lm = BatchNormalization()(x_lm)
    x_lm = Dropout(0.3)(x_lm)
    
    x_lm = Dense(256, activation='relu')(x_lm)
    x_lm = BatchNormalization()(x_lm)
    x_lm = Dropout(0.3)(x_lm)
    
    x_lm = Dense(128, activation='relu')(x_lm)
    x_lm = BatchNormalization()(x_lm)
    x_lm = Dropout(0.3)(x_lm)
    
    # Fusion layer - menggabungkan kedua stream
    fusion = concatenate([x_img, x_lm], name='fusion_layer')
    
    # Fully connected layers setelah fusion
    x = Dense(512, activation='relu')(fusion)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer
    output = Dense(num_classes, activation='softmax', name='output')(x)
    
    # Build model
    model = Model(inputs=[image_input, landmark_input], outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Buat callbacks
checkpoint = ModelCheckpoint(
    'D:/Models/hybrid_fusion_model_best.h5',
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# Latih model hybrid fusion
hybrid_model = build_hybrid_fusion_model(
    input_shape_image=X_train_images.shape[1:], 
    input_shape_landmark=X_train_landmarks.shape[1],
    num_classes=num_classes
)

hybrid_history = hybrid_model.fit(
    [X_train_images, X_train_landmarks],
    y_train_one_hot,
    batch_size=32,
    epochs=100,
    validation_data=([X_val_images, X_val_landmarks], y_val_one_hot),
    callbacks=[checkpoint, early_stopping]
)

# Evaluasi model
hybrid_model = load_model('D:/Models/hybrid_fusion_model_best.h5')  # Load model terbaik
test_loss, test_acc = hybrid_model.evaluate([X_test_images, X_test_landmarks], y_test_one_hot)
print(f"Hybrid Fusion Test accuracy: {test_acc}")

# Prediksi untuk evaluasi detail
y_pred_hybrid = np.argmax(hybrid_model.predict([X_test_images, X_test_landmarks]), axis=1)
y_true = np.argmax(y_test_one_hot, axis=1)

# Tampilkan classification report
print("\nHybrid Fusion Classification Report:")
print(classification_report(y_true, y_pred_hybrid, target_names=target_names))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_hybrid)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Hybrid Fusion Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('D:/Models/hybrid_fusion_confusion_matrix.png')
plt.close()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(hybrid_history.history['accuracy'])
plt.plot(hybrid_history.history['val_accuracy'])
plt.title('Hybrid Fusion Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(hybrid_history.history['loss'])
plt.plot(hybrid_history.history['val_loss'])
plt.title('Hybrid Fusion Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('D:/Models/hybrid_fusion_training_history.png')
plt.close()