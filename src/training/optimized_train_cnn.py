import numpy as np
import tensorflow as tf
import os
import time
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ============================
# AMD GPU SETUP FOR RX 6600 LE
# ============================
def setup_amd_gpu():
    """Setup AMD GPU with DirectML for optimal performance"""
    print("🎮 Setting up AMD RX 6600 LE...")
    
    try:
        # Check for GPU availability
        gpu_devices = tf.config.list_physical_devices('GPU')
        dml_devices = tf.config.list_physical_devices('DML')
        
        total_devices = len(gpu_devices) + len(dml_devices)
        
        if total_devices > 0:
            print(f"✅ Found {total_devices} GPU device(s)")
            
            # Configure GPU memory growth to prevent OOM
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
            
            # Test GPU functionality
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test_tensor, test_tensor)
                print("✅ GPU computation test passed")
            
            # Enable XLA compilation for better performance
            tf.config.optimizer.set_jit(True)
            print("✅ XLA compilation enabled")
            
            return True, total_devices
            
        else:
            print("⚠️ No GPU detected, using CPU")
            return False, 0
            
    except Exception as e:
        print(f"⚠️ GPU setup failed: {e}")
        print("💡 Falling back to CPU mode")
        return False, 0

# Setup GPU
gpu_available, num_gpus = setup_amd_gpu()

# ============================
# OPTIMIZED CONFIGURATIONS
# ============================

# GPU-optimized batch sizes for RX 6600 LE (8GB VRAM)
if gpu_available:
    BATCH_SIZE = 32  # Optimal for RX 6600 LE
    EPOCHS = 50
    VALIDATION_FREQ = 1
    print(f"🚀 GPU Mode: Batch size {BATCH_SIZE}, {EPOCHS} epochs")
else:
    BATCH_SIZE = 16  # Conservative for CPU
    EPOCHS = 30
    VALIDATION_FREQ = 1
    print(f"💻 CPU Mode: Batch size {BATCH_SIZE}, {EPOCHS} epochs")

# Paths
BASE_PATH = 'D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/data/'
MODEL_PATH = 'D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/'

# Create directories
os.makedirs(MODEL_PATH, exist_ok=True)

# ============================
# LOAD DATA
# ============================
print("📁 Loading data...")

try:
    X_train_images = np.load(BASE_PATH + 'images/X_train_images.npy')
    y_train = np.load(BASE_PATH + 'images/y_train_images.npy')
    X_val_images = np.load(BASE_PATH + 'images/X_val_images.npy')
    y_val = np.load(BASE_PATH + 'images/y_val_images.npy')
    X_test_images = np.load(BASE_PATH + 'images/X_test_images.npy')
    y_test = np.load(BASE_PATH + 'images/y_test_images.npy')
    
    print(f"✅ Data loaded successfully!")
    print(f"Training samples: {X_train_images.shape[0]:,}")
    print(f"Validation samples: {X_val_images.shape[0]:,}")
    print(f"Test samples: {X_test_images.shape[0]:,}")
    print(f"Image shape: {X_train_images.shape[1:]}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# ============================
# PREPROCESS LABELS
# ============================
print("🏷️ Processing labels...")

# Create label mapping
unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
label_map = {label: i for i, label in enumerate(unique_labels)}
num_classes = len(label_map)

print(f"Number of classes: {num_classes}")
print(f"Classes: {list(unique_labels)}")

# Convert string labels to numerical
y_train_num = np.array([label_map[label] for label in y_train])
y_val_num = np.array([label_map[label] for label in y_val])
y_test_num = np.array([label_map[label] for label in y_test])

# Convert to one-hot encoding
y_train_one_hot = to_categorical(y_train_num, num_classes)
y_val_one_hot = to_categorical(y_val_num, num_classes)
y_test_one_hot = to_categorical(y_test_num, num_classes)

print("✅ Label preprocessing completed")

# ============================
# GPU-OPTIMIZED CNN MODEL
# ============================
def build_optimized_cnn_model(input_shape, num_classes):
    """Build CNN model optimized for AMD RX 6600 LE"""
    
    # Use device context for GPU
    device_context = '/GPU:0' if gpu_available else '/CPU:0'
    
    with tf.device(device_context):
        model = Sequential([
            # Block 1 - Optimized for GPU memory
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
            
            # Block 4 - Conditional based on GPU availability
            Conv2D(256, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            Conv2D(256, (3, 3), padding='same', activation='relu') if gpu_available else Conv2D(128, (3, 3), padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # Fully connected layers
            Flatten(),
            Dense(512 if gpu_available else 256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256 if gpu_available else 128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
    
    # Compile with GPU-optimized settings
    learning_rate = 0.0001 if gpu_available else 0.001
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================
# CALLBACKS FOR GPU TRAINING
# ============================
print("⚙️ Setting up callbacks...")

# Model checkpoint - save best model
checkpoint = ModelCheckpoint(
    MODEL_PATH + 'cnn_model_best.h5',
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1,
    save_weights_only=False
)

# Early stopping - prevent overfitting and save time
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=15 if gpu_available else 10,  # More patience with GPU
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

# Learning rate reduction
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=8 if gpu_available else 5,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# ============================
# TRAIN MODEL
# ============================
print(f"🚀 Training CNN model on {'GPU' if gpu_available else 'CPU'}...")
print("=" * 60)

# Build model
cnn_model = build_optimized_cnn_model(X_train_images.shape[1:], num_classes)

print("📊 Model Summary:")
print(f"Total Parameters: {cnn_model.count_params():,}")
trainable_params = sum([tf.size(w).numpy() for w in cnn_model.trainable_weights])
print(f"Trainable Parameters: {trainable_params:,}")

if gpu_available:
    print(f"🎮 GPU Memory Usage Estimate: ~{(cnn_model.count_params() * 4 * BATCH_SIZE) / (1024**3):.2f} GB")

# Training with optimized settings
start_time = time.time()

try:
    # Force GPU device placement if available
    device_context = '/GPU:0' if gpu_available else '/CPU:0'
    
    with tf.device(device_context):
        cnn_history = cnn_model.fit(
            X_train_images,
            y_train_one_hot,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val_images, y_val_one_hot),
            callbacks=callbacks,
            verbose=1,
            validation_freq=VALIDATION_FREQ
        )
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time/3600:.2f} hours")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    # Try with smaller batch size if GPU memory error
    if gpu_available and 'memory' in str(e).lower():
        print("💡 Retrying with smaller batch size...")
        BATCH_SIZE = 16
        cnn_history = cnn_model.fit(
            X_train_images,
            y_train_one_hot,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val_images, y_val_one_hot),
            callbacks=callbacks,
            verbose=1
        )

# ============================
# EVALUATION
# ============================
print("\n📊 Evaluating model...")

# Load best model
cnn_model = load_model(MODEL_PATH + 'cnn_model_best.h5')

# Evaluate on test set
test_loss, test_acc = cnn_model.evaluate(X_test_images, y_test_one_hot, verbose=0)
print(f"🎯 Test Accuracy: {test_acc:.4f}")
print(f"📉 Test Loss: {test_loss:.4f}")

# Detailed predictions
y_pred = np.argmax(cnn_model.predict(X_test_images, batch_size=BATCH_SIZE), axis=1)
y_true = np.argmax(y_test_one_hot, axis=1)

# Classification report
target_names = list(label_map.keys())
print("\n📈 Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# ============================
# VISUALIZATION
# ============================
print("📊 Creating visualizations...")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=target_names, yticklabels=target_names)
plt.title(f'CNN Confusion Matrix (Test Acc: {test_acc:.4f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(MODEL_PATH + 'cnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Training history
if 'cnn_history' in locals():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(cnn_history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(cnn_history.history['loss'], label='Training Loss')
    axes[1].plot(cnn_history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(MODEL_PATH + 'cnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================
# SAVE RESULTS
# ============================
print("💾 Saving results...")

# Save label mapping
with open(MODEL_PATH + 'label_map.pkl', 'wb') as f:
    pickle.dump(label_map, f)

# Save training info
training_info = {
    'model_type': 'CNN',
    'gpu_used': gpu_available,
    'num_gpus': num_gpus,
    'batch_size': BATCH_SIZE,
    'epochs_trained': len(cnn_history.history['accuracy']) if 'cnn_history' in locals() else 0,
    'test_accuracy': test_acc,
    'test_loss': test_loss,
    'total_parameters': cnn_model.count_params(),
    'training_time_hours': training_time/3600 if 'training_time' in locals() else 0,
    'device_info': f"AMD RX 6600 LE ({num_gpus} GPU(s))" if gpu_available else "CPU"
}

with open(MODEL_PATH + 'cnn_training_info.pkl', 'wb') as f:
    pickle.dump(training_info, f)

# ============================
# SUMMARY
# ============================
print("\n" + "=" * 60)
print("🎉 CNN TRAINING COMPLETED!")
print("=" * 60)
print(f"📊 Final Test Accuracy: {test_acc:.4f}")
print(f"🎮 Device Used: {'AMD RX 6600 LE GPU' if gpu_available else 'CPU'}")
print(f"⏱️ Training Time: {training_time/3600:.2f} hours" if 'training_time' in locals() else "N/A")
print(f"💾 Model saved to: {MODEL_PATH}cnn_model_best.h5")
print(f"📈 Visualizations saved to: {MODEL_PATH}")
print("\n🚀 Ready for next step: Train Landmark model!")
print("=" * 60)
