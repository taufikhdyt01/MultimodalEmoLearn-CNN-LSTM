import numpy as np
import tensorflow as tf
import os
import time
import pickle
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# AMD GPU SETUP FOR RX 6600 LE
# ============================
def setup_amd_gpu():
    """Setup AMD GPU with DirectML for optimal performance"""
    print("🎮 Setting up AMD GPU for Hybrid Fusion Training...")
    
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        dml_devices = tf.config.list_physical_devices('DML')
        total_devices = len(gpu_devices) + len(dml_devices)
        
        if total_devices > 0:
            print(f"✅ Found {total_devices} GPU device(s)")
            
            # Configure GPU memory growth
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
            
            # Test GPU functionality
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test_tensor, test_tensor)
                print("✅ GPU computation test passed")
            
            # Enable optimizations
            tf.config.optimizer.set_jit(True)
            print("✅ XLA compilation enabled")
            
            return True, total_devices
        else:
            print("⚠️ No GPU detected, using CPU")
            return False, 0
    except Exception as e:
        print(f"⚠️ GPU setup failed: {e}")
        return False, 0

# Setup GPU
gpu_available, num_gpus = setup_amd_gpu()

# ============================
# OPTIMIZED CONFIGURATIONS
# ============================

# Hybrid models are most memory intensive
if gpu_available:
    BATCH_SIZE = 16  # Conservative for dual input model
    EPOCHS = 80
    LEARNING_RATE = 0.0001
    print(f"🚀 GPU Mode: Batch size {BATCH_SIZE}, {EPOCHS} epochs")
else:
    BATCH_SIZE = 8
    EPOCHS = 40
    LEARNING_RATE = 0.001
    print(f"💻 CPU Mode: Batch size {BATCH_SIZE}, {EPOCHS} epochs")

# Paths
BASE_PATH = 'D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/data/'
MODEL_PATH = 'D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/'

# ============================
# LOAD DATA
# ============================
print("📁 Loading multimodal data...")

try:
    # Load image data
    X_train_images = np.load(BASE_PATH + 'images/X_train_images.npy')
    X_val_images = np.load(BASE_PATH + 'images/X_val_images.npy')
    X_test_images = np.load(BASE_PATH + 'images/X_test_images.npy')
    
    # Load landmark data
    X_train_landmarks = np.load(BASE_PATH + 'landmarks/X_train_landmarks.npy')
    X_val_landmarks = np.load(BASE_PATH + 'landmarks/X_val_landmarks.npy')
    X_test_landmarks = np.load(BASE_PATH + 'landmarks/X_test_landmarks.npy')
    
    # Load labels
    y_train = np.load(BASE_PATH + 'images/y_train_images.npy')
    y_val = np.load(BASE_PATH + 'images/y_val_images.npy')
    y_test = np.load(BASE_PATH + 'images/y_test_images.npy')
    
    print(f"✅ Data loaded successfully!")
    print(f"Training samples: {X_train_images.shape[0]:,}")
    print(f"Validation samples: {X_val_images.shape[0]:,}")
    print(f"Test samples: {X_test_images.shape[0]:,}")
    print(f"Image shape: {X_train_images.shape[1:]}")
    print(f"Landmark features: {X_train_landmarks.shape[1]}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    exit(1)

# ============================
# LOAD LABEL MAPPING
# ============================
print("🏷️ Loading label mapping...")

try:
    with open(MODEL_PATH + 'label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    
    num_classes = len(label_map)
    target_names = list(label_map.keys())
    
    print(f"Classes: {target_names}")
    print(f"Number of classes: {num_classes}")
    
except FileNotFoundError:
    print("⚠️ Label mapping not found, creating new one...")
    unique_labels = np.unique(np.concatenate([y_train, y_val, y_test]))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    num_classes = len(label_map)
    target_names = list(label_map.keys())

# Convert labels
y_train_num = np.array([label_map[label] for label in y_train])
y_val_num = np.array([label_map[label] for label in y_val])
y_test_num = np.array([label_map[label] for label in y_test])

y_train_onehot = to_categorical(y_train_num, num_classes)
y_val_onehot = to_categorical(y_val_num, num_classes)
y_test_onehot = to_categorical(y_test_num, num_classes)

print("✅ Label preprocessing completed")

# ============================
# GPU-OPTIMIZED HYBRID FUSION MODEL
# ============================
def build_optimized_hybrid_model(input_shape_image, input_shape_landmark, num_classes):
    """Build hybrid fusion model optimized for AMD GPU"""
    
    device_context = '/GPU:0' if gpu_available else '/CPU:0'
    
    with tf.device(device_context):
        # Image Stream (CNN)
        image_input = Input(shape=input_shape_image, name='image_input')
        
        # CNN layers - optimized for GPU/CPU
        if gpu_available:
            # More complex CNN with GPU
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
            x_img = Dense(512, activation='relu')(x_img)
            x_img = BatchNormalization()(x_img)
            x_img = Dropout(0.5)(x_img)
            x_img = Dense(256, activation='relu')(x_img)
            x_img = BatchNormalization()(x_img)
            x_img = Dropout(0.5)(x_img)
        else:
            # Simpler CNN for CPU
            x_img = Conv2D(32, (3, 3), activation='relu')(image_input)
            x_img = MaxPooling2D(pool_size=(2, 2))(x_img)
            x_img = Conv2D(64, (3, 3), activation='relu')(x_img)
            x_img = MaxPooling2D(pool_size=(2, 2))(x_img)
            x_img = Conv2D(128, (3, 3), activation='relu')(x_img)
            x_img = MaxPooling2D(pool_size=(2, 2))(x_img)
            x_img = Flatten()(x_img)
            x_img = Dense(256, activation='relu')(x_img)
            x_img = Dropout(0.5)(x_img)
        
        # Landmark Stream
        landmark_input = Input(shape=(input_shape_landmark,), name='landmark_input')
        
        if gpu_available:
            # More complex landmark stream with GPU
            x_lm = Dense(256, activation='relu')(landmark_input)
            x_lm = BatchNormalization()(x_lm)
            x_lm = Dropout(0.3)(x_lm)
            
            x_lm = Dense(512, activation='relu')(x_lm)
            x_lm = BatchNormalization()(x_lm)
            x_lm = Dropout(0.4)(x_lm)
            
            x_lm = Dense(256, activation='relu')(x_lm)
            x_lm = BatchNormalization()(x_lm)
            x_lm = Dropout(0.3)(x_lm)
            
            x_lm = Dense(128, activation='relu')(x_lm)
            x_lm = BatchNormalization()(x_lm)
            x_lm = Dropout(0.3)(x_lm)
        else:
            # Simpler landmark stream for CPU
            x_lm = Dense(128, activation='relu')(landmark_input)
            x_lm = Dropout(0.3)(x_lm)
            x_lm = Dense(256, activation='relu')(x_lm)
            x_lm = Dropout(0.3)(x_lm)
            x_lm = Dense(128, activation='relu')(x_lm)
            x_lm = Dropout(0.3)(x_lm)
        
        # Fusion layer - combine both streams
        fusion = concatenate([x_img, x_lm], name='fusion_layer')
        
        # Post-fusion layers
        if gpu_available:
            x = Dense(512, activation='relu')(fusion)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            
            x = Dense(256, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
        else:
            x = Dense(256, activation='relu')(fusion)
            x = Dropout(0.5)(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
        
        # Output layer
        output = Dense(num_classes, activation='softmax', name='output')(x)
        
        # Build model
        model = Model(inputs=[image_input, landmark_input], outputs=output)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================
# CALLBACKS
# ============================
print("⚙️ Setting up callbacks...")

checkpoint = ModelCheckpoint(
    MODEL_PATH + 'hybrid_fusion_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=25 if gpu_available else 15,
    restore_best_weights=True,
    verbose=1,
    mode='max'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=12 if gpu_available else 8,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# ============================
# TRAIN MODEL
# ============================
print(f"🚀 Training Hybrid Fusion model on {'GPU' if gpu_available else 'CPU'}...")
print("=" * 60)

# Build model
hybrid_model = build_optimized_hybrid_model(
    X_train_images.shape[1:], 
    X_train_landmarks.shape[1],
    num_classes
)

print("📊 Model Summary:")
print(f"Total Parameters: {hybrid_model.count_params():,}")

# Calculate trainable parameters correctly for TF 2.10
trainable_params = sum([tf.size(w).numpy() for w in hybrid_model.trainable_weights])
print(f"Trainable Parameters: {trainable_params:,}")

if gpu_available:
    print(f"🎮 GPU Memory Usage Estimate: ~{(hybrid_model.count_params() * 4 * BATCH_SIZE) / (1024**3):.2f} GB")

# Training
start_time = time.time()

try:
    device_context = '/GPU:0' if gpu_available else '/CPU:0'
    
    with tf.device(device_context):
        hybrid_history = hybrid_model.fit(
            [X_train_images, X_train_landmarks],
            y_train_onehot,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=([X_val_images, X_val_landmarks], y_val_onehot),
            callbacks=callbacks,
            verbose=1
        )
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time/3600:.2f} hours")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    if gpu_available and 'memory' in str(e).lower():
        print("💡 Retrying with smaller batch size...")
        BATCH_SIZE = 8
        hybrid_history = hybrid_model.fit(
            [X_train_images, X_train_landmarks],
            y_train_onehot,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=([X_val_images, X_val_landmarks], y_val_onehot),
            callbacks=callbacks,
            verbose=1
        )

# ============================
# EVALUATION
# ============================
print("\n📊 Evaluating model...")

# Load best model
hybrid_model = load_model(MODEL_PATH + 'hybrid_fusion_model_best.h5')

# Evaluate on test set
test_loss, test_acc = hybrid_model.evaluate([X_test_images, X_test_landmarks], y_test_onehot, verbose=0)
print(f"🎯 Test Accuracy: {test_acc:.4f}")
print(f"📉 Test Loss: {test_loss:.4f}")

# Detailed predictions
y_pred = np.argmax(hybrid_model.predict([X_test_images, X_test_landmarks], batch_size=BATCH_SIZE), axis=1)
y_true = np.argmax(y_test_onehot, axis=1)

# Classification report
print("\n📈 Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# ============================
# VISUALIZATION
# ============================
print("📊 Creating visualizations...")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
           xticklabels=target_names, yticklabels=target_names)
plt.title(f'Hybrid Fusion Confusion Matrix (Test Acc: {test_acc:.4f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(MODEL_PATH + 'hybrid_fusion_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Training history
if 'hybrid_history' in locals():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(hybrid_history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(hybrid_history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Hybrid Fusion Model Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(hybrid_history.history['loss'], label='Training Loss')
    axes[1].plot(hybrid_history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Hybrid Fusion Model Loss')
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(MODEL_PATH + 'hybrid_fusion_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================
# SAVE RESULTS
# ============================
print("💾 Saving results...")

# Save training info
training_info = {
    'model_type': 'Hybrid_Fusion',
    'gpu_used': gpu_available,
    'num_gpus': num_gpus,
    'batch_size': BATCH_SIZE,
    'epochs_trained': len(hybrid_history.history['accuracy']) if 'hybrid_history' in locals() else 0,
    'test_accuracy': test_acc,
    'test_loss': test_loss,
    'total_parameters': hybrid_model.count_params(),
    'training_time_hours': training_time/3600 if 'training_time' in locals() else 0,
    'device_info': f"AMD GPU ({num_gpus} device(s))" if gpu_available else "CPU"
}

with open(MODEL_PATH + 'hybrid_fusion_training_info.pkl', 'wb') as f:
    pickle.dump(training_info, f)

# ============================
# SUMMARY
# ============================
print("\n" + "=" * 60)
print("🎉 HYBRID FUSION TRAINING COMPLETED!")
print("=" * 60)
print(f"📊 Final Test Accuracy: {test_acc:.4f}")
print(f"🎮 Device Used: {'AMD GPU' if gpu_available else 'CPU'}")
print(f"⏱️ Training Time: {training_time/3600:.2f} hours" if 'training_time' in locals() else "N/A")
print(f"💾 Model saved to: {MODEL_PATH}hybrid_fusion_model_best.h5")
print(f"📈 Visualizations saved to: {MODEL_PATH}")
print("\n🚀 Ready for next step: Train Late Fusion!")
print("=" * 60)
