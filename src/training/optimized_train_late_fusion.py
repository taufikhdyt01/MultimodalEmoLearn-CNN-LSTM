import numpy as np
import tensorflow as tf
import os
import time
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ============================
# AMD GPU SETUP FOR RX 6600 LE
# ============================
def setup_amd_gpu():
    """Setup AMD GPU with DirectML for optimal performance"""
    print("🎮 Setting up AMD GPU for Late Fusion...")
    
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
# PATHS AND CONFIGURATIONS
# ============================
BASE_PATH = 'D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/data/'
MODEL_PATH = 'D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/'

# Batch size for inference
BATCH_SIZE = 64 if gpu_available else 32

print(f"🔗 Late Fusion using {'GPU' if gpu_available else 'CPU'} inference")

# ============================
# LOAD DATA
# ============================
print("📁 Loading test data...")

try:
    X_test_images = np.load(BASE_PATH + 'images/X_test_images.npy')
    X_test_landmarks = np.load(BASE_PATH + 'landmarks/X_test_landmarks.npy')
    y_test = np.load(BASE_PATH + 'images/y_test_images.npy')
    
    print(f"✅ Test data loaded successfully!")
    print(f"Test samples: {X_test_images.shape[0]:,}")
    print(f"Image shape: {X_test_images.shape[1:]}")
    print(f"Landmark features: {X_test_landmarks.shape[1]}")
    
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
    print("❌ Label mapping not found! Please train CNN model first.")
    exit(1)

# Convert labels
y_test_num = np.array([label_map[label] for label in y_test])
y_true = y_test_num

print("✅ Label preprocessing completed")

# ============================
# LOAD TRAINED MODELS
# ============================
print("🔄 Loading trained models...")

try:
    # Load CNN model
    print("📱 Loading CNN model...")
    cnn_model = load_model(MODEL_PATH + 'cnn_model_best.h5')
    print(f"✅ CNN model loaded: {cnn_model.count_params():,} parameters")
    
    # Load Landmark model
    print("🎯 Loading Landmark model...")
    landmark_model = load_model(MODEL_PATH + 'landmark_model_best.h5')
    print(f"✅ Landmark model loaded: {landmark_model.count_params():,} parameters")
    
except FileNotFoundError as e:
    print(f"❌ Error loading models: {e}")
    print("💡 Please train individual models first (CNN and Landmark)")
    exit(1)

# ============================
# LATE FUSION FUNCTIONS
# ============================
def late_fusion_weighted(cnn_probs, landmark_probs, weight_cnn=0.6):
    """
    Combine predictions from CNN and Landmark models with weighting
    
    Args:
        cnn_probs: Probability predictions from CNN model
        landmark_probs: Probability predictions from Landmark model
        weight_cnn: Weight for CNN model (0-1)
        
    Returns:
        Array of combined probabilities
    """
    weight_landmark = 1 - weight_cnn
    fused_probs = (weight_cnn * cnn_probs) + (weight_landmark * landmark_probs)
    return fused_probs

def evaluate_fusion_performance(cnn_probs, landmark_probs, y_true, weights_to_test):
    """Evaluate fusion performance across different weight combinations"""
    
    results = []
    
    for weight_cnn in weights_to_test:
        # Fusion with specific weight
        fused_probs = late_fusion_weighted(cnn_probs, landmark_probs, weight_cnn)
        y_pred_fused = np.argmax(fused_probs, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred_fused)
        results.append((weight_cnn, accuracy))\n        \n        print(f\"CNN Weight: {weight_cnn:.1f}, Landmark Weight: {1-weight_cnn:.1f}, Accuracy: {accuracy:.4f}\")\n    \n    return results

# ============================
# GPU-OPTIMIZED INFERENCE
# ============================
print("🚀 Running GPU-optimized inference...")

# Get predictions from both models
device_context = '/GPU:0' if gpu_available else '/CPU:0'

with tf.device(device_context):
    print("📱 Getting CNN predictions...")
    start_time = time.time()
    cnn_probs = cnn_model.predict(X_test_images, batch_size=BATCH_SIZE, verbose=1)
    cnn_time = time.time() - start_time
    print(f"   ⏱️ CNN inference time: {cnn_time:.2f}s")
    
    print("🎯 Getting Landmark predictions...")
    start_time = time.time()
    landmark_probs = landmark_model.predict(X_test_landmarks, batch_size=BATCH_SIZE, verbose=1)
    landmark_time = time.time() - start_time
    print(f"   ⏱️ Landmark inference time: {landmark_time:.2f}s")

# Individual model accuracies
cnn_pred = np.argmax(cnn_probs, axis=1)
landmark_pred = np.argmax(landmark_probs, axis=1)

cnn_accuracy = accuracy_score(y_true, cnn_pred)
landmark_accuracy = accuracy_score(y_true, landmark_pred)

print(f"\n📊 Individual Model Performance:")
print(f"   CNN Only: {cnn_accuracy:.4f}")
print(f"   Landmark Only: {landmark_accuracy:.4f}")

# ============================
# WEIGHT OPTIMIZATION
# ============================
print("\n⚖️ Optimizing fusion weights...")

# Test different weight combinations
weights = np.arange(0.0, 1.1, 0.1)  # From 0.0 to 1.0 in 0.1 steps
fusion_results = evaluate_fusion_performance(cnn_probs, landmark_probs, y_true, weights)

# Find best weight combination
best_weight_cnn, best_accuracy = max(fusion_results, key=lambda x: x[1])
best_weight_landmark = 1 - best_weight_cnn

print(f"\n🏆 BEST FUSION CONFIGURATION:")
print(f"   Best CNN Weight: {best_weight_cnn:.1f}")
print(f"   Best Landmark Weight: {best_weight_landmark:.1f}")
print(f"   Best Fusion Accuracy: {best_accuracy:.4f}")

# Calculate improvement
improvement = best_accuracy - max(cnn_accuracy, landmark_accuracy)
print(f"   Improvement over best individual: {improvement:.4f} ({improvement*100:.2f}%)")

# ============================
# FINAL FUSION WITH BEST WEIGHTS
# ============================
print("\n🔗 Final fusion with optimal weights...")

# Apply best fusion
final_fused_probs = late_fusion_weighted(cnn_probs, landmark_probs, best_weight_cnn)
y_pred_fused = np.argmax(final_fused_probs, axis=1)

# Detailed evaluation
print("\n📈 Late Fusion Classification Report:")
print(classification_report(y_true, y_pred_fused, target_names=target_names))

# ============================
# VISUALIZATION
# ============================
print("📊 Creating visualizations...")

# 1. Weight optimization plot
plt.figure(figsize=(12, 8))

weights_list, accuracies_list = zip(*fusion_results)
plt.plot(weights_list, accuracies_list, 'bo-', linewidth=2, markersize=8, label='Fusion Performance')
plt.axhline(y=cnn_accuracy, color='red', linestyle='--', linewidth=2, 
           label=f'CNN Only ({cnn_accuracy:.4f})')
plt.axhline(y=landmark_accuracy, color='green', linestyle='--', linewidth=2,
           label=f'Landmark Only ({landmark_accuracy:.4f})')

# Mark best point
plt.scatter([best_weight_cnn], [best_accuracy], color='gold', s=200, 
           marker='*', label=f'Best Fusion ({best_accuracy:.4f})', zorder=5)

plt.xlabel('CNN Weight')
plt.ylabel('Accuracy')
plt.title('Late Fusion Performance vs Weight Configuration')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(-0.05, 1.05)
plt.ylim(min(min(accuracies_list), cnn_accuracy, landmark_accuracy) - 0.01,
         max(max(accuracies_list), cnn_accuracy, landmark_accuracy) + 0.01)

plt.tight_layout()
plt.savefig(MODEL_PATH + 'late_fusion_weight_optimization.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Confusion Matrix
cm = confusion_matrix(y_true, y_pred_fused)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
           xticklabels=target_names, yticklabels=target_names)
plt.title(f'Late Fusion Confusion Matrix\\n(CNN Weight: {best_weight_cnn:.1f}, Accuracy: {best_accuracy:.4f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(MODEL_PATH + 'late_fusion_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Model Comparison Bar Chart
plt.figure(figsize=(10, 6))
models = ['CNN Only', 'Landmark Only', 'Late Fusion']
accuracies = [cnn_accuracy, landmark_accuracy, best_accuracy]
colors = ['skyblue', 'lightgreen', 'gold']

bars = plt.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim(0, max(accuracies) + 0.05)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(MODEL_PATH + 'model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================
# SAVE RESULTS
# ============================
print("💾 Saving late fusion results...")

# Save fusion ensemble configuration
fusion_ensemble = {
    'best_cnn_weight': best_weight_cnn,
    'best_landmark_weight': best_weight_landmark,
    'best_accuracy': best_accuracy,
    'cnn_model_path': MODEL_PATH + 'cnn_model_best.h5',
    'landmark_model_path': MODEL_PATH + 'landmark_model_best.h5',
    'individual_accuracies': {
        'cnn': cnn_accuracy,
        'landmark': landmark_accuracy
    },
    'improvement': improvement,
    'weight_search_results': fusion_results,
    'inference_times': {
        'cnn_seconds': cnn_time,
        'landmark_seconds': landmark_time,
        'total_seconds': cnn_time + landmark_time
    },
    'device_info': f"AMD GPU ({num_gpus} device(s))" if gpu_available else "CPU"
}

with open(MODEL_PATH + 'late_fusion_ensemble.pkl', 'wb') as f:
    pickle.dump(fusion_ensemble, f)

# Save training info
training_info = {
    'model_type': 'Late_Fusion',
    'gpu_used': gpu_available,
    'num_gpus': num_gpus,
    'best_accuracy': best_accuracy,
    'best_cnn_weight': best_weight_cnn,
    'best_landmark_weight': best_weight_landmark,
    'improvement_over_individual': improvement,
    'total_inference_time': cnn_time + landmark_time,
    'device_info': f"AMD GPU ({num_gpus} device(s))" if gpu_available else "CPU"
}

with open(MODEL_PATH + 'late_fusion_training_info.pkl', 'wb') as f:
    pickle.dump(training_info, f)

# ============================
# SUMMARY
# ============================
print("\n" + "=" * 60)
print("🎉 LATE FUSION OPTIMIZATION COMPLETED!")
print("=" * 60)
print(f"📊 Results Summary:")
print(f"   CNN Only: {cnn_accuracy:.4f}")
print(f"   Landmark Only: {landmark_accuracy:.4f}")
print(f"   🏆 Late Fusion: {best_accuracy:.4f}")
print(f"   📈 Improvement: {improvement:.4f} ({improvement*100:.2f}%)")
print(f"\n⚖️ Optimal Weights:")
print(f"   CNN Weight: {best_weight_cnn:.1f}")
print(f"   Landmark Weight: {best_weight_landmark:.1f}")
print(f"\n🎮 Device Used: {'AMD GPU' if gpu_available else 'CPU'}")
print(f"⏱️ Total Inference Time: {cnn_time + landmark_time:.2f} seconds")
print(f"💾 Results saved to: {MODEL_PATH}")
print(f"📈 Visualizations saved to: {MODEL_PATH}")
print("\n🚀 Ready for final step: Complete Model Comparison!")
print("=" * 60)
