import numpy as np
import tensorflow as tf
import os
import time
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ============================
# AMD GPU SETUP FOR RX 6600 LE
# ============================
def setup_amd_gpu():
    """Setup AMD GPU with DirectML for optimal performance"""
    print("🎮 Setting up AMD GPU for Model Comparison...")
    
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

print(f"📊 Model Comparison using {'GPU' if gpu_available else 'CPU'} inference")

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
    print("❌ Label mapping not found!")
    exit(1)

# Convert labels
y_test_num = np.array([label_map[label] for label in y_test])
y_true = y_test_num

# ============================
# LOAD ALL TRAINED MODELS
# ============================
print("🔄 Loading all trained models...")

models = {}
model_files = {
    'CNN': 'cnn_model_best.h5',
    'Landmark': 'landmark_model_best.h5',
    'Hybrid Fusion': 'hybrid_fusion_model_best.h5'
}

device_context = '/GPU:0' if gpu_available else '/CPU:0'

for model_name, filename in model_files.items():
    try:
        print(f"📱 Loading {model_name} model...")
        with tf.device(device_context):
            model = load_model(MODEL_PATH + filename)
            models[model_name] = model
            print(f"✅ {model_name}: {model.count_params():,} parameters")
    except FileNotFoundError:
        print(f"⚠️ {model_name} model not found, skipping...")

# Load late fusion configuration
try:
    with open(MODEL_PATH + 'late_fusion_ensemble.pkl', 'rb') as f:
        late_fusion_config = pickle.load(f)
    print(f"✅ Late fusion config loaded")
except FileNotFoundError:
    print("⚠️ Late fusion config not found")
    late_fusion_config = None

# ============================
# INFERENCE AND EVALUATION
# ============================
print("\n🚀 Running comprehensive model evaluation...")

def late_fusion_weighted(cnn_probs, landmark_probs, weight_cnn=0.6):
    """Combine predictions from CNN and Landmark models"""
    weight_landmark = 1 - weight_cnn
    return (weight_cnn * cnn_probs) + (weight_landmark * landmark_probs)

# Store results
results = {}
inference_times = {}

# Evaluate each model
with tf.device(device_context):
    
    # 1. CNN Model
    if 'CNN' in models:
        print("📱 Evaluating CNN model...")
        start_time = time.time()
        cnn_probs = models['CNN'].predict(X_test_images, batch_size=BATCH_SIZE, verbose=0)
        cnn_pred = np.argmax(cnn_probs, axis=1)
        inference_times['CNN'] = time.time() - start_time
        results['CNN'] = cnn_pred
    
    # 2. Landmark Model
    if 'Landmark' in models:
        print("🎯 Evaluating Landmark model...")
        start_time = time.time()
        landmark_probs = models['Landmark'].predict(X_test_landmarks, batch_size=BATCH_SIZE, verbose=0)
        landmark_pred = np.argmax(landmark_probs, axis=1)
        inference_times['Landmark'] = time.time() - start_time
        results['Landmark'] = landmark_pred
    
    # 3. Late Fusion
    if 'CNN' in models and 'Landmark' in models and late_fusion_config:
        print("⚖️ Evaluating Late Fusion...")
        start_time = time.time()
        best_weight = late_fusion_config['best_cnn_weight']
        late_fusion_probs = late_fusion_weighted(cnn_probs, landmark_probs, best_weight)
        late_fusion_pred = np.argmax(late_fusion_probs, axis=1)
        inference_times['Late Fusion'] = inference_times['CNN'] + inference_times['Landmark']
        results['Late Fusion'] = late_fusion_pred
    
    # 4. Hybrid Fusion Model
    if 'Hybrid Fusion' in models:
        print("🔗 Evaluating Hybrid Fusion model...")
        start_time = time.time()
        hybrid_probs = models['Hybrid Fusion'].predict([X_test_images, X_test_landmarks], 
                                                       batch_size=BATCH_SIZE, verbose=0)
        hybrid_pred = np.argmax(hybrid_probs, axis=1)
        inference_times['Hybrid Fusion'] = time.time() - start_time
        results['Hybrid Fusion'] = hybrid_pred

# ============================
# CALCULATE METRICS
# ============================
print("\n📊 Calculating performance metrics...")

model_performance = {}

for model_name, predictions in results.items():
    # Basic metrics
    accuracy = accuracy_score(y_true, predictions)
    f1_macro = f1_score(y_true, predictions, average='macro')
    f1_weighted = f1_score(y_true, predictions, average='weighted')
    precision_macro = precision_score(y_true, predictions, average='macro')
    recall_macro = recall_score(y_true, predictions, average='macro')
    
    # Per-class F1 scores
    f1_per_class = f1_score(y_true, predictions, average=None)
    
    model_performance[model_name] = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_per_class': f1_per_class,
        'inference_time': inference_times.get(model_name, 0),
        'parameters': models[model_name].count_params() if model_name in models else 0
    }

# ============================
# RESULTS SUMMARY
# ============================
print("\n" + "=" * 80)
print("📈 COMPREHENSIVE MODEL COMPARISON RESULTS")
print("=" * 80)

# Create results DataFrame
df_results = pd.DataFrame({
    'Model': list(model_performance.keys()),
    'Accuracy': [perf['accuracy'] for perf in model_performance.values()],
    'F1-Score (Macro)': [perf['f1_macro'] for perf in model_performance.values()],
    'F1-Score (Weighted)': [perf['f1_weighted'] for perf in model_performance.values()],
    'Precision (Macro)': [perf['precision_macro'] for perf in model_performance.values()],
    'Recall (Macro)': [perf['recall_macro'] for perf in model_performance.values()],
    'Inference Time (s)': [perf['inference_time'] for perf in model_performance.values()],
    'Parameters (M)': [perf['parameters']/1e6 for perf in model_performance.values()]
})

# Sort by accuracy
df_results = df_results.sort_values('Accuracy', ascending=False)

print("🏆 RANKING BY ACCURACY:")
for idx, row in df_results.iterrows():
    print(f"{row.name+1}. {row['Model']}: {row['Accuracy']:.4f}")

print(f"\n📊 DETAILED METRICS:")
print(df_results.round(4).to_string(index=False))

# Best model
best_model = df_results.iloc[0]['Model']
best_accuracy = df_results.iloc[0]['Accuracy']
print(f"\n🥇 BEST MODEL: {best_model} (Accuracy: {best_accuracy:.4f})")

# ============================
# VISUALIZATIONS
# ============================
print("\n📊 Creating comprehensive visualizations...")

# Set up the plotting style
plt.style.use('default')
sns.set_palette("husl")

# 1. Overall Performance Comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Accuracy comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(df_results['Model'], df_results['Accuracy'], 
               color=['skyblue', 'lightgreen', 'gold', 'lightcoral'][:len(df_results)])
ax1.set_title('Model Accuracy Comparison')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, max(df_results['Accuracy']) + 0.05)
for bar, acc in zip(bars1, df_results['Accuracy']):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
ax1.tick_params(axis='x', rotation=45)

# F1-Score comparison
ax2 = axes[0, 1]
ax2.bar(df_results['Model'], df_results['F1-Score (Macro)'], 
        color=['skyblue', 'lightgreen', 'gold', 'lightcoral'][:len(df_results)])
ax2.set_title('F1-Score (Macro) Comparison')
ax2.set_ylabel('F1-Score')
ax2.tick_params(axis='x', rotation=45)

# Inference time comparison
ax3 = axes[0, 2]
ax3.bar(df_results['Model'], df_results['Inference Time (s)'], 
        color=['skyblue', 'lightgreen', 'gold', 'lightcoral'][:len(df_results)])
ax3.set_title('Inference Time Comparison')
ax3.set_ylabel('Time (seconds)')
ax3.tick_params(axis='x', rotation=45)

# Parameters comparison
ax4 = axes[1, 0]
ax4.bar(df_results['Model'], df_results['Parameters (M)'], 
        color=['skyblue', 'lightgreen', 'gold', 'lightcoral'][:len(df_results)])
ax4.set_title('Model Complexity (Parameters)')
ax4.set_ylabel('Parameters (Millions)')
ax4.tick_params(axis='x', rotation=45)

# Precision vs Recall
ax5 = axes[1, 1]
scatter = ax5.scatter(df_results['Precision (Macro)'], df_results['Recall (Macro)'], 
                     s=200, alpha=0.7, c=range(len(df_results)), cmap='viridis')
for i, model in enumerate(df_results['Model']):
    ax5.annotate(model, (df_results.iloc[i]['Precision (Macro)'], 
                        df_results.iloc[i]['Recall (Macro)']),
                xytext=(5, 5), textcoords='offset points', fontsize=10)
ax5.set_xlabel('Precision (Macro)')
ax5.set_ylabel('Recall (Macro)')
ax5.set_title('Precision vs Recall')
ax5.grid(True, alpha=0.3)

# Performance radar chart
ax6 = axes[1, 2]
metrics = ['Accuracy', 'F1-Score (Macro)', 'Precision (Macro)', 'Recall (Macro)']
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for i, model in enumerate(df_results['Model']):
    values = [df_results.iloc[i]['Accuracy'], 
              df_results.iloc[i]['F1-Score (Macro)'],
              df_results.iloc[i]['Precision (Macro)'], 
              df_results.iloc[i]['Recall (Macro)']]
    values += values[:1]  # Complete the circle
    
    ax6.plot(angles, values, 'o-', linewidth=2, label=model)
    ax6.fill(angles, values, alpha=0.25)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(metrics)
ax6.set_ylim(0, 1)
ax6.set_title('Performance Radar Chart')
ax6.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

plt.tight_layout()
plt.savefig(MODEL_PATH + 'comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Per-class F1 Score Comparison
if len(results) > 0:
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(target_names))
    width = 0.8 / len(results)
    
    for i, (model_name, perf) in enumerate(model_performance.items()):
        f1_scores = perf['f1_per_class']
        offset = (i - len(results)/2 + 0.5) * width
        ax.bar(x + offset, f1_scores, width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Emotion Classes')
    ax.set_ylabel('F1-Score')
    ax.set_title('Per-Class F1-Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(target_names, rotation=45)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(MODEL_PATH + 'per_class_f1_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3. Confusion Matrices for All Models
n_models = len(results)
if n_models > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    cmaps = ['Blues', 'Greens', 'Oranges', 'Purples']
    
    for i, (model_name, predictions) in enumerate(results.items()):
        if i >= 4:  # Max 4 subplots
            break
            
        cm = confusion_matrix(y_true, predictions)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[i], 
                   xticklabels=target_names, yticklabels=target_names,
                   ax=axes[i])
        
        accuracy = model_performance[model_name]['accuracy']
        axes[i].set_title(f'{model_name} Confusion Matrix\\n(Accuracy: {accuracy:.4f})')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
    
    # Hide empty subplots
    for j in range(i+1, 4):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(MODEL_PATH + 'all_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

# ============================
# SAVE COMPREHENSIVE RESULTS
# ============================
print("💾 Saving comprehensive comparison results...")

# Prepare final results
comparison_results = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'device_info': f"AMD GPU ({num_gpus} device(s))" if gpu_available else "CPU",
    'test_samples': len(y_true),
    'num_classes': num_classes,
    'class_names': target_names,
    'model_rankings': df_results.to_dict('records'),
    'detailed_performance': model_performance,
    'best_model': {
        'name': best_model,
        'accuracy': best_accuracy,
        'all_metrics': model_performance[best_model]
    },
    'confusion_matrices': {
        model_name: confusion_matrix(y_true, predictions).tolist()
        for model_name, predictions in results.items()
    }
}

# Save to pickle
with open(MODEL_PATH + 'comprehensive_model_comparison.pkl', 'wb') as f:
    pickle.dump(comparison_results, f)

# Save to CSV for easy viewing
df_results.to_csv(MODEL_PATH + 'model_comparison_results.csv', index=False)

# Generate text report
report_lines = [
    "=" * 80,
    "MULTIMODAL EMOTION RECOGNITION - FINAL MODEL COMPARISON",
    "=" * 80,
    f"Date: {comparison_results['timestamp']}",
    f"Device: {comparison_results['device_info']}",
    f"Test Samples: {comparison_results['test_samples']:,}",
    f"Classes: {', '.join(target_names)}",
    "",
    "RANKING BY ACCURACY:",
]

for i, row in df_results.iterrows():
    report_lines.append(f"{i+1}. {row['Model']}: {row['Accuracy']:.4f}")

report_lines.extend([
    "",
    f"🥇 BEST MODEL: {best_model}",
    f"🎯 BEST ACCURACY: {best_accuracy:.4f}",
    "",
    "DETAILED METRICS:",
    str(df_results.round(4)),
    "",
    "=" * 80
])

# Save text report
with open(MODEL_PATH + 'model_comparison_report.txt', 'w') as f:
    f.write('\\n'.join(report_lines))

# ============================
# FINAL SUMMARY
# ============================
print("\n" + "=" * 80)
print("🎉 COMPREHENSIVE MODEL COMPARISON COMPLETED!")
print("=" * 80)
print(f"📊 Models Evaluated: {len(results)}")
print(f"🥇 Best Model: {best_model} (Accuracy: {best_accuracy:.4f})")
print(f"🎮 Device Used: {'AMD GPU' if gpu_available else 'CPU'}")
print(f"⏱️ Total Evaluation Time: {sum(inference_times.values()):.2f} seconds")
print(f"\n💾 Results saved to:")
print(f"   - {MODEL_PATH}comprehensive_model_comparison.pkl")
print(f"   - {MODEL_PATH}model_comparison_results.csv") 
print(f"   - {MODEL_PATH}model_comparison_report.txt")
print(f"📈 Visualizations saved to:")
print(f"   - {MODEL_PATH}comprehensive_model_comparison.png")
print(f"   - {MODEL_PATH}per_class_f1_comparison.png")
print(f"   - {MODEL_PATH}all_confusion_matrices.png")
print(f"\n🎊 MULTIMODAL EMOTION RECOGNITION PROJECT COMPLETED!")
print("=" * 80)
