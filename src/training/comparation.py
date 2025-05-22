# Buat visualisasi perbandingan model
models = ['CNN', 'Landmark', 'Late Fusion', 'Hybrid Fusion']

# Load hasil evaluasi
cnn_model = load_model('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/cnn_model_best.h5')
landmark_model = load_model('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/landmark_model_best.h5')
hybrid_model = load_model('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/hybrid_fusion_model_best.h5')

# Load weight late fusion terbaik
with open('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/late_fusion_ensemble.pkl', 'rb') as f:
    late_fusion_data = pickle.load(f)
best_weight = late_fusion_data['best_weight']

# Prediksi
cnn_probs = cnn_model.predict(X_test_images)
landmark_probs = landmark_model.predict(X_test_landmarks)
hybrid_probs = hybrid_model.predict([X_test_images, X_test_landmarks])
late_fusion_probs = late_fusion_weighted(cnn_probs, landmark_probs, best_weight)

# Convert ke prediksi kelas
cnn_pred = np.argmax(cnn_probs, axis=1)
landmark_pred = np.argmax(landmark_probs, axis=1)
hybrid_pred = np.argmax(hybrid_probs, axis=1)
late_fusion_pred = np.argmax(late_fusion_probs, axis=1)

# Hitung akurasi
accuracies = [
    np.mean(cnn_pred == y_true),
    np.mean(landmark_pred == y_true),
    np.mean(late_fusion_pred == y_true),
    np.mean(hybrid_pred == y_true)
]

# Plot perbandingan akurasi
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['royalblue', 'lightgreen', 'orange', 'crimson'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontweight='bold')
plt.savefig('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/model_comparison.png')
plt.close()

# Bandingkan per kelas emosi
from sklearn.metrics import f1_score, precision_score, recall_score

# Hitung F1-score per kelas
f1_scores = {
    'CNN': f1_score(y_true, cnn_pred, average=None),
    'Landmark': f1_score(y_true, landmark_pred, average=None),
    'Late Fusion': f1_score(y_true, late_fusion_pred, average=None),
    'Hybrid Fusion': f1_score(y_true, hybrid_pred, average=None)
}

# Visualisasi F1-score per kelas emosi
fig, ax = plt.subplots(figsize=(14, 8))
bar_width = 0.2
index = np.arange(len(target_names))

for i, (model_name, f1) in enumerate(f1_scores.items()):
    ax.bar(index + i*bar_width, f1, bar_width, label=model_name)

ax.set_xlabel('Emotion')
ax.set_ylabel('F1-Score')
ax.set_title('F1-Score per Emotion Class')
ax.set_xticks(index + bar_width * 1.5)
ax.set_xticklabels(target_names)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/f1_score_comparison.png')
plt.close()

# Simpan hasil perbandingan
comparison_results = {
    'models': models,
    'accuracies': accuracies,
    'f1_scores': f1_scores,
    'confusion_matrices': {
        'CNN': confusion_matrix(y_true, cnn_pred),
        'Landmark': confusion_matrix(y_true, landmark_pred),
        'Late Fusion': confusion_matrix(y_true, late_fusion_pred),
        'Hybrid Fusion': confusion_matrix(y_true, hybrid_pred)
    }
}

with open('D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/models/model_comparison_results.pkl', 'wb') as f:
    pickle.dump(comparison_results, f)

# Print ringkasan hasil
print("\n===== MODEL COMPARISON =====")
for i, model in enumerate(models):
    print(f"{model}: Accuracy = {accuracies[i]:.4f}")

print("\nMacro F1-Scores:")
for model, f1 in f1_scores.items():
    print(f"{model}: {np.mean(f1):.4f}")