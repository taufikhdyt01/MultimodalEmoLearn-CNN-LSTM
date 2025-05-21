def late_fusion_weighted(cnn_probs, landmark_probs, weight_cnn=0.6):
    """
    Menggabungkan prediksi dari model CNN dan Landmark dengan pembobotan
    
    Args:
        cnn_probs: Probabilitas prediksi dari model CNN
        landmark_probs: Probabilitas prediksi dari model Landmark
        weight_cnn: Bobot untuk model CNN (0-1)
        
    Returns:
        Array probabilitas gabungan
    """
    weight_landmark = 1 - weight_cnn
    fused_probs = (weight_cnn * cnn_probs) + (weight_landmark * landmark_probs)
    return fused_probs

# Load model terbaik
cnn_model = load_model('D:/Models/cnn_model_best.h5')
landmark_model = load_model('D:/Models/landmark_model_best.h5')

# Prediksi dengan kedua model
cnn_probs = cnn_model.predict(X_test_images)
landmark_probs = landmark_model.predict(X_test_landmarks)

# Uji berbagai bobot fusion
weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
results = []

for weight_cnn in weights:
    # Fusion dengan bobot tertentu
    fused_probs = late_fusion_weighted(cnn_probs, landmark_probs, weight_cnn)
    y_pred_fused = np.argmax(fused_probs, axis=1)
    
    # Hitung akurasi
    accuracy = np.mean(y_pred_fused == y_true)
    results.append((weight_cnn, accuracy))
    print(f"CNN Weight: {weight_cnn:.1f}, Accuracy: {accuracy:.4f}")

# Pilih bobot terbaik
best_weight = max(results, key=lambda x: x[1])[0]
print(f"\nBest CNN Weight: {best_weight:.1f}")

# Fusion akhir dengan bobot terbaik
final_fused_probs = late_fusion_weighted(cnn_probs, landmark_probs, best_weight)
y_pred_fused = np.argmax(final_fused_probs, axis=1)

# Evaluasi late fusion
print("\nLate Fusion Classification Report:")
print(classification_report(y_true, y_pred_fused, target_names=target_names))

# Plot confusion matrix
cm = confusion_matrix(y_true, y_pred_fused)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title(f'Late Fusion Confusion Matrix (CNN Weight: {best_weight:.1f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('D:/Models/late_fusion_confusion_matrix.png')
plt.close()

# Plot perbandingan akurasi
plt.figure(figsize=(10, 6))
plt.plot(weights, [r[1] for r in results], marker='o')
plt.axhline(y=test_acc, color='r', linestyle='--', label=f'CNN Only: {test_acc:.4f}')
plt.axhline(y=np.mean(y_pred == y_true), color='g', linestyle='--', 
            label=f'Landmark Only: {np.mean(y_pred == y_true):.4f}')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('CNN Weight')
plt.ylabel('Accuracy')
plt.title('Late Fusion Performance with Different Weights')
plt.legend()
plt.savefig('D:/Models/late_fusion_weights.png')
plt.close()

# Simpan hasil sebagai model ensemble
import pickle
with open('D:/Models/late_fusion_ensemble.pkl', 'wb') as f:
    pickle.dump({
        'best_weight': best_weight,
        'cnn_model_path': 'D:/Models/cnn_model_best.h5',
        'landmark_model_path': 'D:/Models/landmark_model_best.h5'
    }, f)