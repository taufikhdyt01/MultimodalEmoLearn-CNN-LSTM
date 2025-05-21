import pandas as pd
import numpy as np
import os

def generate_emotion_labels_from_faceapi(excel_path, output_dir):
    """
    Mengekstrak label emosi dari data Face API JS dan membuat ground truth.
    
    Args:
        excel_path: Path ke file Excel yang berisi data Face API JS
        output_dir: Direktori untuk menyimpan hasil
    """
    # Buat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {excel_path}...")
    df = pd.read_excel(excel_path)
    
    # Identifikasi kolom emosi
    emotion_columns = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
    
    # Pastikan semua kolom emosi ada
    missing_columns = [col for col in emotion_columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: Missing emotion columns: {missing_columns}")
        emotion_columns = [col for col in emotion_columns if col in df.columns]
    
    # Konversi kolom emosi ke format numerik
    for col in emotion_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Identifikasi emosi dominan untuk setiap baris
    if emotion_columns:
        # Buat DataFrame dengan hanya kolom emosi untuk mencari nilai maksimum
        emotions_only = df[emotion_columns]
        
        # Temukan indeks kolom dengan nilai tertinggi untuk setiap baris
        dominant_indices = emotions_only.idxmax(axis=1)
        
        # Tambahkan kolom untuk emosi dominan dan nilai konfiden
        df['Dominant_Emotion'] = dominant_indices
        df['Confidence_Score'] = emotions_only.max(axis=1)
        
        # Filter hanya data dengan konfiden tinggi (>= 0.5) dan bukan Low Confidence
        valid_mask = (df['Confidence_Score'] >= 0.5) & \
                     ((df['Classification'] != 'Low Confidence') | df['Classification'].isna())
        
        high_confidence_df = df[valid_mask].copy()
        
        # Hitung statistik
        total_frames = len(df)
        valid_frames = len(high_confidence_df)
        valid_percent = (valid_frames / total_frames * 100) if total_frames > 0 else 0
        
        print(f"Total frames: {total_frames}")
        print(f"Frames with high confidence: {valid_frames} ({valid_percent:.2f}%)")
        
        # Hitung distribusi emosi
        emotion_distribution = high_confidence_df['Dominant_Emotion'].value_counts()
        print("\nEmotion Distribution (High Confidence):")
        for emotion, count in emotion_distribution.items():
            percent = (count / valid_frames * 100) if valid_frames > 0 else 0
            print(f"{emotion}: {count} ({percent:.2f}%)")
        
        # Simpan data hasil
        labeled_data_path = os.path.join(output_dir, "labeled_data.xlsx")
        high_confidence_df.to_excel(labeled_data_path, index=False)
        print(f"\nSaved labeled data to {labeled_data_path}")
        
        # Simpan statistik
        stats_path = os.path.join(output_dir, "emotion_statistics.txt")
        with open(stats_path, 'w') as f:
            f.write(f"Total frames: {total_frames}\n")
            f.write(f"Frames with high confidence: {valid_frames} ({valid_percent:.2f}%)\n\n")
            f.write("Emotion Distribution (High Confidence):\n")
            for emotion, count in emotion_distribution.items():
                percent = (count / valid_frames * 100) if valid_frames > 0 else 0
                f.write(f"{emotion}: {count} ({percent:.2f}%)\n")
        
        # Analisis emosi berdasarkan sampel dan tantangan
        if 'user_id' in df.columns and 'page' in df.columns:
            print("\nAnalyzing emotions by sample and challenge...")
            
            # Ekstrak informasi tantangan dari kolom page
            def extract_challenge(page):
                if pd.isna(page) or not isinstance(page, str):
                    return "Unknown"
                if '/tantangan/' in page:
                    parts = page.split('/')
                    try:
                        tantangan_index = parts.index('tantangan')
                        if tantangan_index + 1 < len(parts) and parts[tantangan_index + 1]:
                            return parts[tantangan_index + 1]
                    except ValueError:
                        pass
                return "Unknown"
            
            high_confidence_df['Challenge'] = high_confidence_df['page'].apply(extract_challenge)
            
            # Analisis per sampel dan tantangan
            sample_challenge_stats = high_confidence_df.groupby(['user_id', 'Challenge']).agg({
                'Dominant_Emotion': lambda x: x.value_counts().to_dict(),
                'id': 'count'  # Gunakan kolom 'id' untuk menghitung jumlah frame
            }).reset_index()
            
            sample_challenge_stats.columns = ['user_id', 'Challenge', 'Emotion_Counts', 'Frame_Count']
            
            # Simpan ke Excel
            sample_stats_path = os.path.join(output_dir, "emotion_by_sample_challenge.xlsx")
            
            # Konversi dictionary ke format yang bisa disimpan di Excel
            def format_emotion_counts(counts_dict):
                return ', '.join([f"{emotion}: {count}" for emotion, count in counts_dict.items()])
            
            sample_challenge_stats['Emotion_Distribution'] = sample_challenge_stats['Emotion_Counts'].apply(format_emotion_counts)
            sample_challenge_stats.drop(columns=['Emotion_Counts']).to_excel(sample_stats_path, index=False)
            
            print(f"Saved sample and challenge analysis to {sample_stats_path}")
        
        return high_confidence_df
    else:
        print("Error: No emotion columns found in the data")
        return None

# Contoh penggunaan
if __name__ == "__main__":
    excel_path = "D:/Preprocessing/Cleaned/all_cleaned_data.xlsx"  # Path ke file Excel gabungan setelah dibersihkan
    output_dir = "D:/Preprocessing/Emotion_Labels"
    labeled_data = generate_emotion_labels_from_faceapi(excel_path, output_dir)