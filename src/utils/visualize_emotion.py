import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_emotion_distribution(labeled_data_path, output_dir):
    """
    Membuat visualisasi distribusi emosi dari data berlabel.
    
    Args:
        labeled_data_path: Path ke file Excel dengan data berlabel
        output_dir: Direktori untuk menyimpan visualisasi
    """
    # Pastikan direktori output ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_excel(labeled_data_path)
    
    # 1. Distribusi Emosi Keseluruhan
    plt.figure(figsize=(12, 6))
    emotion_counts = df['Dominant_Emotion'].value_counts()
    emotion_percent = emotion_counts / len(df) * 100
    
    colors = sns.color_palette("husl", len(emotion_counts))
    bars = plt.bar(emotion_counts.index, emotion_counts.values, color=colors)
    
    # Tambahkan persentase di atas bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f"{emotion_percent.iloc[i]:.1f}%",
                ha='center', va='bottom', rotation=0)
    
    plt.title('Overall Emotion Distribution', fontsize=16)
    plt.xlabel('Emotion', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_emotion_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Distribusi Emosi per Sampel (jika ada kolom user_id)
    if 'user_id' in df.columns:
        plt.figure(figsize=(15, 8))
        sample_emotions = pd.crosstab(df['user_id'], df['Dominant_Emotion'])
        sample_emotions_percent = sample_emotions.div(sample_emotions.sum(axis=1), axis=0) * 100
        
        # Plot heatmap
        sns.heatmap(sample_emotions_percent, annot=True, fmt='.1f', cmap='YlGnBu', 
                    linewidths=.5, cbar_kws={'label': 'Percentage (%)'})
        plt.title('Emotion Distribution by Sample', fontsize=16)
        plt.xlabel('Emotion', fontsize=14)
        plt.ylabel('Sample ID', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_by_sample.png'), dpi=300)
        plt.close()
    
    # 3. Distribusi Emosi per Tantangan (jika ada kolom Challenge)
    if 'Challenge' in df.columns and df['Challenge'].nunique() > 1:
        plt.figure(figsize=(15, 8))
        challenge_emotions = pd.crosstab(df['Challenge'], df['Dominant_Emotion'])
        challenge_emotions_percent = challenge_emotions.div(challenge_emotions.sum(axis=1), axis=0) * 100
        
        # Plot heatmap
        sns.heatmap(challenge_emotions_percent, annot=True, fmt='.1f', cmap='YlGnBu', 
                    linewidths=.5, cbar_kws={'label': 'Percentage (%)'})
        plt.title('Emotion Distribution by Challenge', fontsize=16)
        plt.xlabel('Emotion', fontsize=14)
        plt.ylabel('Challenge', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'emotion_by_challenge.png'), dpi=300)
        plt.close()
    
    # 4. Distribusi Nilai Konfiden
    plt.figure(figsize=(10, 6))
    plt.hist(df['Confidence_Score'], bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Confidence Scores', fontsize=16)
    plt.xlabel('Confidence Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=300)
    plt.close()
    
    # 5. Distribusi Konfiden per Emosi
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Dominant_Emotion', y='Confidence_Score', data=df)
    plt.title('Confidence Distribution by Emotion', fontsize=16)
    plt.xlabel('Emotion', fontsize=14)
    plt.ylabel('Confidence Score', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_by_emotion.png'), dpi=300)
    plt.close()
    
    # 6. Jika timestamp tersedia, analisis temporal
    if 'timestamp' in df.columns:
        try:
            # Konversi ke datetime jika belum
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Tambahkan kolom waktu
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            
            # Visualisasi distribusi emosi berdasarkan jam
            plt.figure(figsize=(14, 8))
            hour_emotion = pd.crosstab(df['hour'], df['Dominant_Emotion'])
            hour_emotion_percent = hour_emotion.div(hour_emotion.sum(axis=1), axis=0) * 100
            
            sns.heatmap(hour_emotion_percent, annot=True, fmt='.1f', cmap='YlGnBu',
                       linewidths=.5, cbar_kws={'label': 'Percentage (%)'})
            plt.title('Emotion Distribution by Hour', fontsize=16)
            plt.xlabel('Emotion', fontsize=14)
            plt.ylabel('Hour', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'emotion_by_hour.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Warning: Could not perform temporal analysis: {e}")
    
    print(f"All visualizations saved to {output_dir}")

# Contoh penggunaan
if __name__ == "__main__":
    labeled_data_path = "D:/Preprocessing/Emotion_Labels/labeled_data.xlsx"
    output_dir = "D:/Preprocessing/Emotion_Labels/Visualizations"
    visualize_emotion_distribution(labeled_data_path, output_dir)