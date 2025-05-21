import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def split_dataset(labeled_data_path, output_dir, train_size=0.8, val_size=0.1, test_size=0.1, stratify=True, random_state=42):
    """
    Membagi dataset menjadi set pelatihan, validasi, dan pengujian.
    
    Args:
        labeled_data_path: Path ke file Excel dengan data berlabel
        output_dir: Direktori untuk menyimpan hasil split
        train_size: Proporsi data untuk set pelatihan (default: 0.8)
        val_size: Proporsi data untuk set validasi (default: 0.1)
        test_size: Proporsi data untuk set pengujian (default: 0.1)
        stratify: Jika True, pastikan distribusi emosi seimbang di setiap set (default: True)
        random_state: Seed untuk random state (default: 42)
    """
    # Pastikan proporsi pembagian valid (berjumlah 1.0)
    assert abs((train_size + val_size + test_size) - 1.0) < 1e-10, "Proporsi train, val, dan test harus berjumlah 1.0"
    
    # Pastikan direktori output ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_excel(labeled_data_path)
    
    # Pastikan ada kolom Dominant_Emotion
    if 'Dominant_Emotion' not in df.columns:
        print("Error: Column 'Dominant_Emotion' not found. Cannot split dataset.")
        return
    
    # Hitung total baris
    total_rows = len(df)
    
    if total_rows == 0:
        print("Error: Dataset is empty.")
        return
    
    # Tentukan stratify parameter
    stratify_column = df['Dominant_Emotion'] if stratify else None
    
    # Split menjadi train dan sisa (val+test)
    train, temp = train_test_split(
        df, 
        train_size=train_size, 
        random_state=random_state,
        stratify=stratify_column
    )
    
    # Update stratify column untuk split berikutnya
    if stratify:
        stratify_column = temp['Dominant_Emotion']
    
    # Hitung proporsi relatif val dan test dari sisa data
    relative_val_size = val_size / (val_size + test_size)
    
    # Split sisa menjadi val dan test
    val, test = train_test_split(
        temp, 
        train_size=relative_val_size, 
        random_state=random_state,
        stratify=stratify_column
    )
    
    # Simpan masing-masing subset
    train.to_excel(os.path.join(output_dir, "train_data.xlsx"), index=False)
    val.to_excel(os.path.join(output_dir, "val_data.xlsx"), index=False)
    test.to_excel(os.path.join(output_dir, "test_data.xlsx"), index=False)
    
    # Hitung statistik
    train_count = len(train)
    val_count = len(val)
    test_count = len(test)
    
    print(f"Dataset split complete:")
    print(f"  - Total records: {total_rows}")
    print(f"  - Training set: {train_count} ({train_count/total_rows*100:.2f}%)")
    print(f"  - Validation set: {val_count} ({val_count/total_rows*100:.2f}%)")
    print(f"  - Test set: {test_count} ({test_count/total_rows*100:.2f}%)")
    
    # Periksa distribusi emosi di setiap set
    print("\nEmotion Distribution:")
    
    sets = {
        "Training": train,
        "Validation": val,
        "Test": test
    }
    
    emotion_stats = {}
    
    for name, dataset in sets.items():
        emotion_counts = dataset['Dominant_Emotion'].value_counts()
        emotion_percent = emotion_counts / len(dataset) * 100
        
        emotion_stats[name] = {
            'counts': emotion_counts,
            'percent': emotion_percent
        }
        
        print(f"\n{name} Set:")
        for emotion, count in emotion_counts.items():
            percent = emotion_percent[emotion]
            print(f"  {emotion}: {count} ({percent:.2f}%)")
    
    # Simpan ringkasan statistik
    with open(os.path.join(output_dir, "split_statistics.txt"), 'w') as f:
        f.write(f"Dataset Split Statistics\n")
        f.write(f"----------------------\n\n")
        f.write(f"Total records: {total_rows}\n")
        f.write(f"Training set: {train_count} ({train_count/total_rows*100:.2f}%)\n")
        f.write(f"Validation set: {val_count} ({val_count/total_rows*100:.2f}%)\n")
        f.write(f"Test set: {test_count} ({test_count/total_rows*100:.2f}%)\n\n")
        
        f.write(f"Emotion Distribution\n")
        f.write(f"-------------------\n\n")
        
        for name, stats in emotion_stats.items():
            f.write(f"{name} Set:\n")
            for emotion, count in stats['counts'].items():
                percent = stats['percent'][emotion]
                f.write(f"  {emotion}: {count} ({percent:.2f}%)\n")
            f.write("\n")
    
    return train, val, test

# Contoh penggunaan
if __name__ == "__main__":
    labeled_data_path = "D:/Preprocessing/Emotion_Labels/labeled_data.xlsx"
    output_dir = "D:/Preprocessing/Dataset_Split"
    train_set, val_set, test_set = split_dataset(
        labeled_data_path, 
        output_dir, 
        train_size=0.8,  # 80% untuk training
        val_size=0.1,    # 10% untuk validasi
        test_size=0.1    # 10% untuk testing
    )