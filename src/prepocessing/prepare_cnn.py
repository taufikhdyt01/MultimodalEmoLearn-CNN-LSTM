import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import shutil
import re

def extract_challenge_from_page(page):
    """
    Mengekstrak nama tantangan dari URL page.
    
    Args:
        page: URL halaman, misal: '/tantangan/penjumlahan-dua-angka'
    
    Returns:
        Nama tantangan atau None jika tidak ditemukan
    """
    if pd.isna(page) or not isinstance(page, str):
        return None
        
    if '/tantangan/' in page:
        parts = page.split('/')
        try:
            tantangan_index = parts.index('tantangan')
            if tantangan_index + 1 < len(parts) and parts[tantangan_index + 1]:
                return parts[tantangan_index + 1]
        except ValueError:
            pass
    
    return None

def format_timestamp_to_filename(timestamp):
    """
    Mengkonversi timestamp ke format nama file frame.
    
    Args:
        timestamp: Timestamp dalam berbagai format
    
    Returns:
        Filename dalam format frame_HH_MM_SS.jpg atau None jika gagal
    """
    try:
        if pd.isna(timestamp):
            return None
        
        # Jika timestamp sudah dalam format pandas Timestamp
        if isinstance(timestamp, pd.Timestamp):
            return f"frame_{timestamp.hour:02d}_{timestamp.minute:02d}_{timestamp.second:02d}.jpg"
        
        # Jika string, coba parse
        dt = pd.to_datetime(timestamp)
        return f"frame_{dt.hour:02d}_{dt.minute:02d}_{dt.second:02d}.jpg"
    except Exception as e:
        print(f"Error formatting timestamp {timestamp}: {e}")
        return None

def get_sample_number_from_user_id(user_id, user_id_mapping):
    """
    Mendapatkan nomor sampel dari user_id menggunakan mapping.
    
    Args:
        user_id: User ID dari data
        user_id_mapping: Dictionary untuk mapping {sample_number: user_id}
    
    Returns:
        Nomor sampel atau None jika tidak ditemukan
    """
    for sample_num, uid in user_id_mapping.items():
        if uid == user_id:
            return sample_num
    return None

def find_image_in_sample_folder(images_dir, sample_num, challenge, frame_name):
    """
    Mencari file gambar di folder sampel, baik berdasarkan challenge spesifik
    atau di semua folder challenge.
    
    Args:
        images_dir: Direktori utama data gambar
        sample_num: Nomor sampel (1-20)
        challenge: Nama challenge (bisa None)
        frame_name: Nama file frame
    
    Returns:
        Path lengkap ke gambar jika ditemukan, None jika tidak
    """
    sample_dir = os.path.join(images_dir, f"Sample {sample_num}")
    if not os.path.exists(sample_dir):
        return None
    
    # Jika challenge diketahui, coba cari di folder challenge tersebut
    if challenge is not None:
        img_path = os.path.join(sample_dir, challenge, "cleaned_frames", frame_name)
        if os.path.exists(img_path):
            return img_path
    
    # Jika tidak ditemukan atau challenge tidak diketahui, cari di semua folder challenge
    for item in os.listdir(sample_dir):
        item_path = os.path.join(sample_dir, item)
        if os.path.isdir(item_path):
            # Cek apakah ada folder cleaned_frames
            frames_path = os.path.join(item_path, "cleaned_frames")
            if os.path.exists(frames_path):
                img_path = os.path.join(frames_path, frame_name)
                if os.path.exists(img_path):
                    return img_path
    
    return None

def prepare_image_data_for_cnn(split_dir, images_dir, output_dir, user_id_mapping, img_size=(224, 224)):
    """
    Mempersiapkan data gambar untuk input CNN.
    
    Args:
        split_dir: Direktori yang berisi file split data (train, val, test)
        images_dir: Direktori yang berisi gambar frame
        output_dir: Direktori untuk menyimpan data yang telah diformat untuk CNN
        user_id_mapping: Dictionary untuk mapping {sample_number: user_id}
        img_size: Ukuran gambar setelah resize (default: 224x224)
    """
    # Pastikan direktori output ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Define split files
    split_files = {
        'train': os.path.join(split_dir, 'train_data.xlsx'),
        'val': os.path.join(split_dir, 'val_data.xlsx'),
        'test': os.path.join(split_dir, 'test_data.xlsx')
    }
    
    # Process each split
    for split_name, split_file in split_files.items():
        if not os.path.exists(split_file):
            print(f"Warning: {split_file} not found. Skipping.")
            continue
        
        print(f"Processing {split_name} data...")
        df = pd.read_excel(split_file)
        
        # Create output directories for images
        split_img_dir = os.path.join(output_dir, f"{split_name}_images")
        os.makedirs(split_img_dir, exist_ok=True)
        
        # Ekstrak Challenge dari kolom page jika ada
        if 'page' in df.columns:
            df['Challenge'] = df['page'].apply(extract_challenge_from_page)
            print(f"Extracted Challenge from page column.")
            challenge_counts = df['Challenge'].value_counts()
            print(f"Challenge distribution: {challenge_counts.to_dict()}")
        
        # Create directories for each emotion class
        if 'Dominant_Emotion' in df.columns:
            emotions = df['Dominant_Emotion'].unique()
            for emotion in emotions:
                if not pd.isna(emotion):
                    os.makedirs(os.path.join(split_img_dir, str(emotion)), exist_ok=True)
        else:
            print(f"Error: 'Dominant_Emotion' column not found in {split_file}. Skipping.")
            continue
        
        # List for storing frame paths and labels for numpy arrays
        X_images = []
        y_emotions = []
        frame_paths = []
        
        # Process each row
        skipped = 0
        found = 0
        not_found = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            try:
                # Extract necessary information
                if 'user_id' not in df.columns or 'timestamp' not in df.columns or 'Dominant_Emotion' not in df.columns:
                    skipped += 1
                    continue
                
                user_id = row['user_id']
                timestamp = row['timestamp']
                emotion = row['Dominant_Emotion']
                
                if pd.isna(user_id) or pd.isna(timestamp) or pd.isna(emotion):
                    skipped += 1
                    continue
                
                # Get sample number from user_id using the mapping
                sample_num = get_sample_number_from_user_id(user_id, user_id_mapping)
                if sample_num is None:
                    if not_found < 5:
                        print(f"User ID {user_id} not found in mapping")
                    skipped += 1
                    continue
                
                # Convert timestamp to frame filename format
                frame_name = format_timestamp_to_filename(timestamp)
                if frame_name is None:
                    skipped += 1
                    continue
                
                # Get challenge if available
                challenge = row['Challenge'] if 'Challenge' in df.columns and not pd.isna(row['Challenge']) else None
                
                # Find the image in the cleaned directory using sample number
                img_path = find_image_in_sample_folder(images_dir, sample_num, challenge, frame_name)
                
                if img_path is None or not os.path.exists(img_path):
                    # Debug info for first few not found
                    if not_found < 5:
                        print(f"Image not found: user_id={user_id}, sample_num={sample_num}, challenge={challenge}, frame={frame_name}")
                        print(f"  Looked in: {os.path.join(images_dir, f'Sample {sample_num}')}")
                        if challenge:
                            print(f"  Specific path checked: {os.path.join(images_dir, f'Sample {sample_num}', challenge, 'cleaned_frames', frame_name)}")
                    not_found += 1
                    continue
                
                # Copy the image to the emotion-specific directory
                dest_path = os.path.join(split_img_dir, str(emotion), f"{user_id}_{frame_name}")
                shutil.copy2(img_path, dest_path)
                
                # Read the image and preprocess for array
                img = cv2.imread(img_path)
                if img is None:
                    skipped += 1
                    continue
                
                # Resize and normalize
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize to [0, 1]
                
                X_images.append(img)
                y_emotions.append(emotion)
                frame_paths.append(dest_path)
                found += 1
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                skipped += 1
        
        if not X_images:
            print(f"No valid image data found for {split_name}. Skipping.")
            continue
        
        # Convert to numpy arrays
        X = np.array(X_images)
        y = np.array(y_emotions)
        
        # Save as numpy arrays
        np.save(os.path.join(output_dir, f"X_{split_name}_images.npy"), X)
        np.save(os.path.join(output_dir, f"y_{split_name}_images.npy"), y)
        
        # Save frame paths for reference
        paths_df = pd.DataFrame({
            'path': frame_paths,
            'emotion': y_emotions
        })
        paths_df.to_csv(os.path.join(output_dir, f"{split_name}_image_paths.csv"), index=False)
        
        # Print statistics
        print(f"  - Found images: {found}")
        print(f"  - Images not found: {not_found}")
        print(f"  - Skipped: {skipped}")
        print(f"  - Shape of X_{split_name}_images: {X.shape}")
        
        # Print emotion distribution
        emotions, counts = np.unique(y, return_counts=True)
        print(f"  - Emotion distribution:")
        for emotion, count in zip(emotions, counts):
            percent = count / len(y) * 100
            print(f"    {emotion}: {count} ({percent:.2f}%)")
    
    print(f"Image data preparation complete. Files saved to {output_dir}")

# Definisi mapping antara nomor sampel dan user_id
def get_user_id_mapping():
    return {
        1: 97,  
        2: 117,    
        3: 99,
        4: 100,
        5: 101,
        6: 103,
        7: 102,
        8: 118,
        9: 104,
        10: 106,
        11: 107,
        12: 108,
        13: 109,
        14: 110,
        15: 111,
        16: 112,
        17: 114,
        18: 113,
        19: 115,
        20: 116
    }

# Contoh penggunaan
if __name__ == "__main__":
    split_dir = "D:/Preprocessing/Dataset_Split"  # Direktori dengan file train_data.xlsx, val_data.xlsx, test_data.xlsx
    images_dir = "D:/Preprocessing/Cleaned"       # Direktori dengan folder Sample 1, Sample 2, dll.
    output_dir = "D:/Preprocessing/CNN_Data"      # Direktori output untuk data CNN
    user_id_mapping = get_user_id_mapping()       # Mapping nomor sampel ke user_id
    
    prepare_image_data_for_cnn(split_dir, images_dir, output_dir, user_id_mapping)