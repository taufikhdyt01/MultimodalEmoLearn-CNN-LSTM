import subprocess
import os
import glob
import json
from datetime import datetime, timedelta

# Path ke video OBS (GANTI DENGAN PATH VIDEO ANDA)
video_path = 'D:/research/2025_iris_taufik/MultimodalEmoLearn-CNN-LSTM/data/raw/Sample 5/2025-05-21 14-38-50.mkv'

# Buat direktori output
output_dir = 'output'
frames_dir = f"{output_dir}/frames"
temp_dir = f"{output_dir}/temp_frames"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Waktu mulai video dan waktu mulai ekstraksi frame
video_start_time = '14:38:50'  # Waktu video OBS mulai
frame_start_time = '14:39:49'  # Waktu frame pertama diambil

# Konversi ke datetime object
video_start_datetime = datetime.strptime(video_start_time, '%H:%M:%S')
frame_start_datetime = datetime.strptime(frame_start_time, '%H:%M:%S')

# Hitung selisih waktu dalam detik
time_diff_seconds = (frame_start_datetime - video_start_datetime).total_seconds()
print(f"Selisih waktu: {time_diff_seconds} detik")

# 1. Dapatkan durasi video menggunakan FFprobe
duration_cmd = [
    'ffprobe',
    '-v', 'error',
    '-show_entries', 'format=duration',
    '-of', 'json',
    video_path
]

result = subprocess.run(duration_cmd, stdout=subprocess.PIPE, text=True)
duration_info = json.loads(result.stdout)
total_seconds = float(duration_info['format']['duration'])
print(f"Total durasi video: {total_seconds:.2f} detik")

# 2. Ekstrak frame setiap 5 detik mulai dari detik ke-16
print(f"Mengekstrak frame setiap 5 detik mulai dari detik ke-{time_diff_seconds}...")
extract_cmd = [
    'ffmpeg',
    '-i', video_path,
    '-ss', f'00:00:{int(time_diff_seconds)}',  # Mulai dari selisih waktu
    '-vf', 'fps=1/5',   # 1 frame setiap 5 detik
    '-q:v', '2',        # Kualitas tinggi
    f'{temp_dir}/frame_%04d.jpg'
]
subprocess.run(extract_cmd)

# 3. Ganti nama file dengan timestamp
print("Mengganti nama file dengan timestamp...")

# Dapatkan semua file frame yang dihasilkan, urutkan berdasarkan nama
frame_files = sorted(glob.glob(f"{temp_dir}/frame_*.jpg"))

for i, file_path in enumerate(frame_files):
    # Hitung timestamp untuk frame ini
    frame_time = frame_start_datetime + timedelta(seconds=i*5)
    timestamp_str = frame_time.strftime('%H_%M_%S')
    
    # Path file baru dengan timestamp
    new_file_path = f"{frames_dir}/frame_{timestamp_str}.jpg"
    
    # Pindahkan dan ganti nama file
    os.rename(file_path, new_file_path)
    print(f"Renamed frame to {new_file_path}")

# Hapus direktori temporary
if os.path.exists(temp_dir) and not os.listdir(temp_dir):
    os.rmdir(temp_dir)

print(f"Selesai! {len(frame_files)} frame telah disimpan di folder '{frames_dir}'")