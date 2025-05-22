# Conda Environment Setup untuk Multimodal Emotion Recognition

## 🐍 Step 1: Install Anaconda/Miniconda (jika belum ada)

### Download dan Install:
- **Miniconda** (lighter): https://docs.conda.io/en/latest/miniconda.html
- **Anaconda** (full): https://www.anaconda.com/products/distribution

### Verify Installation:
```cmd
conda --version
conda info
```

## 🚀 Step 2: Create Clean Environment

```cmd
# Create environment dengan Python 3.9 (most stable for DirectML)
conda create -n multimodal-emotion python=3.9 -y

# Activate environment
conda activate multimodal-emotion
```

## 📦 Step 3: Install TensorFlow DirectML

```cmd
# Install TensorFlow CPU dan DirectML Plugin
pip install tensorflow-cpu==2.10.0
pip install tensorflow-directml-plugin

# Install compatible NumPy
pip install "numpy<2.0"
```

## 📊 Step 4: Install Data Science Packages

```cmd
# Core packages
conda install pandas matplotlib seaborn scikit-learn jupyter -y

# Additional packages
pip install pillow opencv-python tqdm
```

## 🧪 Step 5: Test GPU Functionality

Buat file `test_setup.py`:

```python
import sys
import os

print("=== ENVIRONMENT CHECK ===")
print(f"Python: {sys.version}")
print(f"Environment: {os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')}")

# Test NumPy
try:
    import numpy as np
    print(f"NumPy: {np.__version__}")
except ImportError as e:
    print(f"NumPy error: {e}")

# Test TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
    
    # Check GPU/DML devices
    gpu_devices = tf.config.list_physical_devices('GPU')
    dml_devices = tf.config.list_physical_devices('DML')
    
    print(f"GPU devices: {len(gpu_devices)}")
    print(f"DML devices: {len(dml_devices)}")
    
    if gpu_devices or dml_devices:
        print("✅ AMD RX 6600 LE detected!")
        
        # Test GPU computation
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[2.0, 1.0], [1.0, 2.0]])
            c = tf.matmul(a, b)
            print("🚀 GPU computation successful!")
            print(f"Result: {c.numpy()}")
    else:
        print("⚠️ No GPU detected")
        
except ImportError as e:
    print(f"TensorFlow error: {e}")

# Test other packages
packages = ['pandas', 'matplotlib', 'seaborn', 'sklearn']
for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg}: OK")
    except ImportError:
        print(f"❌ {pkg}: Missing")

print("\n=== SETUP COMPLETE ===")
```

Test dengan:
```cmd
python test_setup.py
```

## 🔧 Step 6: Install Jupyter Kernel

```cmd
# Install ipykernel untuk environment ini
pip install ipykernel

# Add kernel ke Jupyter
python -m ipykernel install --user --name multimodal-emotion --display-name "Multimodal Emotion (GPU)"
```

## 📁 Step 7: Navigate ke Project Directory

```cmd
# Navigate ke folder project
cd D:\research\2025_iris_taufik\MultimodalEmoLearn-CNN-LSTM

# Start Jupyter
jupyter notebook
```

Pilih kernel **"Multimodal Emotion (GPU)"** saat membuka notebook.

## 🎯 Expected Results

Setelah setup berhasil, Anda akan punya:

### Environment Info:
```
Python: 3.9.x
Environment: multimodal-emotion
NumPy: 1.24.x (< 2.0)
TensorFlow: 2.10.0
GPU devices: 1 (RX 6600 LE)
```

### GPU Performance:
- **3-5x faster** training vs CPU
- **~6GB VRAM** available untuk models
- **Batch sizes**: CNN=32, Landmark=64, Fusion=16

## 🐛 Troubleshooting

### Issue 1: Conda not found
```cmd
# Add conda to PATH atau restart terminal
# Atau buka Anaconda Prompt
```

### Issue 2: GPU not detected
```cmd
# Update AMD drivers
# Restart setelah driver install
# Verify DirectML plugin installation
pip show tensorflow-directml-plugin
```

### Issue 3: Kernel not showing in Jupyter
```cmd
# List available kernels
jupyter kernelspec list

# Remove dan install ulang kernel
jupyter kernelspec remove multimodal-emotion
python -m ipykernel install --user --name multimodal-emotion
```

## 📋 Quick Commands Reference

```cmd
# Activate environment
conda activate multimodal-emotion

# Deactivate environment  
conda deactivate

# List environments
conda env list

# Update packages
conda update --all

# Export environment
conda env export > environment.yml

# Create from export
conda env create -f environment.yml
```

## 🎨 VS Code Integration (Optional)

Jika menggunakan VS Code:

1. Install **Python extension**
2. **Ctrl+Shift+P** → "Python: Select Interpreter"
3. Pilih interpreter dari environment: 
   `~/anaconda3/envs/multimodal-emotion/python.exe`

## ✅ Verification Checklist

- [ ] Conda installed dan working
- [ ] Environment `multimodal-emotion` created
- [ ] TensorFlow DirectML plugin installed
- [ ] AMD RX 6600 LE detected
- [ ] GPU computation test passed
- [ ] Jupyter kernel available
- [ ] All required packages installed

Setelah semua checklist ✅, Anda siap menjalankan notebooks dengan GPU acceleration!