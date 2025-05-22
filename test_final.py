import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

print('✅ All packages working!')
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'TensorFlow: {tf.__version__}')
print(f'GPUs: {len(tf.config.list_physical_devices("GPU"))}')