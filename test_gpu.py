PS D:\research\2025_iris_taufik\MultimodalEmoLearn-CNN-LSTM> python test_gpu.py

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.1.3 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "D:\research\2025_iris_taufik\MultimodalEmoLearn-CNN-LSTM\test_gpu.py", line 1, in <module>
    import tensorflow as tf
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\__init__.py", line 37, in <module>
    from tensorflow.python.tools import module_util as _module_util
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\__init__.py", line 37, in <module>
    from tensorflow.python.eager import context
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\eager\context.py", line 35, in <module>
    from tensorflow.python.client import pywrap_tf_session
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\client\pywrap_tf_session.py", line 19, in <module>
    from tensorflow.python.client._pywrap_tf_session import *
AttributeError: _ARRAY_API not found

A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.1.3 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "D:\research\2025_iris_taufik\MultimodalEmoLearn-CNN-LSTM\test_gpu.py", line 1, in <module>
    import tensorflow as tf
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\__init__.py", line 37, in <module>
    from tensorflow.python.tools import module_util as _module_util
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\__init__.py", line 42, in <module>
    from tensorflow.python import data
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\__init__.py", line 21, in <module>
    from tensorflow.python.data import experimental
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\experimental\__init__.py", line 96, in <module>
    from tensorflow.python.data.experimental import service
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\experimental\service\__init__.py", line 419, in <module>
    from tensorflow.python.data.experimental.ops.data_service_ops import distribute
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\experimental\ops\data_service_ops.py", line 24, in <module>
    from tensorflow.python.data.experimental.ops import compression_ops
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\experimental\ops\compression_ops.py", line 16, in <module>
    from tensorflow.python.data.util import structure
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\util\structure.py", line 23, in <module>
    from tensorflow.python.data.util import nest
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\util\nest.py", line 36, in <module>
    from tensorflow.python.framework import sparse_tensor as _sparse_tensor
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\framework\sparse_tensor.py", line 24, in <module>
    from tensorflow.python.framework import constant_op
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\framework\constant_op.py", line 25, in <module>
    from tensorflow.python.eager import execute
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\eager\execute.py", line 23, in <module>
    from tensorflow.python.framework import dtypes
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\framework\dtypes.py", line 29, in <module>
    from tensorflow.python.lib.core import _pywrap_bfloat16
AttributeError: _ARRAY_API not found
ImportError: numpy.core._multiarray_umath failed to import
ImportError: numpy.core.umath failed to import
Traceback (most recent call last):
  File "D:\research\2025_iris_taufik\MultimodalEmoLearn-CNN-LSTM\test_gpu.py", line 1, in <module>
    import tensorflow as tf
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\__init__.py", line 37, in <module>
    from tensorflow.python.tools import module_util as _module_util
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\__init__.py", line 42, in <module>
    from tensorflow.python import data
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\__init__.py", line 21, in <module>
    from tensorflow.python.data import experimental
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\experimental\__init__.py", line 96, in <module>
    from tensorflow.python.data.experimental import service
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\experimental\service\__init__.py", line 419, in <module>
    from tensorflow.python.data.experimental.ops.data_service_ops import distribute
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\experimental\ops\data_service_ops.py", line 24, in <module>
    from tensorflow.python.data.experimental.ops import compression_ops
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\experimental\ops\compression_ops.py", line 16, in <module>
    from tensorflow.python.data.util import structure
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\util\structure.py", line 23, in <module>
    from tensorflow.python.data.util import nest
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\data\util\nest.py", line 36, in <module>
    from tensorflow.python.framework import sparse_tensor as _sparse_tensor
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\framework\sparse_tensor.py", line 24, in <module>
    from tensorflow.python.framework import constant_op
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\framework\constant_op.py", line 25, in <module>
    from tensorflow.python.eager import execute
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\eager\execute.py", line 23, in <module>
    from tensorflow.python.framework import dtypes
  File "C:\Users\fitra\AppData\Local\Programs\Python\Python310\lib\site-packages\tensorflow\python\framework\dtypes.py", line 34, in <module>
    _np_bfloat16 = _pywrap_bfloat16.TF_bfloat16_type()
TypeError: Unable to convert function return value to a Python type! The signature was
        () -> handle