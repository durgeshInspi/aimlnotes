# NumPy (Numerical Python)

# 📚 NumPy Complete Notes for Students

## What is NumPy?

**NumPy (Numerical Python)** is a Python library used for **numerical computing**.

It provides support for:
- Large, multi-dimensional arrays
- Mathematical operations
- Statistical calculations
- Linear Algebra
- Matrix operations
- Random number generation
- Scientific Computing

NumPy is much **faster** than Python Lists because it stores data in continuous memory and is implemented in C.

---

# Why Do We Use NumPy?

Without NumPy:
- Python Lists are slow for numerical operations.
- Cannot perform vectorized operations.
- Difficult to work with matrices.
- More memory usage.

With NumPy:
- Faster Execution
- Less Memory
- Easy Mathematical Operations
- Easy Matrix Operations
- Data Science Ready
- Machine Learning Ready

---

# Where is NumPy Used?

NumPy is used in:
- Data Science
- Machine Learning
- Artificial Intelligence
- Deep Learning
- Computer Vision
- Image Processing
- Signal Processing
- Scientific Research
- Statistics
- Robotics
- Finance

---

# Installation

```bash
pip install numpy
```

Check Version:
```python
import numpy as np

print(np.__version__)
```

---

# Import NumPy

```python
import numpy as np
```

`np` is an alias. Instead of writing `numpy.array()`, we write `np.array()`.

---

# Creating Arrays

## 1D Array
```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])
print(arr)
```

## 2D Array
```python
arr = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
print(arr)
```

## 3D Array
```python
arr = np.array([
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])
print(arr)
```

---

# Difference Between List and NumPy Array

**Python List**:
```python
a = [1, 2, 3]
b = [4, 5, 6]
print(a + b) # Output: [1, 2, 3, 4, 5, 6]
```

**NumPy**:
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b) # Output: [5 7 9]
```

---

# Array Attributes

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.shape) # Output: (2, 3) -> 2 Rows, 3 Columns
print(arr.ndim)  # Output: 2 -> Dimensions
print(arr.size)  # Output: 6 -> Total elements
print(arr.dtype) # Output: int64
```

---

# Creating Special Arrays

```python
np.zeros((3, 4))      # 3x4 Matrix of Zeros
np.ones((2, 5))       # 2x5 Matrix of Ones
np.eye(4)             # 4x4 Identity Matrix
np.full((3, 3), 100)  # Fill Matrix with 100
```

---

# Range Functions

```python
# arange(start, stop, step)
np.arange(1, 11)      # [1, 2, 3, ..., 10]
np.arange(0, 20, 2)   # [0, 2, 4, ..., 18]

# linspace(start, stop, count) -> equally spaced numbers
np.linspace(0, 10, 5) # [0. , 2.5, 5. , 7.5, 10.]
```

---

# Random Numbers

```python
np.random.randint(1, 100)       # Random Integer
np.random.randint(1, 100, 10)   # Array of 10 Random Integers
np.random.rand(5)              # Random Floats between 0 and 1
np.random.rand(3, 3)           # 3x3 Random Matrix
```

---

# Reshape & Flatten

```python
arr = np.arange(1, 13)

# Reshape into 3 rows, 4 columns
arr_reshaped = arr.reshape(3, 4)

# Flatten back to 1D
arr_flat = arr_reshaped.flatten()
```

---

# Indexing & Slicing

```python
arr = np.array([10, 20, 30, 40, 50])
print(arr[2])   # 30
print(arr[1:4]) # [20 30 40]

arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(arr_2d[1, 2]) # 6
print(arr_2d[:, 1]) # [2 5]
```

---

# Mathematical & Aggregate Functions

```python
a = np.array([10, 20, 30, 40])

print(a.sum())    # Sum
print(a.mean())   # Mean
print(a.max())    # Maximum
print(a.min())    # Minimum
print(np.median(a)) # Median
print(a.std())    # Standard Deviation
print(a.var())    # Variance
```

---

# NumPy vs Python List Summary

| Feature | Python List | NumPy Array |
| :--- | :--- | :--- |
| **Speed** | Slow | Fast |
| **Memory** | More | Less |
| **Math Operations** | Difficult / Loop required | Vectorized / Built-in |
| **Matrix Support** | No | Yes |
| **Machine Learning** | No | Yes |
