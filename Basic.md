# Numpy
NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

# TensorFlow

## Install TensorFlow

```bash
pip install tensorflow
```

## Import TensorFlow

```python
import tensorflow as tf
```

## Understand the Basics

### TensorFlow is based on:

- #### Tensors: Multidimensional arrays (similar to NumPy arrays)

- #### Graphs: Computation graphs where mathematical operations are nodes, and edges are tensors

## Define a tensor ( || edge || constant )

#### You can't have a Tensor with mixed types. Example: [[1, "Church"], [3, 4]] will result in a ValueError: Can't convert Python sequence with mixed types to Tensor

- ### Scalar

```python
scalar = tf.constant(5)
```

- ### Vector

```python
vector = tf.constant([1.0, 2.0, 3.0])       # || vector = tf.constant([1., 2., 3.]) 
```

- ### Matrix

```python
matrix = tf.constant([[1, 2], [3, 4]])
```

- ### 3D Tensor

```python
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
```

## Print a tensor

```python
hello = tf.constant("Hello, TensorFlow!")
tf.print(hello)

# A normal print(hello) would return "tf.Tensor(b'Hello, TensorFlow!', shape=(), dtype=string)"
# While tf.print(hello) return the value of the tensor
```

## Tensor Operations

- ### Basic Math

```python
a = tf.constant([2, 3, 4])
b = tf.constant([1, 3, 2])

add = tf.add(a, b)  # Element-wise addition = [3, 6, 6]
mul = tf.multiply(a, b)  # Element-wise multiplication = [2, 9, 8]
# tf.matmul(matrix1, matrix2) is used to mutiply matrices
sub = tf.subtract(a, b)  # Element-wise subtraction = [1, 0, 2]
div = tf.divide(a, b)  # Element-wise division = [2, 1, 2]
```

- ### Reshaping Tensors

```python
matrix = tf.constant([[1, 2], [3, 4]])

reshaped = tf.reshape(matrix, [4, 1])  # Reshape 2x2 matrix into 4x1
```

- ### Create a tensor with all zeros

```python
zero_tensor = tf.zeros([2, 3])  #A matrix with 2 rows and 3 columns
```

## Variables

#### Unlike constants, variables can change their values. They are useful for machine learning models

- ### Creating Variables

```python
var = tf.Variable([1.0, 2.0, 3.0])
```

- ### Assigning New Values

```python
var.assign([4.0, 5.0, 6.0])
```

## Eager Execution

#### Newer versions of TensorFlow use __*eager executions*__, meaning operations are executed immediately. No need to build and run a computation graph explicitly

```python
x = tf.constant(3.0)
y = tf.constant(4.0)
z = x * y  # Executes immediately
```

## Keras

- ### Load a dataset
```python
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

- ### Build a machine learning model
```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])
```
