import tensorflow as tf
import keras as keras

# Define a constant
hello = tf.constant("Hello, TensorFlow!")
# tf.print(hello)


# Perform a basic operation
a = tf.constant([12, 3])
b = tf.reduce_prod(a)
c = tf.reduce_sum(a)
final = tf.add(b, c)
# tf.print(b)
# tf.print(c)
# tf.print(final)


x = tf.constant(3.0)
y = tf.constant(4.0)
z = x * y
# tf.print("Z =", z)


# 3D Tensor
tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
# tf.print("3D Tensor:\n", tensor_3d)


# Matrix
matrix = tf.constant([[1, 2], [3, 4]])
# tf.print("Matrix:\n", matrix)
# tf.print("Shape:", matrix.shape)       # Dimensions
# tf.print("Data Type:", matrix.dtype)   # Data type
# tf.print("Rank:", tf.rank(matrix))     # Number of dimensions

# Reshape a tensor
reshaped = tf.reshape(matrix, [4, 1])  # Reshape 2x2 matrix into 4x1
# tf.print("Reshaped Tensor:\n", reshaped)

# Create a tensor with all zeros
zero_tensor = tf.zeros([2, 3])
# tf.print("Zero Tensor:\n", zero_tensor)


# Arithmetic operations
a = tf.constant([2, 3, 4])
b = tf.constant([1, 3, 2])

add = tf.add(a, b)  # Element-wise addition
mul = tf.multiply(a, b)  # Element-wise multiplication
sub = tf.subtract(a, b)  # Element-wise subtraction
div = tf.divide(a, b)  # Element-wise division
# tf.print("Addition:", add)
# tf.print("Multiplication:", mul)
# tf.print("Subtraction:", sub)
# tf.print("Division:", div)


# Create a variable
var = tf.Variable([1.0, 2.0, 3.0])
# tf.print("Variable:", var)

# Assign a new value
var.assign([4., 5., 6.])
# tf.print("Updated Variable:", var)


# Create tensors
x = tf.constant([[1, 2], [3, 4]])
y = tf.constant([[5, 6], [7, 8]])

# Perform operations
sum = tf.add(x, y)
product = tf.matmul(x, y)

# tf.print("Sum:\n", sum)
# tf.print("Product:\n", product)


# creating nodes in computation graph
node1 = tf.constant(3, dtype=tf.int32)
node2 = tf.constant(5, dtype=tf.int32)
node3 = tf.add(node1, node2)

# create tensorflow session object
sess = tf.compat.v1.Session()

# evaluating node3 and printing the result
print("sum of node1 and node2 is :",sess.run(node3))
# closing the session
sess.close()
