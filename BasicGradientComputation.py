import tensorflow as tf

# Define a variable (trainable parameter)
x = tf.Variable(3.0)

# Define a simple function: y = x²
def function():
    return x ** 2

# Compute the gradient of y with respect to x
with tf.GradientTape() as tape:
    y = function()

grad = tape.gradient(y, x)

print("Value of x:", x.numpy())
print("Value of y = x^2:", y.numpy())
print("Gradient dy/dx:", grad.numpy())  # Should print 2*x
