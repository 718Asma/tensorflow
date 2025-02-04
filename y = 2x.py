# Import TensorFlow
import tensorflow as tf
import numpy as np

# Step 1: Prepare the data
x_train = np.array([1, 2, 3, 4])  # Input values
y_train = np.array([2, 4, 6, 8])  # Output values (what we want it to learn)

# Step 2: Build a simple model
# We'll create a single layer with one neuron
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Step 3: Compile the model
# We specify how it should learn (optimizer) and how it measures success (loss)
model.compile(optimizer='sgd', loss='mean_squared_error')

# Step 4: Train the model
# Let the model learn from the data (it adjusts to match x_train -> y_train)
model.fit(x_train, y_train, epochs=500, verbose=0)  # Train for 500 cycles (epochs)

# Step 5: Use the trained model
# Let's predict what y would be for a new x value (e.g., x = 10)
result = model.predict(np.array([10]))
print(f"When x = 10, the predicted y is: {result[0][0]:.2f}")