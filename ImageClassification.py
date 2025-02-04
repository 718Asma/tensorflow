# Import TensorFlow
import tensorflow as tf

# Step 1: Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Step 2: Normalize the data
# Pixel values range from 0 to 255, so we scale them to 0-1 to help the model train faster and perform better.
x_train, x_test = x_train / 255.0, x_test / 255.0

# Step 3: Build the model
# We're creating a neural network with 2 layers: a Flatten layer and a Dense layer.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Converts 28x28 images into a 1D array
    tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons
    tf.keras.layers.Dense(10, activation='softmax') # Output layer with 10 classes (0–9)
])

# Step 4: Compile the model
# Use 'adam' for optimization and sparse_categorical_crossentropy for multi-class classification
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Train the model
# Train using the training data
model.fit(x_train, y_train, epochs=5)

# Step 6: Evaluate the model
# Test the model's performance on unseen data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Step 7: Make predictions
# Let's predict the first 5 images in the test set
predictions = model.predict(x_test[:5])

# Print predictions for each image
for i, prediction in enumerate(predictions):
    print(f"Image {i+1}: Predicted digit is {tf.argmax(prediction).numpy()}")
