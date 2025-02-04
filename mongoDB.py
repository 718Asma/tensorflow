import tensorflow as tf
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ----------------- 1. Connect to MongoDB and Stream Data -----------------
def data_generator():
    # Connect to MongoDB (local or remote)
    client = MongoClient("mongodb://localhost:27017/")
    db = client["movies_db"]  # Replace with your database name
    collection = db["movies"]  # Replace with your collection name

    # Fetch data from MongoDB (cursor will fetch data in batches)
    cursor = collection.find({}, {"_id": 0, "genre": 1, "rating": 1})

    # Yield each document as a tensor (converting genre and rating to a numpy array)
    for document in cursor:
        genre = document["genre"]
        rating = document["rating"]
        yield [genre, rating]  # Yield as a list

    client.close()  # Close the MongoDB connection after the iteration is done

# ---------------- 2. Convert Data to TensorFlow Dataset -----------------
# Create a TensorFlow Dataset from the generator
dataset = tf.data.Dataset.from_generator(
    data_generator, 
    output_signature=(tf.TensorSpec(shape=(2,), dtype=tf.string))  # Expecting genre as string and rating as string
)

# ----------------- 3. Preprocessing - Label Encoding ------------------
# Create a mapping for genre labels
def preprocess_data(dataset):
    def encode_batch(batch):
        genre, rating = batch
        label_encoder = LabelEncoder()

        # Convert genre from tensor to numpy for LabelEncoder
        genre_np = genre.numpy()
        genre_encoded = label_encoder.fit_transform(genre_np)
        
        # Convert genre_encoded back to tensor
        genre_encoded = tf.convert_to_tensor(genre_encoded, dtype=tf.float32)
        return genre_encoded, rating

    # Map the preprocessing function to the dataset
    dataset = dataset.map(lambda batch: tf.py_function(encode_batch, [batch], [tf.float32, tf.float32]))
    return dataset

# -------------------- 4. Optimize Data Loading ---------------------
batch_size = 32
dataset = preprocess_data(dataset)  # Apply preprocessing
dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)  # Batch and prefetch for optimization

# ----------------- 5. Define the Model --------------------
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(1,)), # Define input shape explicitly, adjust to 1 as we have single encoded genre
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# ---------------- 6. Train the Model ----------------------
model.fit(dataset, epochs=10)
