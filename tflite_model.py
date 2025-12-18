"""
Convert Car Classification Model to TensorFlow Lite
Purpose: Mobile and embedded deployment for automotive applications
"""

import tensorflow as tf

# Load trained car classification model
model = tf.keras.models.load_model("models/resnet50_cars.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite car classification model saved successfully!")
