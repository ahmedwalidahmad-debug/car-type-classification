"""
Real-time Car Type Classification using Webcam
Models: ResNet50 / InceptionV3
Application: Automotive recognition systems
"""

import cv2
import tensorflow as tf
import numpy as np

# Load trained car classification model
model = tf.keras.models.load_model("models/resnet50_cars.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_id = np.argmax(predictions)

    cv2.putText(
        frame,
        f"Predicted Car Class: {class_id}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Real-Time Car Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
