"""
Grad-CAM visualization for Car Type Classification
Dataset: Stanford Cars Dataset
Supported Models: ResNet50, InceptionV3

This module helps visualize fine-grained features such as
headlights, grills, and car logos that influence model predictions.
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates Grad-CAM heatmap for a given car image and CNN model
    """

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def overlay_heatmap(img_path, heatmap, alpha=0.4):
    """
    Overlays Grad-CAM heatmap on the original car image
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    output = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Example last convolutional layers:
# ResNet50     → conv5_block3_out
# InceptionV3  → mixed10
