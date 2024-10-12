import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# Load pre-trained object detection model
model = tf.saved_model.load("ssd_mobilenet_v2_coco/saved_model")

# Load and process an image
image = cv2.imread('input_image.jpg')
input_tensor = tf.convert_to_tensor(image)
input_tensor = input_tensor[tf.newaxis,...]

# Detect objects in the image
detections = model(input_tensor)

# Process results (for visualization)
boxes = detections['detection_boxes'].numpy()
scores = detections['detection_scores'].numpy()
classes = detections['detection_classes'].numpy()

# Visualize the results
for i in range(len(boxes)):
    if scores[i] > 0.5:  # Show detections above 50% confidence
        box = boxes[i]
        plt.imshow(image)
        plt.show()
