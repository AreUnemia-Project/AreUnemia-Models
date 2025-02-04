# -*- coding: utf-8 -*-
"""testing-imageSegmentation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1a4-798FkIjjpwkiWheL7RhUbGL8Smwx3
"""

import matplotlib.pyplot as plt
import cv2
import os

!pip install ultralytics

from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
bestModelpath = '/content/best.pt'
bestModel = YOLO(bestModelpath)

imagePath = '/content/8614-pink-eye.jpg'

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
fig.suptitle('Validation Set Inferences')

# Ensure the image exists at the given path
assert os.path.exists(imagePath), f"Image not found at {imagePath}"

# Predict and plot
results = bestModel.predict(source=imagePath, imgsz=640)
annotatedImage = results[0].plot()
annotatedImageRGB = cv2.cvtColor(annotatedImage, cv2.COLOR_BGR2RGB)

# Debugging: Print the results to check what is returned
# print(results)

# Display the image
ax.imshow(annotatedImageRGB)
ax.axis('off')

plt.tight_layout()
plt.show()

