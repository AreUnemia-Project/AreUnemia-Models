# -*- coding: utf-8 -*-
"""testing-imageSegmentation-v1.1

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1a4-798FkIjjpwkiWheL7RhUbGL8Smwx3
"""

import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon

bestModelpath = '/content/best.pt'
bestModel = YOLO(bestModelpath)

imagePath = '/content/depositphotos_209849422-stock-photo-man-pulls-lower-eyelid-survey.jpg' # Input the correct path

# Predict and plot
results = bestModel.predict(source=imagePath)

masks=results[0].masks.xy

for mask in masks:
  # Example polygon coordinates (replace with actual polygon coordinates)
  polygon_coords = mask

  # Create a Shapely Polygon object
  polygon = Polygon(polygon_coords)

  # Assuming `image` is your original image
  image = cv2.imread(imagePath)

  # Create a mask image with the same shape as the original image
  mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)

  # Draw the filled polygon on the mask
  cv2.fillPoly(mask, [np.array(polygon_coords, dtype=np.int32)], color=255)

  # Bitwise AND operation to get the cropped region
  cropped_image = cv2.bitwise_and(image, image, mask=mask)

  # Display the result
  cv2.imwrite('crop_mask.jpg',cropped_image)
