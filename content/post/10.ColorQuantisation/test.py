import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
image = cv2.imread('ging.jpeg')

# OpenCV reads images in BGR format, convert it to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image
pixels = image.reshape((-1, 3))

# Convert to floating point
pixels = np.float32(pixels)

# Define criteria and apply kmeans
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 16
_, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert back to 8-bit values
centers = np.uint8(centers)

# Map the labels to the centers
segmented_image = centers[labels.flatten()]

# Reshape back to the original image
segmented_image = segmented_image.reshape(image.shape)

# Convert the segmented image back to RGB format for Matplotlib
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

# Plotting the original and segmented images side by side
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Quantized Image')
plt.imshow(segmented_image_rgb)
plt.axis('off')

plt.show()
