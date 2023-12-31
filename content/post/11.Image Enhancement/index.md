+++
author = "Puja Chaudhury"
title = "Mastering Image Enhancement Techniques"
date = "2023-08-11"
description = "Histogram Equalization and Contrast Stretching"
image = "intro.png"
+++

Hey there, tech enthusiasts! Today, we're diving deep into the world of image enhancement. If we've ever wondered how to make our images pop, we're in the right place. We'll be exploring two powerful techniques: histogram equalization and contrast stretching. These methods are particularly useful when we're dealing with low-contrast or poorly lit images. So, let's get started!

## What is Image Enhancement?

Before we dive in, let's clarify what image enhancement is all about. Essentially, it's a collection of techniques designed to improve the visual quality of an image. Whether we're looking to sharpen an image, boost its contrast, or highlight specific features, image enhancement has got we covered.

## Histogram Equalization

### A Quick Recap on Histograms

For a detailed understanding of what a histogram is, we can check out my previous blog post on [Image Histograms](https://catplotlib.com/p/image-histograms/). In short, a histogram in the context of images shows the frequency distribution of pixel intensities.

### How Does Histogram Equalization Work?

Histogram equalization is a technique that redistributes the intensity levels of an image to span the entire range. This often leads to images with enhanced contrast and detail.

Here's a simplified algorithm for histogram equalization:

1. **Calculate the Histogram**: Count the frequency of each intensity level.
2. **Compute the Cumulative Distribution Function (CDF)**: Sum up the frequencies cumulatively.
3. **Normalize the CDF**: Scale the CDF to fit the intensity range of the image.
4. **Map the Original Intensities**: Replace each pixel's intensity based on the normalized CDF.

#### Python Code Snippet using OpenCV

```python
import cv2
import numpy as np

# Read the image
image = cv2.imread('image.jpg', 0)

# Perform histogram equalization
equalized_image = cv2.equalizeHist(image)

# Save the enhanced image
cv2.imwrite('equalized_image.jpg', equalized_image)
```
![Histogram Equalization](equalised.png)

## Contrast Stretching
### What is Contrast Stretching?
Contrast stretching aims to improve the contrast of an image by stretching the range of intensity levels it contains. Unlike histogram equalization, which changes the shape of the histogram, contrast stretching only expands or compresses it.

### How to Perform Contrast Stretching?
The basic algorithm for contrast stretching is quite straightforward:

- Identify Min and Max Intensities: Find the minimum and maximum intensity values in the image.
- Stretch the Intensity Levels: Apply a linear transformation to stretch the intensity levels between the desired minimum and maximum.
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(img):
    # Initialize min and max pixel values for each channel
    min_out = 0
    max_out = 255
    
    # Split the image into its color channels
    b, g, r = cv2.split(img)
    
    # Perform contrast stretching for each channel
    for channel in [b, g, r]:
        min_in = np.min(channel)
        max_in = np.max(channel)
        
        # Apply the contrast stretching transformation
        channel[:] = ((channel - min_in) / (max_in - min_in)) * (max_out - min_out) + min_out
    
    # Merge the channels back together
    return cv2.merge([b, g, r])

# Read the image
img = cv2.imread('nature.png')

# Perform contrast stretching
stretched_img = contrast_stretching(img)

# Convert BGR images to RGB (OpenCV loads images in BGR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
stretched_img = cv2.cvtColor(stretched_img, cv2.COLOR_BGR2RGB)

#Save the enhanced image
cv2.imwrite('stretched_img.jpg', stretched_img)
```
![Contrast Stretching](contrast.png)

## Conclusion
Both histogram equalization and contrast stretching are powerful tools for image enhancement. While histogram equalization is more effective for images with poor contrast, contrast stretching is simpler and can be more intuitive to use. Either way, mastering these techniques can significantly up our image processing game.

That's it for today! Feel free to experiment with these techniques and let me know how it goes. Until next time, happy coding!