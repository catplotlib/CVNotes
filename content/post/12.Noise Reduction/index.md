+++
author = "Puja Chaudhury"
title = "Noise Reduction in Images"
date = "2023-10-01"
description = "Understanding and Implementing Techniques."
image = "h.png"
+++

Hey there! Today we're diving deep into the world of image noise and its reduction techniques. If we've ever snapped a photo in low light conditions or dealt with raw sensor data, we've likely encountered image noise. This pesky artifact can compromise image quality, making it difficult for both humans and algorithms to interpret the content. In this blog, we'll discuss different types of noise and how to tackle them using Python and OpenCV.

## Understanding Image Noise

### Types of Noise
- Gaussian Noise: This is caused by electronic interference and sensor limitations. It's normally distributed and affects each pixel independently.
![Gaussian Noise](https://www.seas.ucla.edu/dsplab/ie/lenna_gau.JPG)
- Salt-and-Pepper Noise: This type of noise presents itself as sparsely occurring white and black pixels.
![Salt-and-Pepper Noise](https://www.fit.vutbr.cz/~vasicek/imagedb/img_corrupted/impnoise_005/108073.png)
- Speckle Noise: This occurs in images from coherent imaging systems, like ultrasound or synthetic aperture radar.
![Speckle Noise](https://www.researchgate.net/publication/221906504/figure/fig3/AS:305311647846404@1449803377512/A-contaminated-image-Fig-5-with-speckle-noise-and-Gaussian-noise-both-having-the.png)

Poisson Noise: Predominant in low-light conditions, this noise is proportional to the brightness of the image.

![Poisson Noise](https://i.stack.imgur.com/wVRcA.jpg)

## Implementing Noise Reduction Techniques
Let's jump into some code! We'll tackle noise reduction techniques for each type of noise. We'll use OpenCV and Matplotlib for this.

First off, let's install the required packages if we haven't already:

```bash
pip install opencv-python matplotlib numpy
```
The code for displaying images is the same for all techniques, so we'll define a function for that.

```python
def display_images(image_with_noise, image_denoised, noise_type):
    """
    Display the original noisy image and the denoised image side-by-side.

    Parameters:
        image_with_noise (numpy.ndarray): The noisy image.
        image_denoised (numpy.ndarray): The denoised image.
        noise_type (str): The type of noise added ("Gaussian", "Salt-and-Pepper", etc.).
    """
    # Convert BGR images to RGB for displaying with matplotlib
    image_with_noise_rgb = cv2.cvtColor(image_with_noise, cv2.COLOR_BGR2RGB)
    image_denoised_rgb = cv2.cvtColor(image_denoised, cv2.COLOR_BGR2RGB)

    # Display the images using matplotlib
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title(f'Image with {noise_type} Noise')
    plt.imshow(image_with_noise_rgb)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Denoised Image')
    plt.imshow(image_denoised_rgb)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
```

### Gaussian Noise Reduction
For Gaussian noise, a common technique is to use a Gaussian filter.

```python
# Add Gaussian noise
noise = np.random.normal(0, 50, image.shape).astype('float32')
image_with_noise = image.astype('float32') + noise
image_with_noise = np.clip(image_with_noise, 0, 255).astype('uint8')

# Apply Gaussian filter
image_denoised = cv2.GaussianBlur(image_with_noise, (5, 5), 0)

# Display the images
display_images(image_with_noise, image_denoised, "Gaussian")
```
![Gaussian Denoise](https://i.ibb.co/rZHC2Lq/gausrem.png)


### Salt-and-Pepper Noise Reduction
For salt-and-pepper noise, we'll use a median filter. This filter replaces each pixel with the median value of its neighboring pixels. It's effective for removing salt-and-pepper noise while preserving edges.

```python
 noise = np.random.choice([0, 1, 2], size=image.shape, p=[0.9, 0.05, 0.05])
    image_with_noise = image * (noise == 0) + 255 * (noise == 1)
    image_with_noise = image_with_noise.astype('uint8')  # Ensure data type is uint8

    # Apply median filter
    image_denoised = cv2.medianBlur(image_with_noise, 5)

    # Display the images
    display_images(image_with_noise, image_denoised, "Salt-and-Pepper")
```
![Salt-and-Pepper Denoise](https://i.ibb.co/z8HZqjz/snprem.png)

### Speckle Noise Reduction
For speckle noise, we'll use a bilateral filter. This filter preserves edges while removing noise by replacing each pixel with a weighted average of its neighboring pixels. The weights are calculated using a Gaussian filter in the spatial domain and a Gaussian function of pixel intensity differences in the intensity domain.

 ```python
 # Add more speckle noise (increase standard deviation to, say, 0.5)
noise = np.random.normal(0, 0.5, image.shape).astype('float32')
image_with_noise = image.astype('float32') + image.astype('float32') * noise
image_with_noise = np.clip(image_with_noise, 0, 255).astype('uint8')

# Apply bilateral filter
image_denoised = cv2.bilateralFilter(image_with_noise, 9, 75, 75)

# Display the images
display_images(image_with_noise, image_denoised, "Speckle")
```
![Speckle Denoise](https://i.ibb.co/6yzGpQp/spekrem.png)

### Poisson Noise Reduction
For Poisson noise, we'll use a non-local means filter. This filter replaces each pixel with a weighted average of its neighboring pixels, where the weights are calculated using a Gaussian function of pixel intensity differences. It's effective for removing Poisson noise while preserving edges.

```python
# Add more Poisson noise (increase lambda to, say, 60)
noise = np.random.poisson(image.astype('float32') / 255.0 * 60) / 60 * 255
image_with_noise = image.astype('float32') + noise
image_with_noise = np.clip(image_with_noise, 0, 255).astype('uint8')

# Apply Non-Local Means Denoising
image_denoised = cv2.fastNlMeansDenoisingColored(image_with_noise, None, 30, 30, 7, 21)

# Display the images
display_images(image_with_noise, image_denoised, "Poisson")
```
![Poisson Denoise](https://i.ibb.co/VwvZg7Q/posrem.png)

## Conclusion
We've explored four common types of image noise and corresponding denoising techniques. These techniques are fundamental in image processing and are commonly used in various applications from medical imaging to computer vision. Happy coding!

Hope we found this useful! Until next time, keep reducing that noise! 📸🔊🔽

