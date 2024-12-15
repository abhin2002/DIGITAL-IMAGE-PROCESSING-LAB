import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

# Apply Mean filter
image = cv2.imread('path_to_image', cv2.IMREAD_GRAYSCALE)
mean_filtered_image = mean_filter(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(mean_filtered_image, cmap='gray')
plt.title('Mean Filtered Image')
plt.axis('off')

plt.show()


def median_filter(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

# Apply Median filter
median_filtered_image = median_filter(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(median_filtered_image, cmap='gray')
plt.title('Median Filtered Image')
plt.axis('off')

plt.show()


def gaussian_filter(image, kernel_size=3, sigma=1):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

# Apply Gaussian filter
gaussian_filtered_image = gaussian_filter(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gaussian_filtered_image, cmap='gray')
plt.title('Gaussian Filtered Image')
plt.axis('off')

plt.show()


def sharpening_filter(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Apply Sharpening filter
sharpened_image = sharpening_filter(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.show()


def laplacian_4(image):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return cv2.filter2D(image, -1, kernel)

# Apply 4-connected Laplacian filter
laplacian_4_image = laplacian_4(image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(laplacian_4_image, cmap='gray')
plt.title('4-Connected Laplacian Image')
plt.axis('off')

plt.show()
