import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def image_negative(img):
    return 255 - img

img = cv2.imread("grayscale_image.jpg", cv2.IMREAD_GRAYSCALE)

negative_image = image_negative(img)

# Display the original and negative images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_image, cmap='gray')
plt.title('Negative Image')
plt.axis('off')

plt.show()


imge2 = cv2.imread("low_and_heigh_contrast_gray_scale_image.jpg", cv2.IMREAD_GRAYSCALE)
negative_image2 = image_negative(imge2)

# Display the original and negative images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(imge2, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(negative_image2, cmap='gray')
plt.title('Negative Image')
plt.axis('off')

plt.show()


# Plot the histograms of the original and negative images
plt.figure(figsize=(10, 5))

# Histogram for original image
plt.subplot(1, 2, 1)
plt.hist(imge2.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
plt.title('Original Image Histogram')

# Histogram for negative image
plt.subplot(1, 2, 2)
plt.hist(negative_image.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
plt.title('Negative Image Histogram')

plt.show()


