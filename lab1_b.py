import cv2
import numpy as np
import matplotlib.pyplot as plt

def log_transformation(image, c=1):
    # Apply the log transformation with a constant c
    # Ensure that pixel values are in float32 for calculation
    image_log = c * np.log1p(image.astype(np.float32))  # np.log1p is log(1 + image)
    
    # Normalize the output image to the 0-255 range for display
    image_log = np.uint8(np.clip(image_log / np.max(image_log) * 255, 0, 255))
    
    return image_log

# Load a grayscale image
image = cv2.imread('low_and_heigh_contrast_gray_scale_image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply the log transformation
log_image = log_transformation(image)

# Display the original and log-transformed images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(log_image, cmap='gray')
plt.title('Log Transformed Image')
plt.axis('off')

plt.show()


# Experiment with different values of the constant 'c'
c_values = [1, 5, 10, 20]

plt.figure(figsize=(15, 10))

for i, c in enumerate(c_values, 1):
    log_image = log_transformation(image, c)
    
    plt.subplot(2, 3, i)
    plt.imshow(log_image, cmap='gray')
    plt.title(f'Log Transformed Image (c={c})')
    plt.axis('off')

plt.show()
