import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the binary image
image = cv2.imread('/content/lenna.png', cv2.IMREAD_GRAYSCALE)

# Threshold the image to ensure it's binary
# _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Define the kernel for morphological operations
kernel = np.ones((5, 5), np.uint8)

# --- 1. Erosion ---
def perform_erosion(image, kernel):
    eroded = cv2.erode(image, kernel, iterations=1)
    return eroded

eroded_image = perform_erosion(image, kernel)

# --- 2. Dilation ---
def perform_dilation(image, kernel):
    dilated = cv2.dilate(image, kernel, iterations=1)
    return dilated

dilated_image = perform_dilation(image, kernel)

# --- 3. Opening ---
def perform_opening(image, kernel):
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened

opened_image = perform_opening(image, kernel)

# --- 4. Closing ---
def perform_closing(image, kernel):
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed

closed_image = perform_closing(image, kernel)

# --- Display Results ---
plt.figure(figsize=(12, 8))

# Original Binary Image
plt.subplot(3, 2, 1)
plt.title("Original Binary Image")
plt.imshow(image, cmap='gray')

plt.subplot(3, 2, 2)
plt.axis('off')

# Erosion
plt.subplot(3, 2, 3)
plt.title("Erosion")
plt.imshow(eroded_image, cmap='gray')

# Dilation
plt.subplot(3, 2, 4)
plt.title("Dilation")
plt.imshow(dilated_image, cmap='gray')

# Opening
plt.subplot(3, 2, 5)
plt.title("Opening")
plt.imshow(opened_image, cmap='gray')

# Closing
plt.subplot(3, 2, 6)
plt.title("Closing")
plt.imshow(closed_image, cmap='gray')


plt.tight_layout()
plt.show()
