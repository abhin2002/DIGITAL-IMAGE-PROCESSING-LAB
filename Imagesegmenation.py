import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('/content/lenna.png', cv2.IMREAD_GRAYSCALE)

# --- 1. Thresholding Segmentation ---
def threshold_segmentation(image, thresh_value=127):
    _, binary = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    return binary

thresholded_image = threshold_segmentation(image)

# --- 2. Region-Based Segmentation (Watershed) ---
def region_based_segmentation(image):
    # Convert to binary image
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)

    # Add 1 to all labels so that the background is 1 instead of 0
    markers = markers + 1

    # Mark the unknown region as 0
    markers[unknown == 255] = 0

    # Watershed algorithm
    color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(color_image, markers)
    color_image[markers == -1] = [255, 0, 0]  # Mark boundary in red

    return color_image

region_segmented_image = region_based_segmentation(image)

# --- 3. Edge-Based Segmentation ---
def edge_based_segmentation(image):
    edges = cv2.Canny(image, 100, 200)  # Canny Edge Detection
    return edges

edge_segmented_image = edge_based_segmentation(image)

# --- Display Results ---
plt.figure(figsize=(10, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

# Thresholding
plt.subplot(2, 2, 2)
plt.title("Thresholding")
plt.imshow(thresholded_image, cmap='gray')

# Region-Based Segmentation
plt.subplot(2, 2, 3)
plt.title("Region-Based (Watershed)")
plt.imshow(cv2.cvtColor(region_segmented_image, cv2.COLOR_BGR2RGB))

# Edge-Based Segmentation
plt.subplot(2, 2, 4)
plt.title("Edge-Based (Canny)")
plt.imshow(edge_segmented_image, cmap='gray')

plt.tight_layout()
plt.show()
