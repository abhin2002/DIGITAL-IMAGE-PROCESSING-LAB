import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the color image
image = cv2.imread('/content/lenna.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- 1. Color Space Conversions ---
# Convert to HSI
image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # HSV is analogous to HSI

# Convert to YCbCr
image_ycbcr = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

# --- 2. Color Histogram Equalization ---
def equalize_color_histogram(image):
    # Convert RGB to YCrCb
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    # Equalize the histogram of the Y channel
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    # Convert back to RGB
    equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return equalized

equalized_image = equalize_color_histogram(image_rgb)

# --- 3. Color Edge Detection ---
def color_edge_detection(image, method='canny'):
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == 'canny':
        edges = cv2.Canny(gray, 100, 200)  # Canny Edge Detection
    elif method == 'sobel':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
        edges = np.uint8(edges)
    return edges

edges_canny = color_edge_detection(image_rgb, method='canny')
edges_sobel = color_edge_detection(image_rgb, method='sobel')

# --- Display Results ---
plt.figure(figsize=(15, 18))

# Original Image
plt.subplot(4, 3, 1)
plt.title("Original Image")
plt.imshow(image_rgb)
plt.axis('off')

# HSI Image
plt.subplot(4, 3, 2)
plt.title("HSI (HSV Equivalent)")
plt.imshow(image_hsv)
plt.axis('off')

# YCbCr Image
plt.subplot(4, 3, 3)
plt.title("YCbCr")
plt.imshow(image_ycbcr)
plt.axis('off')

# Original Histogram
plt.subplot(4, 3, 4)
plt.title("Original Image Histogram")
for i, color in enumerate(['r', 'g', 'b']):
    plt.hist(image_rgb[:, :, i].ravel(), bins=256, color=color, alpha=0.6)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# Equalized Image
plt.subplot(4, 3, 5)
plt.title("Equalized Image")
plt.imshow(equalized_image)
plt.axis('off')

# Equalized Histogram
plt.subplot(4, 3, 6)
plt.title("Equalized Histogram")
for i, color in enumerate(['r', 'g', 'b']):
    plt.hist(equalized_image[:, :, i].ravel(), bins=256, color=color, alpha=0.6)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# Canny Edge Detection
plt.subplot(4, 3, 7)
plt.title("Canny Edge Detection")
plt.imshow(edges_canny, cmap='gray')
plt.axis('off')

# Sobel Edge Detection
plt.subplot(4, 3, 8)
plt.title("Sobel Edge Detection")
plt.imshow(edges_sobel, cmap='gray')
plt.axis('off')

# Original Image for Reference
plt.subplot(4, 3, 9)
plt.title("Original Image (Reference)")
plt.imshow(image_rgb)
plt.axis('off')

plt.tight_layout()
plt.show()
