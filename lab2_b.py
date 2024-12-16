import cv2
import numpy as np
from skimage import exposure
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt

# Load a grayscale image
image = cv2.imread("/content/lenna.png", cv2.IMREAD_GRAYSCALE)

# Load a reference image for histogram matching
reference_image = cv2.imread("/content/barbara.jpg", cv2.IMREAD_GRAYSCALE)

# --- 1. Histogram Equalization ---
def histogram_equalization(image):
    equalized = cv2.equalizeHist(image)
    return equalized

equalized_image = histogram_equalization(image)

# --- 2. Histogram Matching ---
def histogram_matching(source, reference):
    matched = exposure.match_histograms(source, reference)
    return np.uint8(matched)

matched_image = histogram_matching(image, reference_image)

# --- 3. Contrast Enhancement (CLAHE) ---
def contrast_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(image)
    return enhanced

enhanced_image = contrast_enhancement(image)

# --- 4. Entropy Calculation ---
def calculate_entropy(image):
    return shannon_entropy(image)

# Entropy Calculations
entropy_original = calculate_entropy(image)
entropy_reference = calculate_entropy(reference_image)
entropy_equalized = calculate_entropy(equalized_image)
entropy_matched = calculate_entropy(matched_image)
entropy_enhanced = calculate_entropy(enhanced_image)

# --- Display Results ---
plt.figure(figsize=(18, 18))


# Row 1: Original Image and Histogram
plt.subplot(4, 2, 1)
plt.title(f"Original Image\nEntropy: {entropy_original:.2f}")
plt.imshow(image, cmap='gray')

plt.subplot(4, 2, 2)
plt.title("Original Histogram")
plt.hist(image.ravel(), bins=256, color='blue', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# Row 2: Reference Image and Histogram
plt.subplot(4, 2, 3)
plt.title(f"Reference Image\nEntropy: {entropy_reference:.2f}")
plt.imshow(reference_image, cmap='gray')

plt.subplot(4, 2, 4)
plt.title("Reference Histogram")
plt.hist(reference_image.ravel(), bins=256, color='orange', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

# Row 3: Histogram Equalization and Matching
plt.subplot(4, 2, 5)
plt.title(f"Histogram Equalization\nEntropy: {entropy_equalized:.2f}")
plt.imshow(equalized_image, cmap='gray')

plt.subplot(4, 2, 6)
plt.title(f"Histogram Matching\nEntropy: {entropy_matched:.2f}")
plt.imshow(matched_image, cmap='gray')

# Row 4: Contrast Enhanced and Histogram
plt.subplot(4, 2, 7)
plt.title(f"Contrast Enhanced\nEntropy: {entropy_enhanced:.2f}")
plt.imshow(enhanced_image, cmap='gray')

plt.subplot(4, 2, 8)
plt.title("Enhanced Histogram (CLAHE)")
plt.hist(enhanced_image.ravel(), bins=256, color='green', alpha=0.7)
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()