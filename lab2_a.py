import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the grayscale image
image = cv2.imread('/content/barbara.jpg', cv2.IMREAD_GRAYSCALE)

# --- 1. Discrete Fourier Transform (DFT) ---
def perform_dft(image):
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum

dft_result = perform_dft(image)

# --- 2. Z-Transform ---
def z_transform(image, a=0.9):
    rows, cols = image.shape
    z_transformed = np.zeros_like(image, dtype=np.float32)

    for r in range(rows):
        z_transformed[r, :] = np.cumsum(image[r, :] * (a ** np.arange(cols)))

    for c in range(cols):
        z_transformed[:, c] = np.cumsum(z_transformed[:, c] * (a ** np.arange(rows)))

    return z_transformed

z_transform_result = z_transform(image)

# --- 3. Karhunen–Loève Transform (KLT, using PCA) ---
def kl_transform(image, n_components=20):
    rows, cols = image.shape
    flat_image = image.reshape(-1, cols)

    # Perform PCA (KLT Approximation)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(flat_image)
    reconstructed = pca.inverse_transform(reduced)
    return reconstructed.reshape(rows, cols)

klt_result = kl_transform(image, n_components=50)

# --- Display Results ---
plt.figure(figsize=(10, 10))

# Original Image
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

# DFT Result
plt.subplot(2, 2, 2)
plt.title("DFT Magnitude Spectrum")
plt.imshow(dft_result, cmap='gray')

# Z-Transform Result
plt.subplot(2, 2, 3)
plt.title("Z-Transform Result")
plt.imshow(z_transform_result, cmap='gray')

# KLT Result
plt.subplot(2, 2, 4)
plt.title("KLT (PCA) Result")
plt.imshow(klt_result, cmap='gray')

plt.tight_layout()
plt.show()
