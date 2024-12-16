import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
image = cv2.imread("/content/lenna.png", cv2.IMREAD_GRAYSCALE)

# --- 1. 2D Discrete Fourier Transform (DFT) and Inverse ---
def apply_dft(image):
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
    return dft, dft_shift, magnitude_spectrum

def apply_idft(dft_shift):
    dft_ishift = np.fft.ifftshift(dft_shift)
    reconstructed_image = np.abs(np.fft.ifft2(dft_ishift))
    return reconstructed_image

dft, dft_shift, magnitude_spectrum = apply_dft(image)
reconstructed_image = apply_idft(dft_shift)

# --- 2. Design Filters in the Frequency Domain ---
def create_filter(shape, filter_type, d0, w=10):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.float32)

    if filter_type == "low-pass":
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) <= d0:
                    mask[i, j] = 1

    elif filter_type == "high-pass":
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - crow) ** 2 + (j - ccol) ** 2) > d0:
                    mask[i, j] = 1

    elif filter_type == "band-pass":
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                if d0 - w / 2 < d < d0 + w / 2:
                    mask[i, j] = 1

    return mask

# Apply filters
low_pass_filter = create_filter(image.shape, "low-pass", d0=50)
high_pass_filter = create_filter(image.shape, "high-pass", d0=50)
band_pass_filter = create_filter(image.shape, "band-pass", d0=50, w=20)

low_pass_result = apply_idft(dft_shift * low_pass_filter)
high_pass_result = apply_idft(dft_shift * high_pass_filter)
band_pass_result = apply_idft(dft_shift * band_pass_filter)

# --- 3. Homomorphic Filtering ---
def homomorphic_filter(image, gamma_low=0.5, gamma_high=1.5, c=1, d0=50):
    rows, cols = image.shape
    image_log = np.log1p(np.float32(image))  # Log transform
    dft = np.fft.fft2(image_log)
    dft_shift = np.fft.fftshift(dft)

    # Create high-pass filter
    filter = np.zeros((rows, cols), np.float32)
    crow, ccol = rows // 2, cols // 2
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            filter[i, j] = (gamma_high - gamma_low) * (1 - np.exp(-c * (d ** 2 / d0 ** 2))) + gamma_low

    filtered_dft = dft_shift * filter
    dft_ishift = np.fft.ifftshift(filtered_dft)
    result = np.exp(np.real(np.fft.ifft2(dft_ishift))) - 1  # Exponentiate and subtract 1
    result = np.uint8(np.clip(result, 0, 255))  # Clip to valid range
    return result

homomorphic_result = homomorphic_filter(image)

# --- Display Results in Desired Layout ---
plt.figure(figsize=(15, 18))

# Row 1: Original Image and 2D DFT
plt.subplot(6, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(6, 2, 2)
plt.title("2D DFT Magnitude Spectrum")
plt.imshow(magnitude_spectrum, cmap='gray')

# Row 2: 2D DFT and Inverse Image
plt.subplot(6, 2, 3)
plt.title("2D DFT Magnitude Spectrum")
plt.imshow(magnitude_spectrum, cmap='gray')

plt.subplot(6, 2, 4)
plt.title("Reconstructed Image (Inverse DFT)")
plt.imshow(reconstructed_image, cmap='gray')

# Row 3: Low-Pass Mask and Low-Pass Result
plt.subplot(6, 2, 5)
plt.title("Low-Pass Filter Mask")
plt.imshow(low_pass_filter, cmap='gray')

plt.subplot(6, 2, 6)
plt.title("Low-Pass Filter Result")
plt.imshow(low_pass_result, cmap='gray')

# Row 4: High-Pass Mask and High-Pass Result
plt.subplot(6, 2, 7)
plt.title("High-Pass Filter Mask")
plt.imshow(high_pass_filter, cmap='gray')

plt.subplot(6, 2, 8)
plt.title("High-Pass Filter Result")
plt.imshow(high_pass_result, cmap='gray')

# Row 5: Band-Pass Mask and Band-Pass Result
plt.subplot(6, 2, 9)
plt.title("Band-Pass Filter Mask")
plt.imshow(band_pass_filter, cmap='gray')

plt.subplot(6, 2, 10)
plt.title("Band-Pass Filter Result")
plt.imshow(band_pass_result, cmap='gray')

# Row 6: Original Image and Homomorphic Filter Result
plt.subplot(6, 2, 11)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(6, 2, 12)
plt.title("Homomorphic Filtering Result")
plt.imshow(homomorphic_result, cmap='gray')

plt.tight_layout()
plt.show()
