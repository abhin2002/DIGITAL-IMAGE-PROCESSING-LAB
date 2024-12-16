import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images for before/after comparison
before = cv2.imread('/content/input1.png', cv2.IMREAD_COLOR)
after = cv2.imread('/content/input2.png', cv2.IMREAD_COLOR)

# Ensure both images are in the same color space
before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

# Resize images to the same dimensions
if before.shape != after.shape:
    after = cv2.resize(after, (before.shape[1], before.shape[0]))

# Perform absolute difference
difference = cv2.absdiff(before, after)

# Convert the difference image to grayscale
difference_gray = cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY)

# Apply threshold to highlight significant changes
_, thresh_diff = cv2.threshold(difference_gray, 30, 255, cv2.THRESH_BINARY)

# Load the original image and watermark for watermarking example
original = cv2.imread('/content/lenna.png', cv2.IMREAD_GRAYSCALE)
watermark = cv2.imread('/content/watermark.png', cv2.IMREAD_GRAYSCALE)

# Resize the watermark to match the size of the original image
watermark = cv2.resize(watermark, (original.shape[1], original.shape[0]))

# Normalize watermark intensity to scale it
watermark = cv2.normalize(watermark, None, 0, 50, cv2.NORM_MINMAX)

# Embed the watermark into the original image
watermarked_image = cv2.add(original, watermark)

# Recover the original image by subtracting the watermark
recovered_image = cv2.subtract(watermarked_image, watermark)

# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, stddev=20):
    noise = np.random.normal(mean, stddev, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

# Function to add Salt-and-Pepper noise
def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = image.copy()
    num_salt = np.ceil(salt_prob * image.size).astype(int)
    num_pepper = np.ceil(pepper_prob * image.size).astype(int)

    # Add salt noise (white pixels)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper noise (black pixels)
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

# Function to add Impulse noise
def add_impulse_noise(image, prob=0.01):
    noisy_image = image.copy()
    mask = np.random.choice([0, 1, 2], size=image.shape, p=[prob, prob, 1 - 2 * prob])
    noisy_image[mask == 0] = 0  # Black (pepper)
    noisy_image[mask == 1] = 255  # White (salt)
    return noisy_image

# Generate noisy images
gaussian_noisy_images = [add_gaussian_noise(original) for _ in range(5)]
salt_and_pepper_noisy_images = [add_salt_and_pepper_noise(original) for _ in range(5)]
impulse_noisy_images = [add_impulse_noise(original) for _ in range(5)]

# Perform image averaging for each noise type
def average_images(images):
    accumulator = np.zeros_like(images[0], dtype=np.float32)
    for img in images:
        accumulator += img.astype(np.float32)
    return (accumulator / len(images)).astype(np.uint8)

# Average noisy images
averaged_gaussian = average_images(gaussian_noisy_images)
averaged_salt_and_pepper = average_images(salt_and_pepper_noisy_images)
averaged_impulse = average_images(impulse_noisy_images)

# Display the results in 4 rows and 3 columns
plt.figure(figsize=(18, 20))

# Row 1: Before, After, and Difference Image
plt.subplot(4, 3, 1)
plt.title("Before Image")
plt.imshow(before)
plt.axis('off')

plt.subplot(4, 3, 2)
plt.title("After Image")
plt.imshow(after)
plt.axis('off')

plt.subplot(4, 3, 3)
plt.title("Difference (Thresholded)")
plt.imshow(thresh_diff, cmap='gray')
plt.axis('off')

# Row 2: Watermarked and Recovered Image
plt.subplot(4, 3, 4)
plt.title("Original Image")
plt.imshow(original, cmap='gray')
plt.axis('off')

plt.subplot(4, 3, 5)
plt.title("Watermarked Image")
plt.imshow(watermarked_image, cmap='gray')
plt.axis('off')

plt.subplot(4, 3, 6)
plt.title("Recovered Image")
plt.imshow(recovered_image, cmap='gray')
plt.axis('off')

# Row 3: Noisy Images (Gaussian, Salt-and-Pepper, Impulse)
plt.subplot(4, 3, 7)
plt.title("Gaussian Noisy Image")
plt.imshow(gaussian_noisy_images[0], cmap='gray')
plt.axis('off')

plt.subplot(4, 3, 8)
plt.title("Salt-and-Pepper Noisy Image")
plt.imshow(salt_and_pepper_noisy_images[0], cmap='gray')
plt.axis('off')

plt.subplot(4, 3, 9)
plt.title("Impulse Noisy Image")
plt.imshow(impulse_noisy_images[0], cmap='gray')
plt.axis('off')

# Row 4: Averaged Noisy Images
plt.subplot(4, 3, 10)
plt.title("Averaged Gaussian Noise")
plt.imshow(averaged_gaussian, cmap='gray')
plt.axis('off')

plt.subplot(4, 3, 11)
plt.title("Averaged Salt-and-Pepper Noise")
plt.imshow(averaged_salt_and_pepper, cmap='gray')
plt.axis('off')

plt.subplot(4, 3, 12)
plt.title("Averaged Impulse Noise")
plt.imshow(averaged_impulse, cmap='gray')
plt.axis('off')

# Display results
plt.tight_layout()
plt.show()

