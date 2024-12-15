import cv2
import numpy as np
import matplotlib.pyplot as plt


def image_subtraction(image1, image2):
    return cv2.absdiff(image1, image2)

# Assuming image1 is before the event and image2 is after the event
image1 = cv2.imread('before_event_image.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('after_event_image.jpg', cv2.IMREAD_GRAYSCALE)

subtracted_image = image_subtraction(image1, image2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image1, cmap='gray')
plt.title('Before Event')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(subtracted_image, cmap='gray')
plt.title('Image Subtraction (Changes)')
plt.axis('off')

plt.show()


def embed_watermark(image, watermark, alpha=0.5):
    return cv2.addWeighted(image, 1, watermark, alpha, 0)

# Load a watermark image (ensure the watermark has the same dimensions as the image)
watermark = cv2.imread('watermark_image.png', cv2.IMREAD_GRAYSCALE)
watermarked_image = embed_watermark(image, watermark)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(watermarked_image, cmap='gray')
plt.title('Watermarked Image')
plt.axis('off')

plt.show()


def extract_watermark(original_image, watermarked_image, alpha=0.5):
    return cv2.subtract(watermarked_image, original_image)

# Assuming watermarked_image is the image with watermark, and original_image is the original
extracted_watermark = extract_watermark(image, watermarked_image)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(watermarked_image, cmap='gray')
plt.title('Watermarked Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(extracted_watermark, cmap='gray')
plt.title('Extracted Watermark')
plt.axis('off')

plt.show()


def image_averaging(images):
    return np.mean(images, axis=0).astype(np.uint8)

# Assuming images is a list of noisy images
images = [cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE), cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE), ...]

averaged_image = image_averaging(images)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(images[0], cmap='gray')
plt.title('First Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(averaged_image, cmap='gray')
plt.title('Averaged Image')
plt.axis('off')

plt.show()
