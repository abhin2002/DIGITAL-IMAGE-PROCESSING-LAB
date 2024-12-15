import numpy as np
import matplotlib.pyplot as plt

def power_law_transformation(image, gamma):
    """
    Apply power-law (gamma) transformation to an image.

    Parameters:
        image (ndarray): Input image (can be grayscale or color).
        gamma (float): Gamma value for the transformation.

    Returns:
        transformed_image (ndarray): Gamma-transformed image.
    """
    # Normalize the image to the range [0, 1]
    image_normalized = image / 255.0

    # Apply the power-law transformation
    transformed_image = np.power(image_normalized, gamma)

    # Rescale to [0, 255] and convert back to uint8
    transformed_image = np.uint8(transformed_image * 255)

    return transformed_image

# Example usage
if __name__ == "__main__":
    # Load an example image (you can replace this with your image path)

    image = plt.imread('/content/einstein.jpg')

    # Test with different gamma values
    gamma_values = [0.5, 1.0, 2.0]

    plt.figure(figsize=(12, 4))

    for i, gamma in enumerate(gamma_values):
        transformed_image = power_law_transformation(image, gamma)
        plt.subplot(1, len(gamma_values), i+1)
        plt.imshow(transformed_image)
        plt.title(f'Gamma = {gamma}')
        plt.axis('off')

    plt.show()
