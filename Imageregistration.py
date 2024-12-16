import cv2
import numpy as np
from matplotlib import pyplot as plt

def register_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Use FLANN (Fast Library for Approximate Nearest Neighbors) for matching descriptors
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)  # Higher number of checks will improve results but slower
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to keep good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show matches
    plt.figure(figsize=(10, 6))
    plt.imshow(img_matches)
    plt.title("ORB Feature Matching")
    plt.show()

    # Find the homography matrix
    if len(good_matches) > 4:  # At least 4 matches are needed to compute homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Compute the homography matrix using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp the first image to align with the second image
        h, w = gray2.shape
        aligned_image = cv2.warpPerspective(image1, M, (w, h))

        return aligned_image, M
    else:
        print("Not enough matches found!")
        return None, None

# Example usage
image1 = cv2.imread('/content/iR1.png')  # Path to the first image
image2 = cv2.imread('/content/iR2.png')  # Path to the second image

aligned_image, homography_matrix = register_images(image1, image2)

if aligned_image is not None:
    # Show the aligned image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
    plt.title("Aligned Image")
    plt.show()
