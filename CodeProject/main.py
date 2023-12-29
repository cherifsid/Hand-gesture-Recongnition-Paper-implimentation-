import cv2
import numpy as np
from matplotlib import pyplot as plt

def find_top_of_middle_finger(image):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            if image[y, x] == 255:
                return x, y
    return None

def draw_slopes(image, ref_point):
    x_ref, y_ref = ref_point
    # Extract the right slope in yellow
    right_slop_x = []
    right_slop_y = []
    size_slop = 1
    rx = 0

    for x in range(1, (image.shape[1] - x_ref) - 1, size_slop):
        x_new = x_ref + x
        rx += 1
        right_slop_x.append(x_new)
        y_new = y_ref + x
        right_slop_y.append(y_new)

    # Extract the left slope in green
    left_slop_x = []
    left_slop_y = []
    size_slop = 1
    lx = 0
    left_size = x_ref - size_slop

    while left_size != 0:
        left_size -= 1
        lx += 1
        x_new = x_ref - lx
        left_slop_x.append(x_new)
        y_new = y_ref + lx
        left_slop_y.append(y_new)

    # Combine left and right slopes to form a scanned path
    scanned_path_x = left_slop_x + right_slop_x
    scanned_path_y = left_slop_y + right_slop_y

    return scanned_path_x, scanned_path_y, left_slop_x, left_slop_y, right_slop_x, right_slop_y

def count_flips(image, scanned_path_x, scanned_path_y):
    # Initialize flips to -1 to account for the initial state
    flips = -1
    # Loop through the scanned path to count flips
    for i in range(1, len(scanned_path_x)):
        x_prev, y_prev = scanned_path_x[i - 1], scanned_path_y[i - 1]
        x_curr, y_curr = scanned_path_x[i], scanned_path_y[i]

        # Check for flips (transitions from white to black or vice versa)
        if image[y_prev, x_prev] != image[y_curr, x_curr]:
            flips += 1

    return flips

def calculate_solidity(image, largest_contour):
    # Calculate solidity (S)
    area_s = cv2.contourArea(largest_contour)
    convex_hull = cv2.convexHull(largest_contour)
    convex_hull_area = cv2.contourArea(convex_hull)
    solidity = area_s / convex_hull_area if convex_hull_area > 0 else 0

    return solidity

def plot_processing_steps(image_path):
    image = cv2.imread(image_path)
    image2 = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found or unable to load.")
        return

    # Convert to YCbCr color space
    image_ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Define thresholds for filtering in YCbCr color space
    min_YCrCb = np.array([0, 135, 85], np.uint8)
    max_YCrCb = np.array([255, 180, 135], np.uint8)

    # Thresholding the image to get only skin color
    skin_ycbcr = cv2.inRange(image_ycbcr, min_YCrCb, max_YCrCb)

    # Convert to binary image with threshold 0.5
    _, skin_binary = cv2.threshold(skin_ycbcr, 128, 255, cv2.THRESH_BINARY)

    # Apply median filter
    skin_median = cv2.medianBlur(skin_binary, 5)

    # Remove small objects (less than 300 pixels in area)
    num_labels, labels_im = cv2.connectedComponents(skin_median)
    for i in range(1, num_labels):
        if np.sum(labels_im == i) < 300:
            skin_median[labels_im == i] = 0

    # Eroding the image to sharpen the object
    kernel = np.ones((3, 3), np.uint8)
    skin_eroded = cv2.erode(skin_median, kernel, iterations=1)

    # Fill any holes surrounded by white pixels
    skin_filled = cv2.morphologyEx(skin_eroded, cv2.MORPH_CLOSE, kernel)

    # Find the largest object (hand)
    contours, _ = cv2.findContours(skin_filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate the eighth distance
        eighth_distance = int(h / 8)

        # Move down from the top of the middle finger by an eighth distance
        top_of_middle_finger = find_top_of_middle_finger(skin_filled)
        ref_point = (top_of_middle_finger[0], top_of_middle_finger[1] + eighth_distance)

        # Draw right and left slopes
        scanned_path_x, scanned_path_y, left_slop_x, left_slop_y, right_slop_x, right_slop_y = draw_slopes(image, ref_point)

        # Count flips along the scanned path
        flips = count_flips(skin_filled, scanned_path_x, scanned_path_y)
        print("Number of flips:", flips)

        # Calculate solidity
        solidity = calculate_solidity(skin_filled, largest_contour)
        print("Solidity:", solidity)

        # Check for gesture recognition
        if solidity > 0.8:
            gesture_prediction = 0  # Recognized as zero based on solidity
        else:
            gesture_prediction = flips / 2  # Predict based on flips for numbers 1 to 5

        print("Hand gesture prediction:", gesture_prediction)

        # Plotting all the steps
        plt.figure(figsize=(15, 10))
        # Original Image
        plt.subplot(2, 4, 1)
        plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")

        # YCbCr Color Space
        plt.subplot(2, 4, 2)
        plt.imshow(image_ycbcr, cmap='gray')
        plt.title("YCbCr Color Space")

        # YCbCr Filtering
        plt.subplot(2, 4, 3)
        plt.imshow(skin_ycbcr, cmap='gray')
        plt.title("YCbCr Filtering")

        # Binary Image
        plt.subplot(2, 4, 4)
        plt.imshow(skin_binary, cmap='gray')
        plt.title("Binary Image")

        # Median Filter Applied
        plt.subplot(2, 4, 5)
        plt.imshow(skin_median, cmap='gray')
        plt.title("Median Filter Applied")

        # Resultat final Segmentation
        plt.subplot(2, 4, 6)
        plt.imshow(cv2.cvtColor(skin_filled, cv2.COLOR_BGR2RGB))
        plt.title("Resultat final Segmentation")

        # Processed Image with Slopes
        plt.subplot(2, 4, 7)
        plt.imshow(cv2.cvtColor(skin_filled, cv2.COLOR_BGR2RGB))

        # Top of Middle Finger
        plt.plot(top_of_middle_finger[0], top_of_middle_finger[1], 'ro', label='Dessus du majeur')
        # Reference Point
        plt.plot(ref_point[0], ref_point[1], 'bo', label='Point de Reference ')
        # Left Slope in Green
        plt.plot(left_slop_x, left_slop_y, color='green', linewidth=2, label='Pente gauche')
        # Right Slope in Yellow
        plt.plot(right_slop_x, right_slop_y, color='yellow', linewidth=2, label='Pente droite')
        plt.title("Application des droits")
        plt.legend(loc='center left', fontsize='xx-small')

        # Flips and Gesture Prediction
        plt.subplot(2, 4, 8)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.text(0.5, 0.5, f"Nombre de  Flips: {flips}", ha='center', va='center', fontsize=12, color='blue')
        plt.text(120, 30, f" {gesture_prediction}", ha='center', va='center', fontsize=24, color='green')
        plt.axis('off')

        plt.tight_layout()
        # Adjusting the label size and position
        plt.show()

# Replace this with your actual image path
import os

# Path to the directory containing images
directory_path = "hands"

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith(".png") or filename.endswith(".jpg") :  # Check if the file is a PNG image
        file_path = os.path.join(directory_path, filename)
        plot_processing_steps(file_path)



