import cv2
import matplotlib.pyplot as plt
import numpy as np

def imshow(img, title=''):
    """Display an image using matplotlib."""
    plt.figure(dpi=200)
    plt.title(title)
    if len(img.shape) == 2:  # Grayscale image
        plt.imshow(img, cmap='gray')
    else:  # Color image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def process_image(image_path):
    """Process the image by inverting colors, converting to grayscale, applying edge detection,
    and drawing contours on the original image."""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image at {image_path}")
        return

    # Invert colors: Black becomes white, white becomes black
    inverted = cv2.bitwise_not(img)
    imshow(inverted, 'Inverted Image')

    # Convert inverted image to grayscale
    gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    imshow(gray, 'Grayscale Image')

    # Apply Gaussian Blur to reduce noise while preserving edges
    blurred = cv2.GaussianBlur(gray, (3, 3), 1)
    imshow(blurred, 'Blurred Image')

    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 25, 125)
    imshow(edges, 'Edges Image')

    # Apply closing operation (dilation followed by erosion) to fill small holes
    kernel = np.ones((3, 3), np.uint8)  # Define a kernel for the morphological operation
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    imshow(closed, 'Closed Image')

    # Find contours from the closed image
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0, 0, 255), 1)  # Draw contours in red

    # Display the image with contours
    imshow(img_with_contours, 'Image with Contours')

if __name__ == '__main__':
    image_path = '/home/xu/object/wangbo/test_2/1.bmp'  # Replace with your image path
    process_image(image_path)
