import numpy as np
import cv2
from PIL import Image
from PIL.Image import Image as Image_type


def rgb2hex(color):
    [r,g,b] = color
    rgb = (int(np.round(255*r)), int(np.round(255*g)), int(np.round(255*b)))
    return '#%02x%02x%02x' % rgb


def read_image(image):
    """Return image as numpy array of RGB values.
    
    If image is a string, it is assumed to be a path to an image file.
    If image is already a numpy array, it is returned as is.
    
    Args:
        image: Either a string or a numpy array.
        
    Returns:
        A numpy array of RGB values.
    """
    if type(image) is not Image_type:
        img = Image.open(image)
    else:
        img = image
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    im = np.asarray(img)
    im = im[:, :, :3]
    return im


def remove_area(image, box, padding=5, color=(255, 255, 255)):
    """Removes area in box from image.
    
    Args:
        image: Numpy array of RGB values.
        box: A dictionary with keys 'xmin', 'ymin', 'xmax', 'ymax'.
        padding: Number of pixels to add to box.
        color: Color to fill box with.

    Returns:
        A numpy array of RGB values with area in box removed.
    """

    return cv2.rectangle(image, (box['xmin']-padding, box['ymin']-padding), (box['xmax']+padding, box['ymax']+padding), color, -1)


def preprocess(image, legends):
    """Converts image to HSV, removes all black and white objects and detected legends.
    
    Args:
        image: Numpy array of RGB values.
        legends: A list of legend dictionaries with key 'detected_box'
        
    Returns:
        Image with legends and black and white details removed as a numpy array of RGB values.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Remove all black and white objects from orignal image
    image[(hsv[:, :, 1] < 124) | (hsv[:, :, 2] < 124)] = [255, 255, 255]
    
    # Remove legends
    for legend in legends:
        image = remove_area(image, legend['detected_box'])
    return image


COLORS= [
    (0.99, 0.99, 0.99), # white
    (0.99, 0.01, 0.01), # red
    (0.01, 0.99, 0.01), # green
    (0.01, 0.01, 0.99), # blue
    (0.99, 0.99, 0.01), # yellow
    (0.99, 0.01, 0.99), # magenta
    (0.01, 0.99, 0.99), # cyan
    (0.01, 0.01, 0.01)] # black


def classify_pixels(image, n_colors=10, color_threshold=0.1, max_iter=20, epsilon=0.05, attempts=10):
    """Uses k-means clustering to classify pixels by color class.
    
    Args:
        image: Numpy array of RGB values.
        n_colors: Number of colors to classify pixels into.
    
    Returns:
        A numpy array of the same shape as image with each pixel classified by color class.
    """

    # Reshape image to be a list of pixels
    pixel_vals = image.reshape((-1, 3))
    # Convert to float type
    pixel_vals = np.float32(pixel_vals)
    # Define kmeans stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    # Set flags
    flags = cv2.KMEANS_RANDOM_CENTERS
    
    best_score = None
    n_colors = 1
    while best_score is None or best_score > color_threshold:
        # Apply kmeans
        compactness, labels, centers = cv2.kmeans(pixel_vals, n_colors, criteria, criteria, attempts, flags)
        if best_score is None or compactness < best_score:
            best_score = compactness
            n_colors += 1

    # Convert data into 8-bit values
    centers = np.uint8(centers)
    # Flatten the labels array
    labels = labels.flatten()
    # Convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # Reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # list of colors in the segmented image as hex color codes
    palette = [rgb2hex(rgb) for rgb in centers]
    return segmented_image, palette