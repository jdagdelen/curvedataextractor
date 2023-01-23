import numpy as np
import cv2
from PIL import Image
from PIL.Image import Image as Image_type
from utils import rgb2hex, remove_area


def preprocess(image, areas_to_remove, black_threshold=124):
    """Converts image to HSV, removes all black and white objects and other objects.
    
    Args:
        image: Numpy array of RGB values.
        areas_to_remove: A list of areas to remove as dicts with key 'detected_box'
        
    Returns:
        Image with legends and black and white details removed as a numpy array of RGB values.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # If background is black, invert colors
    if np.mean(hsv[:, :, 2]) < 100:
        image = 255 - image
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Remove all black and white objects from orignal image
    # image[(hsv[:, :, 1] < black_threshold) | (hsv[:, :, 2] < black_threshold)] = [255, 255, 255]
    
    # Remove legends
    for area in areas_to_remove:
        image = remove_area(image, area['detected_box'])
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
        A numpy array of RGB values with each pixel replaced by the color of its cluster, a list of colors and a list of cluster scores.
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
    best_n_colors = None
    best_labels = None
    best_centers = None
    n_colors = 1
    while (best_score is None or best_score > color_threshold) and n_colors < 10:
        # Apply kmeans
        print('Trying {} colors'.format(n_colors))
        print(f'Best score: {best_score} (n_colors={n_colors})')
        _, labels, centers = cv2.kmeans(pixel_vals, n_colors, criteria, criteria, attempts, flags)
        # flatten the labels array
        labels = labels.flatten()
        # For each class, calculate mean squared distance of the pixels in the class to their cluster center
        mean_cluster_score = np.mean([np.mean(np.sum((pixel_vals[labels == i] - centers[i])**2, axis=1)) for i in range(n_colors)])
        if best_score is None or mean_cluster_score < best_score:
            best_score = mean_cluster_score
            best_n_colors = n_colors
            best_labels = labels
            best_centers = centers
        n_colors += 1
    # claculate mean squared distance score for pixels in each cluster
    cluster_scores = [np.mean(np.sum((pixel_vals[best_labels == i] - best_centers[i])**2, axis=1)) for i in range(best_n_colors)]
    # Convert data into 8-bit values
    centers = np.uint8(best_centers)
    # Convert all pixels to the color of the centroids
    segmented_image = centers[best_labels]
    # Reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # # list of colors in the segmented image as hex color codes
    # palette = [rgb2hex(rgb) for rgb in centers]
    
    return segmented_image, centers, cluster_scores