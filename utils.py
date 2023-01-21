import numpy as np
from PIL import Image
from PIL.Image import Image as Image_type
import cv2

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

def image_to_numpy(image):
    (im_width, im_height) = image.size
    return np.array(image.convert("RGB").getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)