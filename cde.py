"""Command line interface for the curvedataextractor package."""

import argparse
import os
from distutils.version import StrictVersion
import tensorflow as tf
from PIL import Image
import label_map_util_v2
import numpy as np

from axis_legend_detection import get_image_boxes, load_image_into_numpy_array, detect_axes


def validate_input_file_types(inputdir):
    # Check that all images are pngs or jpgs
    bad_files = []
    for (dirpath, dirnames, filenames) in os.walk(inputdir):
        for filename in filenames:
            if not filename.endswith('.png') or filename.endswith('.jpg'):
                bad_files.append(filename)
    if len(bad_files) > 0:
        # warn user some files will be ignored
        print('Warning: the following files are not .png or .jpg files and will be ignored:')
        for filename in bad_files:
            print(filename)


def open_images(inputdir):
    """Get all images in input directory as PIL images.
    Converts any jpgs to pngs.
    
    Args:
        inputdir: directory containing images to be processed
        
    Returns:
        figure_images: list of PIL images"""
    figure_images = []
    for (dirpath, dirnames, filenames) in os.walk(inputdir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                image = Image.open(os.path.join(dirpath, filename))
                image.save(os.path.join(dirpath, filename[:-4] + '.png'))
            figure_images.extend(Image.open(os.path.join(dirpath, filename)))
    return figure_images

def get_dectection_graph(graph_path):
    """Get the detection graph for the object detection model.
    
    Returns:
        detection_graph: the detection graph"""
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v2.io.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def get_category_index(label_map_path):
    """Get the category index for the object detection model.
    
    TODO: move this into model directory

    Returns:
        category_index: the category index"""
    return label_map_util_v2.create_category_index_from_labelmap(label_map_path, use_display_name=True)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.convert("RGB").getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Extract data from line plots.')
    parser.add_argument('input', help='Input image file directory')
    parser.add_argument('output', help='Output JSON file directory')
    parser.add_argument('--model', help='Path to pretrained model.', default='models/research/object_detection/inference_graph/frozen_inference_graph.pb')

    PATH_TO_FROZEN_GRAPH = parser.parse_args().model

    if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
        raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
    
    input_dir = parser.parse_args().input
    validate_input_file_types(input_dir)
    figure_images = open_images(input_dir)
    detection_graph = get_dectection_graph(PATH_TO_FROZEN_GRAPH)
    category_index = get_category_index('cde_labelmap.pbtxt')

    for image in enumerate(figure_images):
        im_width, im_height = image.size
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Actual detection.
        axes = detect_axes(image_np, detection_graph)

        # Extract images of axes
        axis_data, axis_images = get_image_boxes(
            im_width, 
            im_height,
            image_np, 
            axes['detection_boxes'],
            axes['detection_classes'],
            axes['detection_scores'], 
            category_index, 
            use_normalized_coordinates=True
            )

        


    