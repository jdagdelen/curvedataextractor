"""Command line interface for the curvedataextractor package."""

import argparse
import os
import json
from distutils.version import StrictVersion
import tensorflow as tf
# suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import Image
import label_map_util_v2
import numpy as np
from tqdm import tqdm

from figure_object_detection import get_image_boxes, detect_objects
from utils import image_to_numpy
from posterization import preprocess, classify_pixels

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
            figure_images.append(Image.open(os.path.join(dirpath, filename)))
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


def write_extracted_figure_data(objects, clusters, output):
    """Write extracted figure data to JSON file.
    
    Args:
        objects: list of objects detected in figure
        clusters: list of clusters detected in figure
    """
    # Write cluster and object data to JSON files, and save images
    for i, cluster in enumerate(clusters):
        image = Image.fromarray(cluster['image'])
        image.save(os.path.join(output, 'cluster_{}.png'.format(i)))
    for i, object in enumerate(objects):
        image = Image.fromarray(object['image'])
        image.save(os.path.join(output, 'object_{}.png'.format(i)))
    # add filenames to object and cluster data
    for i, cluster in enumerate(clusters):
        cluster['filename'] = 'cluster_{}.png'.format(i)
    for i, object in enumerate(objects):
        object['filename'] = 'object_{}.png'.format(i)


def extract_figure_data(input_dir, output, model='models/research/object_detection/inference_graph/frozen_inference_graph.pb'):
    """Extract figure data from images in input directory.
    
    Args:
        input_dir: directory containing images to be processed
        output: directory to save output files
        model: path to object detection model
    
    Returns:
        objects: list of objects detected in figure
        clusters: list of clusters detected in figure
    """
    validate_input_file_types(input_dir)
    figure_images = open_images(input_dir)
    detection_graph = get_dectection_graph(model)
    category_index = get_category_index('cde_labelmap.pbtxt')

    print('Extracting figure data from {} images...'.format(len(figure_images)))
    for image in tqdm(figure_images):
        im_width, im_height = image.size
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = image_to_numpy(image)
        # Detect areas containing figure axes and legends.
        detected_objects = detect_objects(image_np, detection_graph)

        # Extract images of axes
        print('Extracting objects...')
        object_data, object_images = get_image_boxes(
            im_width, 
            im_height,
            image_np, 
            detected_objects['detection_boxes'],
            detected_objects['detection_classes'],
            detected_objects['detection_scores'], 
            category_index, 
            use_normalized_coordinates=True
            )

        objects = [{'image': object_image, 'data': object_data[i]} for i, object_image in enumerate(object_images)]
        # Remove legends and axes from plot, along with black and white items
        print('Removing objects...')
        posterized_image = preprocess(image_np, object_data)
        
        # Get pixel classes and color palette for posterized image
        print('Classifying pixels...')
        pixel_classes, color_palette, cluster_scores = classify_pixels(posterized_image)
        # Separate out images for each cluster
        clusters = []
        for i in range(len(color_palette)):
            # copy image
            cluster_image = np.copy(posterized_image)
            # write white to pixels not in cluster
            cluster_image[pixel_classes != i] = 255
            # get coordinates of pixels in cluster
            cluster_pixel_coordinates = np.where(pixel_classes == i)
            clusters.append({'image': cluster_image, 'color': color_palette[i], 'score': cluster_scores[i], 'coordinates': cluster_pixel_coordinates})
        # Sort clusters and by score
        clusters = sorted(clusters, key=lambda k: k['score'], reverse=True)
        # close image
        return objects, clusters


if __name__ == '__main__':
    if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
        raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')
    parser = argparse.ArgumentParser(description='Extract data from line plots.')
    parser.add_argument('input', help='Input image file directory')
    parser.add_argument('output', help='Output JSON file directory')
    parser.add_argument('--model', help='Path to pretrained model.', default='models/research/object_detection/inference_graph/frozen_inference_graph.pb')
    args = parser.parse_args()
    objects, clusters = extract_figure_data(args.input, args.output, args.model)