import numpy as np
import sys
sys.path.insert(0, 'models/research')
import tensorflow as tf
from distutils.version import StrictVersion
from PIL import Image
from object_detection.utils import ops as utils_ops


def detect_objects(image, graph):
    """ Runs the axis detection model on the image and returns the detected axes."""
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')
            # Run inference
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict
    

def get_image_boxes(
    im_width,
    im_height,
    image, 
    boxes, 
    classes, 
    scores, 
    category_index, 
    use_normalized_coordinates=True,
    max_boxes_to_draw=20,
    min_score_thresh=.5):
    """ 
    Args:
        im_width: image width
        im_height: image height
        image: image object
        boxes: numpy array of shape [N, 4]
        classes: numpy array of shape [N]. Note that class indices are 1-based,
            and match the keys in the label map.
        scores: numpy array of shape [N] or None.  If scores=None, then
            this function assumes that the boxes to be plotted are groundtruth
            boxes and plot all boxes as black with no classes or scores.
        category_index: a dict containing category dictionaries (each holding
            category index `id` and category name `name`) keyed by category indices.
        use_normalized_coordinates: whether boxes is to be interpreted as
            normalized coordinates or not.
        max_boxes_to_draw: maximum number of boxes to visualize.  If None, draw
            all boxes.
        min_score_thresh: minimum score threshold for a box to be visualized

    Returns:
        json_records: json record of the image
        subimages: cropped images of the detected boxes
    """
    records = []
    subimages = []

    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box
            if use_normalized_coordinates:
                (left, right, top, bottom) = (int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
            else:
                (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]['name']
            else:
                class_name = 'NA'
            json_record = {
                'image_size': {'height':im_height, 'width':im_width},
                'detected_box': {
                    'label':str(class_name), 
                    'score':int(100*scores[i]), 
                    'xmin':left, 
                    'xmax':right, 
                    'ymin':top, 
                    'ymax':bottom}
            }
            records.append(json_record)
            subimages.append(image[top:bottom,left:right,:])
    return records, subimages