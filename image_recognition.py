import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import numpy as np
from PIL import Image
import pytesseract  # For OCR (you'll need to install it)

# Load label map
category_index = label_map_util.create_category_index_from_labelmap('path/to/label_map.pbtxt', use_display_name=True)

# Load saved model
detect_fn = tf.saved_model.load('path/to/saved_model')

# Load image
image_path = 'path/to/image.jpg'
image_np = cv2.imread(image_path)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image_np)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# Perform detection
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_np_with_detections = image_np.copy()

# Visualize detections
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'],
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.30,  # Adjust the minimum score threshold as needed
    agnostic_mode=False
)

# Extract and recognize plate numbers
for i in range(detections['num_detections']):
    if detections['detection_scores'][i] > 0.5:  # Adjust the confidence threshold as needed
        ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
        plate_img = Image.fromarray(image_np[int(ymin*image_np.shape[0]):int(ymax*image_np.shape[0]), int(xmin*image_np.shape[1]):int(xmax*image_np.shape[1])])
        plate_text = pytesseract.image_to_string(plate_img, config='--psm 11')  # PSM 11 is for sparse text
        print(f"Plate Number: {plate_text.strip()}")

# Display the resulting image
cv2.imshow('object detection', image_np_with_detections)
cv2.waitKey(0)

