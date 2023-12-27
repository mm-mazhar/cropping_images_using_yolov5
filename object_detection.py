# Import necessary libraries for object detection
import io
import json
from io import BytesIO

import requests
import streamlit as st
import yolov5  # Assuming YOLOv5 is used for object detection
from PIL import Image


def detect_objects(data, MAX_NUM_DETECTION, MIN_IOU_THRES, MIN_SCORE_THRES):
    # Initialize image variable
    image = None

    # Implement object detection logic using YOLOv5 or other pretrained model

    # load pretrained model
    model = yolov5.load("yolov5s.pt")

    # set model parameters
    model.conf = MIN_SCORE_THRES  # NMS confidence threshold
    model.iou = MIN_IOU_THRES  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.max_det = MAX_NUM_DETECTION  # maximum number of detections per image

    if isinstance(data, BytesIO):
        # Read the content of the uploaded image
        image_content = data.read()
        # Convert the image content to a PIL Image
        image = Image.open(io.BytesIO(image_content))

    elif isinstance(data, str):
        # Add a check for file extension
        valid_extensions = [".jpg", ".jpeg", ".png"]
        # Check if the string is a URL
        if (data.startswith("https")) and (
            any(data.lower().endswith(ext) for ext in valid_extensions)
        ):
            # If it's a URL, fetch the image content
            response = requests.get(data, stream=True)
            # Open the image from the response content
            image = Image.open(io.BytesIO(response.content))
        else:
            st.warning("Enter a valid URL.")

    # Perform inference with larger input size
    # detected_objects = model(image, size=1280)
    if image is not None:
        results = model(image)

        # parse results
        predictions = results.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        # # show detection bounding boxes on image
        # st.image(detected_objects.show())

        # Store each detected object's coordinates, visibility, and category in a list of dictionaries
        detected_objects = []
        for i in range(len(boxes)):
            obj_dict = {
                "coordinates": boxes[i].tolist(),
                "visibility": scores[i].item(),
                "category": categories[i].item(),
            }
            detected_objects.append(obj_dict)

        return results, detected_objects
    else:
        st.sidebar.warning("Enter URL.")
