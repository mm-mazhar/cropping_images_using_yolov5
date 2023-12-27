# Cropping Images Using YOLOv5

### Overview
This Streamlit-based application utilizes YOLOv5, a popular object detection model, to perform multiple object detection on images. The detected objects can then be cropped and downloaded individually.

### Features
- Object Detection: Detect multiple objects in an uploaded image or from a provided URL.
- Bounding Boxes: Display bounding boxes around detected objects for visualization.
- Cropping: Crop the most visible object based on confidence scores and download the cropped image.
- Flexibility: Supports both local image upload and image URLs.

### Installation
1 - Clone the repository:

`git clone https://github.com/mm-mazhar/cropping_images_using_yolov5.git`

2 - Navigate to the project directory:

`cd your-repo
`

3- Install dependencies:

`pip install streamlit Pillow yolov5
`

### Usage
Run the Streamlit app:

`streamlit run main.py
`

Visit the provided local URL to interact with the application.

### Project Structure
- main.py: Streamlit app for user interaction.
- object_detection.py: Module for YOLOv5-based object detection.
- image_cropper.py: Module for cropping images based on detected object coordinates.

### Contributors
Mazhar

### License
This project is licensed under the MIT License - see the LICENSE.md file for details.
