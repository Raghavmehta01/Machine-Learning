# SafetyWhat: Object Detection Model By Raghav Mehta (12319917)

## Overview
This project uses YOLO (You Only Look Once) for real-time object detection in video files. It processes each frame of a given video, performs object detection, and saves cropped images of detected objects. Additionally, it tracks object counts and computes inference speed (FPS).

## Features
- Object detection using the YOLOv5 model.
- Crops and saves detected objects as separate images.
- Calculates object counts (e.g., person, car, motorcycle, truck).
- Computes and outputs inference time and FPS for video processing.
- Saves all detections and results to a JSON file.

## Project Structure
- `main.py`: Main script that performs video processing and object detection.
- `project_utils/image_utils.py`: Utility functions for image cropping and saving.
- `detections_output.json`: Output JSON file containing detection data, object counts, and inference speed.
- `data/test_video.mp4`: Sample video used for processing.
- `data/cropped_images/`: Directory where cropped images of detected objects are saved.

## Setup

### Prerequisites
- Python 3.8+
- Install dependencies using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project
1. Ensure that you have the video file (`test_video.mp4`) in the `data` directory.
2. Run the main script:
    ```bash
    python main.py
    ```

This will process the video, perform object detection, and save the cropped images in the `data/cropped_images/` directory. It will also output the detection results, including object counts, FPS, and average inference time.

### Output
- `detections_output.json`: Contains all detection data in JSON format.
- Console Output:
    - Average inference time per frame.
    - Inference speed (FPS).
    - Object counts (number of occurrences for each object type).
