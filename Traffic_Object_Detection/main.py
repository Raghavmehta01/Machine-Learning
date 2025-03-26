import cv2
import os
import time
import json
from project_utils.image_utils import crop_image, save_image
from ultralytics import YOLO

VIDEO_PATH = "/Users/raghavmehta/Desktop/Coding/projects py/SafetyWhat/data/test_video.mp4"
CROPPED_IMAGES_DIR = "data/cropped_images/"
FRAME_LIMIT = 100

os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)

object_counts = {"person": 0, "car": 0, "motorcycle": 0, "truck": 0}
detections_data = [] 

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    return cap

def process_frame(frame, model, frame_idx):
    start_time = time.time()

    frame = cv2.resize(frame, (640, 480))
    results = model(frame)
    detections = results[0].boxes.data

    frame_detections = []

    if detections is not None:
        for idx, det in enumerate(detections):
            x1, y1, x2, y2, _, class_id = map(int, det[:6])
            label = results[0].names[class_id]
            
            # Update object counts
            if label in object_counts:
                object_counts[label] += 1

            # Crop and save image
            cropped_img = crop_image(frame, (x1, y1, x2, y2))
            save_path = os.path.join(CROPPED_IMAGES_DIR, f"{label}_frame{frame_idx}_obj{idx}.jpg")
            save_image(cropped_img, save_path)

            # Store detection details
            frame_detections.append({
                "object": label,
                "id": idx + 1,
                "bbox": [x1, y1, x2, y2],
                "subobject": {
                    "object": f"sub_{label}",
                    "id": idx + 1,
                    "bbox": [x1 + 10, y1 + 10, x2 + 10, y2 + 10]
                }
            })

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    inference_time = time.time() - start_time
    return inference_time, frame_detections, frame

def main():
    model = YOLO("yolov5n.pt")
    model.to('cpu')
    model.fuse()
    model.half()
    cap = load_video(VIDEO_PATH)
    frame_idx = 0
    total_inference_time = 0
    processed_frames = 0

    while cap.isOpened() and frame_idx < FRAME_LIMIT:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the current frame
        inference_time, frame_detections, annotated_frame = process_frame(frame, model, frame_idx)
        detections_data.extend(frame_detections)
        total_inference_time += inference_time
        processed_frames += 1

        # Display the annotated frame
        cv2.imshow("Object Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save results to JSON and print output statistics
    if processed_frames > 0:
        average_inference_time = total_inference_time / processed_frames
        fps = 1 / average_inference_time

        output_data = {
            "detections": detections_data,
            "object_counts": object_counts,
            "average_inference_time": average_inference_time,
            "fps": fps
        }

        with open('detections_output.json', 'w') as json_file:
            json.dump(output_data, json_file, indent=4)

        print(json.dumps(output_data, indent=4))
        print(f"Average Inference Time per Frame: {average_inference_time:.4f} seconds")
        print(f"Inference Speed (FPS): {fps:.2f} FPS")
        print(f"Object counts: {object_counts}")
        print("Processing complete.")

if __name__ == "__main__":
    main()







# import cv2
# import os
# import time
# import json
# from project_utils.image_utils import crop_image, save_image
# from ultralytics import YOLO

# VIDEO_PATH = "/Users/raghavmehta/Desktop/Coding/projects py/SafetyWhat/data/test_video.mp4"
# CROPPED_IMAGES_DIR = "data/cropped_images/"
# FRAME_LIMIT = 100

# os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)

# object_counts = {"person": 0, "car": 0, "motorcycle": 0, "truck": 0}
# detections_data = []

# def load_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise ValueError(f"Error opening video file: {video_path}")
#     return cap

# def process_frame(frame, model, frame_idx):
#     start_time = time.time()

#     frame = cv2.resize(frame, (640, 480))
#     results = model(frame)
#     detections = results[0].boxes.data

#     frame_detections = []

#     if detections is not None:
#         for idx, det in enumerate(detections):
#             x1, y1, x2, y2, _, class_id = map(int, det[:6])
#             label = results[0].names[class_id]
            
#             if label in object_counts:
#                 object_counts[label] += 1

#             cropped_img = crop_image(frame, (x1, y1, x2, y2))
#             save_path = os.path.join(CROPPED_IMAGES_DIR, f"{label}_frame{frame_idx}_obj{idx}.jpg")
#             save_image(cropped_img, save_path)

#             frame_detections.append({
#                 "object": label,
#                 "id": idx + 1,
#                 "bbox": [x1, y1, x2, y2],
#                 "subobject": {
#                     "object": f"sub_{label}",
#                     "id": idx + 1,
#                     "bbox": [x1 + 10, y1 + 10, x2 + 10, y2 + 10]
#                 }
#             })

#     inference_time = time.time() - start_time
#     return inference_time, frame_detections

# def main():
#     model = YOLO("yolov5n.pt")
#     model.to('cpu')
#     model.fuse()
#     model.half()
#     cap = load_video(VIDEO_PATH)
#     frame_idx = 0
#     total_inference_time = 0
#     processed_frames = 0

#     while cap.isOpened() and frame_idx < FRAME_LIMIT:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         inference_time, frame_detections = process_frame(frame, model, frame_idx)
#         detections_data.extend(frame_detections)
#         total_inference_time += inference_time
#         processed_frames += 1
        
#         frame_idx += 1

#     cap.release()

#     if processed_frames > 0:
#         average_inference_time = total_inference_time / processed_frames
#         fps = 1 / average_inference_time

#         output_data = {
#             "detections": detections_data,
#             "object_counts": object_counts,
#             "average_inference_time": average_inference_time,
#             "fps": fps
#         }

#         with open('detections_output.json', 'w') as json_file:
#             json.dump(output_data, json_file, indent=4)

#         print(json.dumps(output_data, indent=4))

#         print(f"Average Inference Time per Frame: {average_inference_time:.4f} seconds")
#         print(f"Inference Speed (FPS): {fps:.2f} FPS")
#         print(f"Object counts: {object_counts}")
#         print("Processing complete.")

# if __name__ == "__main__":
#     main()
