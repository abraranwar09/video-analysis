import cv2
import numpy as np

# Load YOLO
weights_path = "yolov4.weights"  # Download from https://pjreddie.com/media/files/yolov4.weights
cfg_path = "yolov4.cfg"  # Download from https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
names_path = "coco.names"  # Download from https://github.com/pjreddie/darknet/blob/master/data/coco.names

print("Loading YOLO model...")
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()

# Check if unconnected_out_layers is a scalar or an array
if isinstance(unconnected_out_layers, np.ndarray) and unconnected_out_layers.ndim > 1:
    output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
else:
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
print("YOLO model and class names loaded successfully.")

# Load video
video_path = "test2.mp4"
output_video_path = "object_detection.mp4"
print(f"Opening video file {video_path}...")
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'avc1')  
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Colors for different classes
colors = np.random.uniform(0, 255, size=(len(classes), 3))

print("Processing video frames...")
frame_count = 0  # Initialize frame counter
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break  # Exit if the video ends

    frame_count += 1
    print(f"Processing frame {frame_count}/{total_frames}...")  # Print frame count

    height, width, _ = frame.shape

    # Convert frame into a YOLO-compatible blob
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.65:  # Confidence threshold
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Perform non-max suppression to remove duplicate bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels on the frame
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            color = colors[class_ids[i]]  # Pick a color for the class
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y = max(y, label_size[1] + 10)
            cv2.rectangle(frame, (x, label_y - label_size[1] - 10), (x + label_size[0], label_y), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Write the frame to the output video
    out.write(frame)

# Release resources
print("Releasing resources...")
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_video_path}")
