import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLO
weights_path = "yolov4.weights"  # Ensure you have YOLOv4 weights
cfg_path = "yolov4.cfg"
names_path = "coco.names"

print("Loading YOLO model...")
try:
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    raise

# Check if unconnected_out_layers is a scalar or an array
if isinstance(unconnected_out_layers, np.ndarray):
    output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
else:
    output_layers = [layer_names[unconnected_out_layers - 1]]

print("Loading class names...")
try:
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print("Class names loaded successfully.")
except FileNotFoundError:
    print(f"File not found: {names_path}")
    raise

# Load video
video_path = "test2.mp4"
output_video_path = "density_heatmap.mp4"
print(f"Opening video file {video_path}...")
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'avc1')  
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Create empty heatmap accumulator
heatmap_accumulator = np.zeros((frame_height, frame_width), dtype=np.float32)

print("Processing video frames...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Confidence threshold
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = center_x - w // 2, center_y - h // 2
                heatmap_accumulator[y:y+h, x:x+w] += 1  # Accumulate object presence

print("Normalizing heatmap...")
heatmap_accumulator = (heatmap_accumulator - np.min(heatmap_accumulator)) / (np.max(heatmap_accumulator) - np.min(heatmap_accumulator))

print("Converting heatmap to color map...")
heatmap_color = cv2.applyColorMap((heatmap_accumulator * 255).astype(np.uint8), cv2.COLORMAP_JET)

# Rewind video and overlay heatmap on each frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("Overlaying heatmap on video frames...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)
    out.write(overlay)

# Release resources
print("Releasing resources...")
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved as {output_video_path}")
