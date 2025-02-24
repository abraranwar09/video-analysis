import cv2
import numpy as np

# Load video
video_path = "test2.mp4"
output_video_path = "dynamic_motion_heatmap.mp4"

cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'avc1')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Read first frame
ret, frame1 = cap.read()
if not ret:
    print("Error: Couldn't read video file")
    cap.release()
    exit()

prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Initialize rolling heatmap (short memory to make it dynamic)
rolling_heatmap = np.zeros_like(prev_gray, dtype=np.float32)

while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Update rolling heatmap (exponential decay to prevent old data from dominating)
    rolling_heatmap = (rolling_heatmap * 0.8) + magnitude  # Adjust 0.8 to control memory length

    # Normalize heatmap for visibility
    heatmap_norm = cv2.normalize(rolling_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_color = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay heatmap on the frame
    overlay = cv2.addWeighted(frame2, 0.6, heatmap_color, 0.4, 0)
    out.write(overlay)

    # Update previous frame
    prev_gray = gray

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Dynamic Motion Heatmap saved as {output_video_path}")
