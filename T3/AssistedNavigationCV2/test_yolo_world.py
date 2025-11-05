from ultralytics import YOLO
import cv2

# Load the YOLO-World segmentation model
model = YOLO("yolo11n.pt")  # or "yolo11-worldv2-seg.pt"

# Describe what you want to detect
description = ["road", "car", "person"]

# Input video path
input_video = "input.mp4"

# Output video path
output_video = "output_segmented.mp4"

# Open video file
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Set up output video writer
out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Process video frame-by-frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO-World segmentation
    results = model.predict(
        source=frame,
        task="segment",
        text=description,
        verbose=False
    )

    # Get annotated frame
    annotated = results[0].plot()

    # Show the frame
    cv2.imshow("YOLO-World Segmentation", annotated)

    # Write the frame to output file
    out.write(annotated)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Done! Segmented video saved to:", output_video)

