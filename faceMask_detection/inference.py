import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO(
    r"C:\Users\train5\weights\best.pt"
)

# FIX: Correct class-id → name mapping (NEW Ultralytics)
# need this because the ids got swapped that why
model.model.names = {
    0: "without_mask",
    1: "with_mask",
    2: "mask_weared_incorrect"
}

# Input & output video
input_video = r"C:\Users\inputs\newmy.mp4"
output_video = r"C:\Users\faceMask_detection\outputs\NewResult.mp4"

cap = cv2.VideoCapture(input_video)

# Get video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)

    annotated_frame = results[0].plot()
    out.write(annotated_frame)

cap.release()
out.release()

print("Inference complete. Output saved as:", output_video)
