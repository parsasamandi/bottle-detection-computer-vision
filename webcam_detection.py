"""
Live webcam bottle detection using YOLO.
Opens your laptop camera and draws bounding boxes around bottles in real time.
Press Q to quit.
"""

# Import OpenCV for capturing webcam frames and displaying the video window
import cv2

# Import Path from pathlib for file path handling
from pathlib import Path

# Import YOLO from ultralytics for object detection
from ultralytics import YOLO

# ── 1. Load model ─────────────────────────────────────────────────────────────

# Load the pretrained YOLO11 nano model (fast enough for real-time webcam use)
model = YOLO("yolo11n.pt")

# ── 2. Find bottle class ID dynamically ───────────────────────────────────────

def find_class_id(model_object, target_class_name):
    """Return the numeric class ID for a class name in a YOLO model."""
    names = model_object.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, class_name in items:
        if str(class_name).strip().lower() == target_class_name.strip().lower():
            return int(class_id)
    raise ValueError(f"Class '{target_class_name}' not found in model.")

# Get the numeric ID for the "bottle" class
BOTTLE_CLASS_ID = find_class_id(model, "bottle")
print(f"Bottle class ID: {BOTTLE_CLASS_ID}")

# ── 3. Open webcam ────────────────────────────────────────────────────────────

# Open the default laptop camera (index 0 = built-in webcam)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Make sure it is not used by another app.")

print("Webcam opened. Press Q to quit.")

# ── 4. Real-time detection loop ───────────────────────────────────────────────

# Loop continuously until the user presses Q
while True:
    # Capture one frame from the webcam
    # ret = True if frame was captured successfully
    ret, frame = cap.read()

    # If frame capture failed, skip this iteration
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO detection on the current frame
    results = model.predict(
        # Pass the raw webcam frame (numpy array) directly to YOLO
        source=frame,
        # Only detect bottles
        classes=[BOTTLE_CLASS_ID],
        # Minimum confidence threshold
        conf=0.25,
        # Standard YOLO inference size
        imgsz=640,
        # Suppress per-frame console output
        verbose=False,
    )

    # Get the annotated frame with bounding boxes drawn by YOLO
    # [0] because predict returns a list; plot() returns a BGR numpy array
    annotated_frame = results[0].plot()

    # Count detected bottles in this frame
    boxes = results[0].boxes
    bottle_count = 0 if boxes is None else len(boxes)

    # Overlay the bottle count as text on the top-left of the frame
    cv2.putText(
        img=annotated_frame,
        text=f"Bottles detected: {bottle_count}",
        # Position: 10 pixels from left, 30 pixels from top
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1.0,
        # Green color in BGR format
        color=(0, 255, 0),
        thickness=2,
    )

    # Show the annotated frame in a window titled "Bottle Detection"
    cv2.imshow("Bottle Detection — Press Q to quit", annotated_frame)

    # Wait 1ms for a key press; if Q is pressed, break the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Q pressed — stopping.")
        break

# ── 5. Cleanup ────────────────────────────────────────────────────────────────

# Release the webcam so other apps can use it
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print("Webcam released. Done.")
