"""
Local test script for bottle detection using YOLO.
Runs the same logic as the Colab script but without Colab-specific imports.
Output is saved to output_annotated.jpg instead of displayed inline.
"""

# Import Path from pathlib for cross-platform file path handling
from pathlib import Path

# Import Image from PIL for image validation before processing
from PIL import Image

# Import YOLO from ultralytics for object detection
from ultralytics import YOLO

# ── 1. Load model ─────────────────────────────────────────────────────────────

# Define the model checkpoint name to load (YOLO11 nano for speed)
MODEL_NAME = "yolo11n.pt"   # YOLO11n: latest stable pretrained nano checkpoint
                            # (used locally; Colab version targets yolo26n.pt)

# Load the pretrained YOLO model; Ultralytics auto-downloads if not present
model = YOLO(MODEL_NAME)

# Print confirmation that the model loaded successfully
print(f"Loaded model: {MODEL_NAME}")

# ── 2. Find bottle class ID dynamically ───────────────────────────────────────

# Define a function to dynamically find the numeric class ID for a class name
def find_class_id(model_object, target_class_name):
    # Get the class name dictionary or list from the loaded model
    names = model_object.names
    # Convert to iterable items: dict.items() for dict, or enumerate() for list
    items = names.items() if isinstance(names, dict) else enumerate(names)
    # Loop through each class ID and name pair in the model
    for class_id, class_name in items:
        # Compare class name to target, case-insensitive with whitespace stripped
        if str(class_name).strip().lower() == target_class_name.strip().lower():
            # Return the numeric class ID as an integer
            return int(class_id)
    # If target class not found, build a comma-separated list of available classes
    available = ", ".join(str(n) for n in (names.values() if isinstance(names, dict) else names))
    # Raise an error with available classes to help user debug
    raise ValueError(
        f"Class '{target_class_name}' not found. Available: {available}"
    )

# Call the function to find the numeric class ID for "bottle"
BOTTLE_CLASS_ID = find_class_id(model, "bottle")

# Print the bottle class ID that was found
print(f"Bottle class ID in this model: {BOTTLE_CLASS_ID}")

# ── 3. Run detection ──────────────────────────────────────────────────────────

# Define the path to the test image file
IMAGE_PATH = Path("test_bottle_image.jpg")  # Test image: 2 bottles @ 87% & 82% conf

# Check that the test image file actually exists on disk
if not IMAGE_PATH.exists():
    # Raise an error with the missing file path to help debugging
    raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

# Validate that the file is a readable image by opening and checking it with PIL
with Image.open(IMAGE_PATH) as img:
    # Verify the image format is valid; raises exception if corrupted
    img.verify()

# Run YOLO detection on the image with bottle-only filtering
results = model.predict(
    # Use the test image path as the detection source
    source=str(IMAGE_PATH),
    # Only detect bottles; ignore all other 79 COCO classes
    classes=[BOTTLE_CLASS_ID],
    # Only keep predictions with at least 25% confidence
    conf=0.25,
    # Resize image to standard YOLO inference size for consistency
    imgsz=640,
    # Suppress verbose console output about the detection
    verbose=False,
)

# Extract the first (and only) result from the results list
result = results[0]

# Get the detected bounding boxes from the result object
boxes  = result.boxes

# Count bottles by checking if boxes exist and getting the length, default to 0
bottle_count = 0 if boxes is None else len(boxes)

# ── 4. Print results ──────────────────────────────────────────────────────────

# Print a formatted header showing which image was processed
print(f"\n=== Result for {IMAGE_PATH.name} ===")

# Check if no bottles were detected
if bottle_count == 0:
    # Print a message indicating no bottles were found
    print("No bottles detected at conf >= 0.25.")
# Check if exactly one bottle was detected
elif bottle_count == 1:
    # Print singular form for one bottle
    print("Detected 1 bottle.")
# If multiple bottles were detected
else:
    # Print the count of detected bottles
    print(f"Detected {bottle_count} bottles.")

# Check if any bottles were actually detected
if bottle_count > 0:
    # Loop through each detected bottle box with its index starting from 1
    for i, box in enumerate(boxes, start=1):
        # Extract the confidence score from the box and convert to float
        conf = float(box.conf[0])
        # Extract the four box coordinates (left, top, right, bottom) as floats
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        # Print bottle number, confidence, and bounding box coordinates
        print(
            f"  Bottle {i}: confidence={conf:.2%}"
            f"  box=(left={x1:.1f}, top={y1:.1f}, right={x2:.1f}, bottom={y2:.1f})"
        )

# ── 5. Save annotated image ───────────────────────────────────────────────────

# Define the output filename for the annotated visualization
output_path = Path("output_annotated.jpg")

# Use Ultralytics' plot method to draw boxes on the image; pil=True returns PIL image
annotated = result.plot(pil=True)

# Save the annotated image to disk as a JPEG file
annotated.save(str(output_path))

# Print confirmation that the annotated image was saved successfully
print(f"\nAnnotated image saved to: {output_path}")
