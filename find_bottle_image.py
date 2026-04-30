"""Scan COCO128 images to find ones with bottle detections and save annotated versions."""

# Import zipfile to handle compressed dataset archives
import zipfile

# Import Path from pathlib for cross-platform file path handling
from pathlib import Path

# Import YOLO from ultralytics for object detection
from ultralytics import YOLO

# Import download utility from ultralytics to fetch datasets
from ultralytics.utils.downloads import download

# Load the pretrained YOLO11 nano model with bottle detection capability
model = YOLO("yolo11n.pt")

# Define the path where the COCO128 dataset zip file will be stored
zip_path = Path("coco128.zip")

# Check if the dataset zip file doesn't already exist locally
if not zip_path.exists():
    # If missing, download the COCO128 mini dataset from the official source
    download("https://ultralytics.com/assets/coco128.zip", dir=".")

# Define the directory path where extracted COCO128 images will be located
img_dir = Path("coco128/images/train2017")

# Check if the extracted image directory doesn't already exist
if not img_dir.exists():
    # Open the downloaded zip file in read mode
    with zipfile.ZipFile(zip_path) as z:
        # Extract all contents of the zip file to the current directory
        z.extractall(".")
    # Print confirmation that extraction completed successfully
    print("Extracted coco128")

# Create a directory to store annotated results
annotated_dir = Path("annotated_detections")
annotated_dir.mkdir(exist_ok=True)

# Initialize an empty list to store images that contain bottle detections
found = []

# Loop through all JPG files in the image directory sorted by filename
for img_path in sorted(img_dir.glob("*.jpg")):
    # Run YOLO detection on the current image and get the first result
    r = model.predict(source=str(img_path), verbose=False)[0]
    # Filter detection boxes to only keep those where class ID is 39 (bottle class)
    bottles = [b for b in (r.boxes or []) if int(b.cls[0]) == 39]
    # Check if any bottle detections were found in this image
    if bottles:
        # Find the highest confidence score among all bottle detections in the image
        best_conf = max(float(b.conf[0]) for b in bottles)
        # Append a tuple of (best_confidence, bottle_count, image_path) to the found list
        found.append((best_conf, len(bottles), img_path))
        
        # Save annotated version with bounding boxes drawn
        annotated_image = r.plot(pil=True)
        output_filename = f"{img_path.stem}_annotated_{best_conf:.2f}.jpg"
        output_path = annotated_dir / output_filename
        annotated_image.save(str(output_path))

# Sort the found images in descending order by confidence (best detections first)
found.sort(reverse=True)

# Print a header showing how many total images contained bottle detections
print(f"\nFound bottles in {len(found)} images:")

# Loop through the top 10 images with the best bottle detections
for conf, count, p in found[:10]:
    # For each image, print its filename, bottle count, and best confidence percentage
    print(f"  {p.name}: {count} bottle(s), best conf={conf:.2%}")

# Print where annotated versions were saved
print(f"\nAnnotated images saved to: {annotated_dir}")

# Check if at least one image with bottle detections was found
if found:
    # Print the full path of the single best image (highest confidence)
    print(f"Best image for test: {found[0][2]}")
