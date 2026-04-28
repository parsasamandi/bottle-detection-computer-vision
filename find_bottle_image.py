"""Scan COCO128 images to find ones with bottle detections."""
import zipfile
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.downloads import download

model = YOLO("yolo11n.pt")

# Download COCO128 mini dataset
zip_path = Path("coco128.zip")
if not zip_path.exists():
    download("https://ultralytics.com/assets/coco128.zip", dir=".")

img_dir = Path("coco128/images/train2017")
if not img_dir.exists():
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(".")
    print("Extracted coco128")

# Scan all images for bottle detections
found = []
for img_path in sorted(img_dir.glob("*.jpg")):
    r = model.predict(source=str(img_path), verbose=False)[0]
    bottles = [b for b in (r.boxes or []) if int(b.cls[0]) == 39]
    if bottles:
        best_conf = max(float(b.conf[0]) for b in bottles)
        found.append((best_conf, len(bottles), img_path))

found.sort(reverse=True)
print(f"\nFound bottles in {len(found)} images:")
for conf, count, p in found[:10]:
    print(f"  {p.name}: {count} bottle(s), best conf={conf:.2%}")

if found:
    print(f"\nBest image for test: {found[0][2]}")
