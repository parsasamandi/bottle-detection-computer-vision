"""
Local test script for bottle detection using YOLO.
Runs the same logic as the Colab script but without Colab-specific imports.
Output is saved to output_annotated.jpg instead of displayed inline.
"""

from pathlib import Path

from PIL import Image
from ultralytics import YOLO

# ── 1. Load model ─────────────────────────────────────────────────────────────
MODEL_NAME = "yolo11n.pt"   # YOLO11n: latest stable pretrained nano checkpoint
                            # (used locally; Colab version targets yolo26n.pt)
model = YOLO(MODEL_NAME)
print(f"Loaded model: {MODEL_NAME}")

# ── 2. Find bottle class ID dynamically ───────────────────────────────────────
def find_class_id(model_object, target_class_name):
    names = model_object.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    for class_id, class_name in items:
        if str(class_name).strip().lower() == target_class_name.strip().lower():
            return int(class_id)
    available = ", ".join(str(n) for n in (names.values() if isinstance(names, dict) else names))
    raise ValueError(
        f"Class '{target_class_name}' not found. Available: {available}"
    )

BOTTLE_CLASS_ID = find_class_id(model, "bottle")
print(f"Bottle class ID in this model: {BOTTLE_CLASS_ID}")

# ── 3. Run detection ──────────────────────────────────────────────────────────
IMAGE_PATH = Path("coco128/images/train2017/000000000142.jpg")  # COCO128: 2 bottles @ 87% conf

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

# Validate it is a real image
with Image.open(IMAGE_PATH) as img:
    img.verify()

results = model.predict(
    source=str(IMAGE_PATH),
    classes=[BOTTLE_CLASS_ID],
    conf=0.25,
    imgsz=640,
    verbose=False,
)

result = results[0]
boxes  = result.boxes
bottle_count = 0 if boxes is None else len(boxes)

# ── 4. Print results ──────────────────────────────────────────────────────────
print(f"\n=== Result for {IMAGE_PATH.name} ===")
if bottle_count == 0:
    print("No bottles detected at conf >= 0.25.")
elif bottle_count == 1:
    print("Detected 1 bottle.")
else:
    print(f"Detected {bottle_count} bottles.")

if bottle_count > 0:
    for i, box in enumerate(boxes, start=1):
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        print(
            f"  Bottle {i}: confidence={conf:.2%}"
            f"  box=(left={x1:.1f}, top={y1:.1f}, right={x2:.1f}, bottom={y2:.1f})"
        )

# ── 5. Save annotated image ───────────────────────────────────────────────────
output_path = Path("output_annotated.jpg")
annotated = result.plot(pil=True)
annotated.save(str(output_path))
print(f"\nAnnotated image saved to: {output_path}")
