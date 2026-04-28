# pyright: reportMissingImports=false, reportMissingModuleSource=false
# %% [markdown]
# # Bottle Detection with Ultralytics YOLO26 in Google Colab
#
# This script is designed for Google Colab. Run each section from top to bottom.
#
# Goal:
# 1. Install the latest Ultralytics package that includes YOLO26 support.
# 2. Load the small pretrained YOLO26 nano model: `yolo26n.pt`.
# 3. Upload one or more images.
# 4. Draw boxes only around objects that the model classifies as `bottle`.
#
# Beginner note:
# YOLO is an object detection model. Instead of only saying "this image has a bottle",
# it predicts where the bottle is by returning a rectangle called a bounding box.

# %%
# -----------------------------------------------------------------------------
# BLOCK 1: Install the latest Ultralytics package
# -----------------------------------------------------------------------------
# Beginner explanation:
# Colab starts with a temporary computer in the cloud. That computer may not have
# the newest YOLO tools installed. This block uses Python itself to install or
# upgrade the `ultralytics` package. This is the official package used to load
# and run modern Ultralytics YOLO models, including YOLO26.

import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics"])

print("Ultralytics installed or updated successfully.")

# %%
# -----------------------------------------------------------------------------
# BLOCK 2: Import the tools we need
# -----------------------------------------------------------------------------
# Beginner explanation:
# Imports are like unpacking tools before starting work.
# - YOLO loads the pretrained model.
# - files opens Colab's upload button.
# - PIL and IPython display help show images inside the notebook.
# - pathlib helps us work safely with uploaded file paths.

from pathlib import Path

from google.colab import files
from IPython.display import display
from PIL import Image
from ultralytics import YOLO

print("Libraries imported successfully.")

# %%
# -----------------------------------------------------------------------------
# BLOCK 3: Load the pretrained YOLO26 model
# -----------------------------------------------------------------------------
# Beginner explanation:
# `yolo26n.pt` is a pretrained model file.
# - `yolo26` means the YOLO26 model family.
# - `n` means nano, which is the smallest/fastest version.
# - `.pt` means it is a PyTorch model checkpoint file.
#
# This model was trained on a large dataset with common object names, including
# `bottle`, so we can use it without training our own custom model first.

MODEL_NAME = "yolo26n.pt"
model = YOLO(MODEL_NAME)

print(f"Loaded model: {MODEL_NAME}")
print("Model classes loaded. Example class names:")
print(model.names)

# %%
# -----------------------------------------------------------------------------
# BLOCK 4: Find the model's internal class ID for "bottle"
# -----------------------------------------------------------------------------
# Beginner explanation:
# AI models usually store class labels as numbers internally.
# For the COCO dataset, `bottle` is commonly class ID 39, but we do not hardcode
# that value here. Instead, we ask the model for its class list and search for
# the name `bottle`. This is safer if class ordering ever changes.


def find_class_id(model_object, target_class_name):
    """Return the numeric class ID for a class name in a YOLO model."""
    names = model_object.names

    if isinstance(names, dict):
        class_items = names.items()
    else:
        class_items = enumerate(names)

    for class_id, class_name in class_items:
        if str(class_name).strip().lower() == target_class_name.strip().lower():
            return int(class_id)

    available_classes = ", ".join(str(name) for name in names.values()) if isinstance(names, dict) else ", ".join(names)
    raise ValueError(
        f"Could not find class '{target_class_name}' in this model. "
        f"Available classes are: {available_classes}"
    )


BOTTLE_CLASS_ID = find_class_id(model, "bottle")
print(f"The model's class ID for 'bottle' is: {BOTTLE_CLASS_ID}")

# %%
# -----------------------------------------------------------------------------
# BLOCK 5: Upload image files from your computer
# -----------------------------------------------------------------------------
# Beginner explanation:
# This block opens a file picker. Choose one or more images, such as JPG or PNG.
# Colab stores uploaded files in the current notebook session. The session is
# temporary, so uploaded files disappear when the Colab runtime is reset.

uploaded_files = files.upload()

if not uploaded_files:
    raise ValueError("No image was uploaded. Please run this cell again and choose an image.")

image_paths = []
for filename, file_bytes in uploaded_files.items():
    image_path = Path(filename)

    # Colab usually writes the uploaded file automatically. This write step makes
    # the behavior explicit and reliable even if the filename contains spaces.
    image_path.write_bytes(file_bytes)
    image_paths.append(image_path)

print("Uploaded image files:")
for image_path in image_paths:
    print(f"- {image_path}")

# %%
# -----------------------------------------------------------------------------
# BLOCK 6: Run bottle detection and display results
# -----------------------------------------------------------------------------
# Beginner explanation:
# This is the main AI step.
#
# Important parameters:
# - `source`: the uploaded image path.
# - `classes=[BOTTLE_CLASS_ID]`: only detect bottles, ignoring other objects.
# - `conf=0.25`: only keep predictions at least 25% confident.
# - `imgsz=640`: resize image processing to a standard YOLO-friendly size.
#
# The model returns boxes, confidence scores, and labels. We then draw those boxes
# on the image and display the annotated result.

CONFIDENCE_THRESHOLD = 0.25
IMAGE_SIZE = 640


def detect_bottles_in_image(image_path, confidence_threshold=CONFIDENCE_THRESHOLD, image_size=IMAGE_SIZE):
    """Detect bottles in one uploaded image and display the annotated output."""
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Validate that PIL can open the image before sending it to YOLO.
    # This gives a clearer error if someone uploads a non-image file.
    with Image.open(image_path) as image:
        image.verify()

    results = model.predict(
        source=str(image_path),
        classes=[BOTTLE_CLASS_ID],
        conf=confidence_threshold,
        imgsz=image_size,
        verbose=False,
    )

    result = results[0]
    boxes = result.boxes
    bottle_count = 0 if boxes is None else len(boxes)

    print(f"\n=== Result for {image_path.name} ===")

    if bottle_count == 0:
        print("No bottles were detected in this image at the current confidence threshold.")
    elif bottle_count == 1:
        print("Detected 1 bottle.")
    else:
        print(f"Detected {bottle_count} bottles.")

    if bottle_count > 0:
        for index, box in enumerate(boxes, start=1):
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            print(
                f"Bottle {index}: "
                f"confidence={confidence:.2%}, "
                f"box=(left={x1:.1f}, top={y1:.1f}, right={x2:.1f}, bottom={y2:.1f})"
            )

    # `plot(pil=True)` returns a PIL image, which Colab can display directly.
    # This avoids RGB/BGR color-order mistakes that can happen with OpenCV arrays.
    annotated_image = result.plot(pil=True)
    display(annotated_image)

    return result


all_results = []
for image_path in image_paths:
    result = detect_bottles_in_image(image_path)
    all_results.append(result)

# %% [markdown]
# ## Logic Summary
#
# | Choice / Parameter | Value Used | Why the AI Chose It | Beginner Translation |
# |---|---:|---|---|
# | Package | `ultralytics` | This is the official Python package for loading and running Ultralytics YOLO models. | Install the toolbox before using the model. |
# | Model checkpoint | `yolo26n.pt` | Pretrained YOLO checkpoints are loaded with their `.pt` file name. The `n` version is small and fast. | Use the fast beginner-friendly model that already knows common objects. |
# | Target class | `bottle` | The task is bottle detection only, so the script filters out all other object classes. | Only draw boxes around bottles, not people, cups, or chairs. |
# | Class lookup | Dynamic lookup from `model.names` | Safer than assuming the bottle class ID is always the same. | Ask the model what number means “bottle.” |
# | Confidence threshold | `0.25` | A practical starting point that avoids many weak guesses without being too strict. | Keep detections the model is at least 25% confident about. |
# | Image size | `640` | A common YOLO inference size that balances speed and accuracy. | Resize processing to a standard size the model handles well. |
# | Display method | `result.plot(pil=True)` | Lets Ultralytics draw boxes and returns an image Colab can display directly. | Let the model draw the boxes, then show the picture. |
# | Upload method | `files.upload()` | Easiest Colab-native way for beginners to provide images. | Click a button and choose an image from your computer. |
#
# ## How to improve results later
#
# - If the model misses bottles, lower `CONFIDENCE_THRESHOLD` to `0.15`.
# - If the model draws boxes around things that are not bottles, raise `CONFIDENCE_THRESHOLD` to `0.40`.
# - If you need better accuracy and have more compute, try a larger checkpoint such as `yolo26s.pt` or `yolo26m.pt`.
# - If you need to detect a special bottle type, such as medical bottles or factory bottles, collect labeled images and fine-tune the model.
