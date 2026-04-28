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

# Import subprocess to run system commands like pip install in the background
import subprocess

# Import sys to access the current Python executable path for pip installation
import sys

# Run pip (Python's package installer) to install or upgrade ultralytics package
# sys.executable ensures we install to the same Python version running this notebook
subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "ultralytics"])

# Print confirmation that the installation completed without errors
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

# Import Path from pathlib for cross-platform file path handling in Colab environment
from pathlib import Path

# Import files from google.colab to open the file upload dialog in Colab
from google.colab import files

# Import display from IPython to show images directly in the notebook
from IPython.display import display

# Import Image from PIL for image validation before YOLO processing
from PIL import Image

# Import YOLO from ultralytics for bottle detection inference
from ultralytics import YOLO

# Print confirmation that all required libraries imported successfully
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

# Define the model checkpoint filename for YOLO26 nano model
# yolo26n = YOLO26 nano (smallest and fastest)
# .pt = PyTorch format checkpoint file that Ultralytics auto-downloads
MODEL_NAME = "yolo26n.pt"

# Load the pretrained YOLO model; Ultralytics auto-downloads from official servers if missing
model = YOLO(MODEL_NAME)

# Print confirmation message showing which model was loaded
print(f"Loaded model: {MODEL_NAME}")

# Print a header indicating model class names are being displayed
print("Model classes loaded. Example class names:")

# Print the dictionary or list of all 80 COCO classes the model can detect (including bottle)
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


# Define a function to dynamically look up the numeric class ID for a class name
def find_class_id(model_object, target_class_name):
    # Docstring explaining the function's purpose and return value
    """Return the numeric class ID for a class name in a YOLO model."""
    
    # Extract the class names dictionary or list from the YOLO model object
    names = model_object.names

    # Check if names is a dictionary (newer Ultralytics versions) or a list (older versions)
    if isinstance(names, dict):
        # If dict, use .items() to get (class_id, class_name) pairs
        class_items = names.items()
    else:
        # If list, use enumerate() to get (class_id, class_name) pairs with indices
        class_items = enumerate(names)

    # Loop through each (class_id, class_name) pair in the model
    for class_id, class_name in class_items:
        # Convert class name to string, strip whitespace, and convert to lowercase for comparison
        if str(class_name).strip().lower() == target_class_name.strip().lower():
            # If this class name matches the target, return its numeric ID as integer
            return int(class_id)

    # If target class not found, build a comma-separated list of all available class names
    available_classes = ", ".join(str(name) for name in names.values()) if isinstance(names, dict) else ", ".join(names)
    
    # Raise an error with the target class name and all available options to help debugging
    raise ValueError(
        f"Could not find class '{target_class_name}' in this model. "
        f"Available classes are: {available_classes}"
    )


# Call find_class_id to dynamically get the numeric ID for class name "bottle"
BOTTLE_CLASS_ID = find_class_id(model, "bottle")

# Print the numeric class ID that was found for reference
print(f"The model's class ID for 'bottle' is: {BOTTLE_CLASS_ID}")

# %%
# -----------------------------------------------------------------------------
# BLOCK 5: Upload image files from your computer
# -----------------------------------------------------------------------------
# Beginner explanation:
# This block opens a file picker. Choose one or more images, such as JPG or PNG.
# Colab stores uploaded files in the current notebook session. The session is
# temporary, so uploaded files disappear when the Colab runtime is reset.

# Trigger Colab's file upload dialog and get a dictionary of uploaded files
# Keys are filenames, values are file contents as bytes
uploaded_files = files.upload()

# Check if the user actually chose any files from the upload dialog
if not uploaded_files:
    # If no files were uploaded, raise an error to prompt the user to retry
    raise ValueError("No image was uploaded. Please run this cell again and choose an image.")

# Initialize an empty list to store the Path objects for all uploaded images
image_paths = []

# Loop through each uploaded file's (filename, file_bytes) pair
for filename, file_bytes in uploaded_files.items():
    # Convert the filename string into a Path object for cross-platform compatibility
    image_path = Path(filename)

    # Write the uploaded file bytes to disk in the current Colab working directory
    # This makes the behavior explicit and reliable even with spaces in filenames
    image_path.write_bytes(file_bytes)
    
    # Add the Path object to our list of image paths for later processing
    image_paths.append(image_path)

# Print a header indicating the list of uploaded files
print("Uploaded image files:")

# Loop through all uploaded image paths and print each one
for image_path in image_paths:
    # Print the filename with a dash prefix for readability
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

# Define the confidence threshold below which detections are discarded
# 0.25 = 25% confidence; this is a balanced default (not too strict, not too loose)
CONFIDENCE_THRESHOLD = 0.25

# Define the standard image size for YOLO inference (in pixels)
# 640x640 is the standard YOLO inference size; balances speed and accuracy
IMAGE_SIZE = 640


# Define a function to detect bottles in a single image and display the results
def detect_bottles_in_image(image_path, confidence_threshold=CONFIDENCE_THRESHOLD, image_size=IMAGE_SIZE):
    # Docstring explaining the function's purpose
    """Detect bottles in one uploaded image and display the annotated output."""
    
    # Convert the image path to a Path object for consistent file handling
    image_path = Path(image_path)

    # Check that the image file actually exists on disk before trying to process it
    if not image_path.exists():
        # Raise an error with the missing path to help with debugging
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Validate that PIL can open and parse the image before sending to YOLO
    # This provides a clearer error if someone uploads a corrupted or non-image file
    with Image.open(image_path) as image:
        # Call verify() which raises an exception if the image format is invalid
        image.verify()

    # Run YOLO detection on the image with bottle-specific filtering and parameters
    results = model.predict(
        # Use the image file path as the detection source
        source=str(image_path),
        # Only detect bottles; ignore all other 79 COCO object classes
        classes=[BOTTLE_CLASS_ID],
        # Only keep predictions with at least this confidence level (default 0.25 = 25%)
        conf=confidence_threshold,
        # Resize image to this standard YOLO inference size (640x640 pixels)
        imgsz=image_size,
        # Suppress verbose console output about the detection process
        verbose=False,
    )

    # Extract the first (and only) result object from the results list
    result = results[0]
    
    # Get the detected bounding boxes from the result object
    boxes = result.boxes
    
    # Count the number of bottles by checking if boxes exist; default to 0 if None
    bottle_count = 0 if boxes is None else len(boxes)

    # Print a formatted header showing which image is being reported on
    print(f"\n=== Result for {image_path.name} ===")

    # Check if no bottles were detected
    if bottle_count == 0:
        # Print a message indicating no bottles found at the current threshold
        print("No bottles were detected in this image at the current confidence threshold.")
    # Check if exactly one bottle was detected
    elif bottle_count == 1:
        # Print singular form for one bottle
        print("Detected 1 bottle.")
    # If multiple bottles were detected
    else:
        # Print the total count of detected bottles
        print(f"Detected {bottle_count} bottles.")

    # Check if at least one bottle was detected to print detailed results
    if bottle_count > 0:
        # Loop through each detected box with an index starting from 1
        for index, box in enumerate(boxes, start=1):
            # Extract and convert the confidence score to a float
            confidence = float(box.conf[0])
            
            # Extract the four bounding box coordinates (left, top, right, bottom) as floats
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
            
            # Print bottle number, confidence percentage, and bounding box coordinates
            print(
                f"Bottle {index}: "
                f"confidence={confidence:.2%}, "
                f"box=(left={x1:.1f}, top={y1:.1f}, right={x2:.1f}, bottom={y2:.1f})"
            )

    # Use Ultralytics' plot method to draw bounding boxes on the image
    # pil=True returns a PIL Image object which Colab can display directly
    # This avoids RGB/BGR color-order mistakes that can occur with OpenCV arrays
    annotated_image = result.plot(pil=True)
    
    # Display the annotated image directly in the Colab notebook
    display(annotated_image)

    # Return the full result object for potential future analysis or processing
    return result


# Initialize an empty list to store detection results from all uploaded images
all_results = []

# Loop through each uploaded image path and process it
for image_path in image_paths:
    # Call the detection function to analyze this single image
    result = detect_bottles_in_image(image_path)
    
    # Store the result object in the list for potential later reference or batch processing
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
