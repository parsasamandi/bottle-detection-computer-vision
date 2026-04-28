# Bottle Detection with YOLO26 / YOLO11

A beginner-friendly computer vision project for detecting bottles in images using Ultralytics YOLO models. Includes both a **Google Colab notebook** (easy, cloud-based) and a **local Python script** (for development).

---

## Quick Start

### Option 1: Google Colab (Recommended for Beginners)

1. Open [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Copy the contents of **`bottle_detection_yolo26_colab.py`** into a code cell
4. Run all cells (`Runtime → Run all`)
5. When prompted, upload any JPG or PNG image containing bottles
6. The model will draw bounding boxes around detected bottles and show results

**No setup required — everything runs in the cloud.**

---

### Option 2: Local Setup (For Testing / Development)

#### Prerequisites
- Python 3.12+ (or 3.11)
- macOS, Linux, or Windows

#### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/parsasamandi/bottle-detection-computer-vision.git
   cd bottle-detection-computer-vision
   ```

2. **Create a Python virtual environment:**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install ultralytics pillow
   ```

#### Run the Test

```bash
python test_local.py
```

**What happens:**
- Loads the YOLO11 nano model (`yolo11n.pt`)
- Detects bottles in `test_bottle_image.jpg`
- Prints detection results (count, confidence, box coordinates)
- Saves annotated image to `output_annotated.jpg`

**Expected output:**
```
Loaded model: yolo11n.pt
Bottle class ID in this model: 39

=== Result for test_bottle_image.jpg ===
Detected 2 bottles.
  Bottle 1: confidence=87.28%  box=(left=258.3, top=85.2, right=419.0, bottom=335.7)
  Bottle 2: confidence=81.73%  box=(left=84.3, top=0.1, right=268.0, bottom=373.5)

Annotated image saved to: output_annotated.jpg
```

---

## Project Files

| File | Purpose |
|---|---|
| `bottle_detection_yolo26_colab.py` | **Main Colab script** — copy this into Google Colab to run |
| `test_local.py` | Local test script — run locally to verify setup |
| `find_bottle_image.py` | Dataset scanner utility — finds COCO128 images with best bottle detections |
| `test_bottle_image.jpg` | Test image (42 KB) — contains 2 detectable bottles |
| `output_annotated.jpg` | Sample output — shows what detection looks like |
| `README.md` | This file |
| `.gitignore` | Git configuration (excludes large files) |

---

## How It Works

### The Model
- **YOLO11n** (locally) or **YOLO26n** (in Colab) — pretrained on COCO dataset with 80 object classes
- **yolo11n.pt** / **yolo26n.pt** — automatically downloaded on first run (~30 MB)
- Detects objects via bounding boxes with confidence scores

### The Detection Pipeline
```
1. Load pretrained model
2. Find bottle class ID dynamically (COCO class 39)
3. Run inference on image with:
   - classes filter: only detect bottles
   - confidence threshold: 0.25 (25% confidence minimum)
   - image size: 640×640 (standard YOLO size)
4. Draw boxes on image
5. Display results
```

### Behind the Scenes
- **No training required** — the model already knows what bottles look like
- **Beginner-friendly** — all code is commented and explained
- **CPU-friendly** — nano models run fast even on laptops

---

## Testing with Your Own Images

### In Google Colab
1. Run the script normally
2. When Block 5 executes, a file picker appears
3. Select any JPG/PNG from your computer
4. Results appear automatically

### Locally
Edit `test_local.py` line 25:
```python
IMAGE_PATH = Path("your_image.jpg")  # Change this to your image
```

Then run:
```bash
python test_local.py
```

---

## Customization

### Adjust Confidence Threshold
Lower confidence = more detections (but more false positives):
```python
CONFIDENCE_THRESHOLD = 0.15  # More lenient (find smaller/distant bottles)
# or
CONFIDENCE_THRESHOLD = 0.50  # More strict (only confident detections)
```

### Use a Larger Model
For better accuracy (slower):
```python
MODEL_NAME = "yolo11m.pt"  # medium model
# or
MODEL_NAME = "yolo11l.pt"  # large model
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'ultralytics'` | Run `pip install ultralytics pillow` |
| Model takes forever to download | First run downloads ~30 MB; patience required or use a faster network |
| No bottles detected | Lower `CONFIDENCE_THRESHOLD` (e.g., `0.15`) or try a different image |
| `FileNotFoundError` for image | Use an absolute path or ensure the image is in the same directory |

---

## Learning Resources

- **Ultralytics YOLO docs:** https://docs.ultralytics.com/
- **COCO dataset classes:** https://cocodataset.org/
- **YOLO11 model card:** https://github.com/ultralytics/ultralytics

---

## Logic Summary Table

| Concept | Value | Why |
|---|---|---|
| Model | YOLO11n (local) / YOLO26n (Colab) | Fast, lightweight, pretrained on 80 object classes |
| Class ID for bottle | 39 | Standard COCO dataset class ID |
| Confidence threshold | 0.25 | Balanced — catches most bottles, low false positives |
| Image size | 640 | Standard YOLO inference resolution |
| Class filtering | `classes=[39]` | Only detect bottles, ignore cups, wine glasses, etc. |

---

## Author

Created as a beginner-friendly introduction to computer vision and object detection using Ultralytics YOLO.

**Questions?** Open an issue on GitHub or review the inline comments in `bottle_detection_yolo26_colab.py`.
