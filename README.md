# Car Part Segmentation (YOLOv11)

Academic project for **instance segmentation of car parts** using **Ultralytics YOLO11** in segmentation mode (`yolo11n-seg`). The workflow covers training on a car-parts segmentation dataset and running predictions through a small desktop GUI.

---

## What this repository contains

| Item | Description |
|------|-------------|
| [`carpartsegmentation.ipynb`](carpartsegmentation.ipynb) | Jupyter notebook: installs `ultralytics`, loads `yolo11n-seg.pt`, and trains for **100 epochs** at **640×640** on a dataset described by `carparts-seg.yaml`. |
| [`testgui.py`](testgui.py) | **Tkinter** app: load an image (JPG/PNG), run the trained model, and visualize **masks and boxes** on a canvas. |
| [`metrics/results.csv`](metrics/results.csv) | Per-epoch training log (losses, box/mask precision, recall, mAP50, mAP50-95, learning rates). |

---

## Model and stack

- **Architecture:** YOLO11 **nano** segmentation (`yolo11n-seg`).
- **Framework:** [Ultralytics](https://docs.ultralytics.com/) (YOLOv8-style API, YOLO11 weights).
- **Task:** Segmentation (pixel masks + bounding boxes per part class).
- **GUI:** Python `tkinter`, **Pillow**, **OpenCV**, Ultralytics `YOLO.predict()`.

---

## Repository layout

```
CarPartSegementation_YOLOv11/
├── carpartsegmentation.ipynb   # Training entry point (notebook)
├── testgui.py                  # Local inference GUI
├── metrics/
│   └── results.csv             # Exported training metrics (100 epochs)
└── README.md
```

---

## Prerequisites

- **Python 3.10+** (notebook metadata references Python 3.11).
- **PyTorch** with a CUDA build if you train or infer on GPU (optional but recommended for training).
- **Weights file for the GUI:** `testgui.py` loads **`carpartseg.pt`**. Place that file in the **same directory** as `testgui.py` (or update the path in code). This file is **not** shipped in this repo; it is produced after training (e.g. `runs/segment/train/weights/best.pt` renamed or copied).

---

## Installation

```bash
pip install ultralytics opencv-python pillow
```

On Windows, `tkinter` is usually included with the standard Python installer. On some Linux distributions you may need `python3-tk`.

---

## Training

1. Obtain a YOLO-format **segmentation** dataset and a matching **`carparts-seg.yaml`** (paths, class names, train/val splits). The notebook expects that YAML in the working directory (or adjust the path in the notebook).
2. Open [`carpartsegmentation.ipynb`](carpartsegmentation.ipynb) and run the cell. It will:
   - `pip install ultralytics`
   - Load `yolo11n-seg.pt`
   - Run `model.train(data="carparts-seg.yaml", epochs=100, imgsz=640)`
3. After training, copy or rename the best weights to `carpartseg.pt` for the GUI, or point `testgui.py` to your `best.pt`.

Training artifacts (plots, weights, logs) are typically under `runs/segment/` in the directory where you executed the notebook.

---

## Running the GUI

From the project root (with `carpartseg.pt` available next to the script unless you changed the path):

```bash
python testgui.py
```

1. Click **Load Image** and choose a `.jpg`, `.jpeg`, or `.png`.
2. Click **Predict** — inference uses `imgsz=640` and `conf=0.25`.
3. **Reset** clears the canvas and disables predict until a new image is loaded.

---

## Metrics summary (from `metrics/results.csv`)

The logged run completes **100 epochs**. On the **final recorded epoch**, validation metrics are approximately:

| Metric | Boxes (B) | Masks (M) |
|--------|-----------|-----------|
| Precision | ~0.60 | ~0.61 |
| Recall | ~0.74 | ~0.75 |
| mAP@0.5 | ~0.66 | ~0.67 |
| mAP@0.5:0.95 | ~0.57 | ~0.55 |

Exact numbers may differ slightly if you retrain with another split, seed, or hardware. Use `metrics/results.csv` or Ultralytics’ own result plots for full curves and per-class breakdowns when available.

---

## Notes for reproducibility

- The notebook was executed in an environment with **CUDA 12.4**–compatible PyTorch (per install logs in the notebook output). Your versions may differ.
- If you do not have `carparts-seg.yaml` or the dataset, training will fail until those are configured.
- For production or deployment, consider exporting with Ultralytics (e.g. ONNX, TensorRT) instead of shipping the full PyTorch GUI stack.

---

## License and credits

- **Ultralytics YOLO** is subject to [Ultralytics’ licensing](https://github.com/ultralytics/ultralytics/blob/main/LICENSE); check their terms for commercial use.
- Dataset licensing depends on the source of your **carparts** segmentation data; keep that license alongside your project if you redistribute data or derivatives.

This repository is suitable as a **course / lab project** demonstrating segmentation training with YOLO11 and a minimal local demo UI.
