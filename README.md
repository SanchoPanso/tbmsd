# YOLOPatchFusion

**YOLOPatchFusion** is a Python library designed to facilitate object detection on large images using the YOLO model. The library splits images into smaller patches, performs detection on each patch, and merges the detected objects into a unified output using geometrical techniques from the Shapely library. This ensures accurate detection results even on high-resolution or oversized images.

---

## Plans

- [x] take trained detection model
- [x] val metrcics for wrapper 1024
- [ ] val metrcics for multiscale 512 1024 with simple iou merging strategy
- [ ] create calibration based on P(conf)
- [ ] create calibration based on P(conf, size)


### val metrcics for wrapper 1024 (maybe incorrect)

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.287
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.455
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.307
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50      | area= small | maxDets=1000 ] = 0.410
 Average Recall     (AR) @[ IoU=0.50      | area=medium | maxDets=1000 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50      | area= large | maxDets=1000 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=  1 ] = 0.078
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 10 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.409
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.505
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.455
 Average Precision  (AP) @[ IoU=0.50      | area= small | maxDets=1000 ] = 0.355
 Average Precision  (AP) @[ IoU=0.50      | area=medium | maxDets=1000 ] = 0.485
 Average Precision  (AP) @[ IoU=0.50      | area= large | maxDets=1000 ] = 0.430
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=  1 ] = 0.076
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 10 ] = 0.234
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.455

Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ]
plane               0.7668237361814695
ship                0.8084246473928823
storage tank        0.5996587622738646
baseball diamond    0.24448226294379583
tennis court        0.8534118555484886
basketball court    0.3992621005089049
ground track field  0.29496993307756963
harbor              0.5990599446176755
bridge              0.3105209165468308
large vehicle       0.6920393617449944
small vehicle       0.4761179180256544
helicopter          0.09900990099009901
roundabout          0.39167969023502114
soccer ball field   0.29171844304735667
swimming pool       0.003863009251744847


## Features

- **Patch-Based Detection:** Automatically splits large images into smaller patches to overcome YOLO's input size limitations.
- **Confidence-Weighted Merging:** Combines overlapping detections using weighted confidence scores proportional to their areas.
- **Class Compatibility:** Ensures merging only occurs between detections of the same class.
- **Shapely Integration:** Uses Shapely for precise geometric operations during merging.
- **YOLO Results Format:** Outputs detection results in the standard YOLO `Results` class format for compatibility.

---

## Installation

Install YOLOPatchFusion using pip:

```bash
pip install yolopatchfusion
```

---

## Requirements

- Python >= 3.8
- ultralytics
- shapely
- numpy
- opencv-python

Install dependencies using pip:

```bash
pip install ultralytics shapely numpy opencv-python
```

---

## Usage

### Basic Example

```python
import cv2
from yolopatchfusion import YOLOPatchFusion

# Initialize the wrapper
wrapper = YOLOPatchFusion(model_path="yolov8.pt")

# Load a large image
image = cv2.imread("large_image.jpg")

# Perform detection
results = wrapper.detect(image, overlap=50)

# Print results
print(results.boxes)  # Bounding boxes, confidences, and classes

# Visualize the results
results.plot()
```

---

## API Reference

### `YOLOPatchFusion`

#### `__init__(model_path: str, img_size: int = 640)`
Initializes the YOLOPatchFusion object.

- **model_path**: Path to the YOLO model file.
- **img_size**: Size of the image patches (default: 640).

#### `detect(image: np.ndarray, overlap: int = 50)`
Performs object detection on a large image.

- **image**: Input image as a NumPy array.
- **overlap**: Overlap size between patches to ensure no information is missed (default: 50).

Returns:
- A YOLO `Results` object containing the merged detections.

---

## Tests

Unit tests for this library are included in the `tests` folder. To run the tests:

```bash
pytest tests
```

---

## Contributing

Contributions are welcome! If you encounter a bug or have suggestions for improvement, feel free to open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [Ultralytics](https://ultralytics.com/) for the YOLO model.
- [Shapely](https://shapely.readthedocs.io/) for powerful geometric operations.
- The open-source community for continuous support and inspiration.

