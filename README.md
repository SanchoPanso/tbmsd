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

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.432
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.689
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.824
 Average Recall     (AR) @[ IoU=0.50      | area= small | maxDets=1000 ] = 0.791
 Average Recall     (AR) @[ IoU=0.50      | area=medium | maxDets=1000 ] = 0.827
 Average Recall     (AR) @[ IoU=0.50      | area= large | maxDets=1000 ] = 0.851
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=  1 ] = 0.120
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 10 ] = 0.435
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.697
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.824
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.689
 Average Precision  (AP) @[ IoU=0.50      | area= small | maxDets=1000 ] = 0.602
 Average Precision  (AP) @[ IoU=0.50      | area=medium | maxDets=1000 ] = 0.733
 Average Precision  (AP) @[ IoU=0.50      | area= large | maxDets=1000 ] = 0.736
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=  1 ] = 0.107
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 10 ] = 0.371
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.579
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.689

Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ]
plane               0.9058150923294056
ship                0.8603419581742636
storage tank        0.7645199764207055
baseball diamond    0.6825703315792494
tennis court        0.922334608009698
basketball court    0.6096947676103865
ground track field  0.5872861944328955
harbor              0.7755330022370226
bridge              0.48229456411794625
large vehicle       0.815196674810924
small vehicle       0.6947861564961166
helicopter          0.5806016103732697
roundabout          0.5848797152345623
soccer ball field   0.4948219698651547
swimming pool       0.5673912626579126
[0.93796918 0.90837054 0.82548476 0.8364486  0.94342105 0.74242424
 0.81944444 0.87655502 0.73491379 0.88397538 0.82861346 0.83561644
 0.72067039 0.7124183  0.75454545]






 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.441
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.698
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.844
 Average Recall     (AR) @[ IoU=0.50      | area= small | maxDets=1000 ] = 0.814
 Average Recall     (AR) @[ IoU=0.50      | area=medium | maxDets=1000 ] = 0.849
 Average Recall     (AR) @[ IoU=0.50      | area= large | maxDets=1000 ] = 0.859
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=  1 ] = 0.124
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 10 ] = 0.441
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.714
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.844
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.698
 Average Precision  (AP) @[ IoU=0.50      | area= small | maxDets=1000 ] = 0.607
 Average Precision  (AP) @[ IoU=0.50      | area=medium | maxDets=1000 ] = 0.755
 Average Precision  (AP) @[ IoU=0.50      | area= large | maxDets=1000 ] = 0.739
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=  1 ] = 0.111
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 10 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.586
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.698

Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ]
plane               0.9162061957819215
ship                0.8648114759884202
storage tank        0.7648441801545823
baseball diamond    0.6914245165441879
tennis court        0.9224129619764025
basketball court    0.6347115083778028
ground track field  0.5653229184588898
harbor              0.782171655064399
bridge              0.5064707540770318
large vehicle       0.8247555485056474
small vehicle       0.6918322714029506
helicopter          0.5997527812868128
roundabout          0.5984913727756158
soccer ball field   0.5278681002312263
swimming pool       0.5850637764093632
[0.9454761  0.91741071 0.83795014 0.85514019 0.94736842 0.78787879
 0.81944444 0.88947368 0.77801724 0.89901983 0.8381758  0.8630137
 0.73743017 0.77777778 0.77272727]




 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.449
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.696
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.834
 Average Recall     (AR) @[ IoU=0.50      | area= small | maxDets=1000 ] = 0.794
 Average Recall     (AR) @[ IoU=0.50      | area=medium | maxDets=1000 ] = 0.842
 Average Recall     (AR) @[ IoU=0.50      | area= large | maxDets=1000 ] = 0.886
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=  1 ] = 0.128
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 10 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.710
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.834
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.696
 Average Precision  (AP) @[ IoU=0.50      | area= small | maxDets=1000 ] = 0.607
 Average Precision  (AP) @[ IoU=0.50      | area=medium | maxDets=1000 ] = 0.750
 Average Precision  (AP) @[ IoU=0.50      | area= large | maxDets=1000 ] = 0.744
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=  1 ] = 0.117
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 10 ] = 0.382
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.589
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.696

Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ]
plane               0.8545463097728714
ship                0.8433779920055917
storage tank        0.7561556188419762
baseball diamond    0.7014484993703304
tennis court        0.9199031068821425
basketball court    0.6157128273049626
ground track field  0.5937891447354743
harbor              0.7741137525258636
bridge              0.5385188368184577
large vehicle       0.8031389479318345
small vehicle       0.6840174493282588
helicopter          0.5883640530917555
roundabout          0.6004185757732811
soccer ball field   0.5610687658872018
swimming pool       0.6002957533552483
[0.90359542 0.89006696 0.82409972 0.85981308 0.93552632 0.74242424
 0.86805556 0.88899522 0.77586207 0.87257807 0.82438396 0.83561644
 0.73743017 0.79084967 0.76136364]



(1024, 2056), 0, no gluing, nms

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.441
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.703
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.855
 Average Recall     (AR) @[ IoU=0.50      | area= small | maxDets=1000 ] = 0.747
 Average Recall     (AR) @[ IoU=0.50      | area=medium | maxDets=1000 ] = 0.883
 Average Recall     (AR) @[ IoU=0.50      | area= large | maxDets=1000 ] = 0.877
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=  1 ] = 0.125
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 10 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.722
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.855
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.703
 Average Precision  (AP) @[ IoU=0.50      | area= small | maxDets=1000 ] = 0.516
 Average Precision  (AP) @[ IoU=0.50      | area=medium | maxDets=1000 ] = 0.730
 Average Precision  (AP) @[ IoU=0.50      | area= large | maxDets=1000 ] = 0.749
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=  1 ] = 0.113
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 10 ] = 0.381
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.591
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.703

Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ]
plane               0.9121599712231623
ship                0.8753356870745767
storage tank        0.7546596145473264
baseball diamond    0.7143360432946314
tennis court        0.9309268750270783
basketball court    0.6288466524987043
ground track field  0.618770864306244
harbor              0.8061155280805519
bridge              0.4973305811991161
large vehicle       0.828566577454708
small vehicle       0.715498205639324
helicopter          0.5657066623485834
roundabout          0.5952422411626994
soccer ball field   0.5289414091630956
swimming pool       0.5662788041247437
[0.9474516  0.92421875 0.82617729 0.87383178 0.95394737 0.78030303
 0.89583333 0.91770335 0.79094828 0.90517438 0.86079441 0.84931507
 0.75418994 0.77777778 0.76818182]


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

