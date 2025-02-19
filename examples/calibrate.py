import json
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import List


class YOLOCalibrator:
    def __init__(self, model_path: str, data: str, output_json: str):
        self.model = YOLO(model_path)
        self.data = data
        self.output_json = output_json

    def calibrate(self):
        
        res = self.model.val(data=data)
        confidence = res.curves_results[2][0]
        calibration_data = {}
            
        for i, name in enumerate(self.model.names):
            precision = res.curves_results[2][1][0 + i]

            conf_points = np.linspace(0, 1, 6)
            ids = np.searchsorted(confidence, conf_points, 'left')
            precision_points = precision[ids]

            calibration_data[name] = {
                'conf_points': conf_points.tolist(),
                'precision_points': precision_points.tolist(),
            }

        # Сохранение калибровочных данных в JSON
        with open(self.output_json, 'w') as f:
            json.dump(calibration_data, f, indent=4)

        print(f'Calibration data saved to {self.output_json}')


model_path = 'dotav1_det/weights/best.pt' 
data = 'DOTAv1.yaml'
output_json = 'calib_1024.json'

calib = YOLOCalibrator(model_path, data, output_json)
calib.calibrate()
