import json
import torch
from abc import ABC
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from typing import List


class Calibrator(ABC):
    
    def __call__(self, results: List[Results]) -> List[Results]:
        pass


class ConfidenceCalibrator(Calibrator):
    def __init__(self, calibration_json):
        with open(calibration_json, 'r') as f:
            self.calibration_data = json.load(f)

    def __call__(self, results: List[Results]) -> List[Results]:
        for res in results:
            for det in res.boxes:
                cls = str(int(det.cls.cpu().numpy()))
                conf = det.conf.cpu().numpy()
                
                if cls in self.calibration_data:
                    conf_points = np.array(self.calibration_data[cls]['conf_points'])
                    precision_points = np.array(self.calibration_data[cls]['precision_points'])
                    
                    # Линейная интерполяция для преобразования уверенности в precision
                    calibrated_precision = np.interp(conf, conf_points, precision_points)
                    det.data[:, 4] = torch.tensor(calibrated_precision)
        
        return results


class SquareCalibrator(Calibrator):
    def __init__(self, calibration_json):
        with open(calibration_json, 'r') as f:
            self.calibration_data = json.load(f)

    def __call__(self, results: List[Results]) -> List[Results]:
        for res in results:
            for det in res.boxes:
                cls = str(int(det.cls.cpu().numpy()))
                conf = det.conf.cpu().numpy()
                xywh = det.xywh.cpu().numpy()
                
                w = xywh[:, 2]
                h = xywh[:, 3]
                sq = w * h

                if cls not in self.calibration_data:
                    continue
                
                square_points = np.array(self.calibration_data[cls]['square_points'])
                precision_points = np.array(self.calibration_data[cls]['precision_points'])
                
                # Линейная интерполяция для преобразования уверенности в precision
                precision = np.interp(sq, square_points, precision_points)
                calibrated_confidence = precision * conf
                det.data[:, 4] = torch.tensor(calibrated_confidence)
        
        return results



# Пример использования калибратора на инференсе
if __name__ == '__main__':
    calibrator = Calibrator('calib_1024.json')
    results = Results(orig_img=np.zeros((100, 100, 3)), path=None, names={"0", "plane"})
    results.boxes = Boxes(torch.tensor([0, 0, 1, 1, 0.2, 0]), (100, 100))
    calibrated_results = calibrator(results)
    print(calibrated_results[0].boxes)
