"""download yolo model from sparsezoo"""
import os
from sparsezoo import Model

stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none"
download_path = 'models/downloads/sparsezoo_yolo'
model = Model(stub, download_path=download_path)
os.system('mv models/downloads/sparsezoo_yolo/model.onnx models/YOLO.onnx')
os.system('rm -dfr models/downloads')

