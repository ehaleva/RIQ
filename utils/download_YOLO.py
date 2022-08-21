"""download yolo model from sparsezoo"""
import os
from sparsezoo.models import Zoo

stub = "zoo:cv/detection/yolov5-s/pytorch/ultralytics/coco/base-none"
override_folder_name = 'sparsezoo_yolo'
model = Zoo.download_model_from_stub(stub, override_folder_name=override_folder_name,
                                     override_parent_path="models/downloads")
os.system('mv models/downloads/sparsezoo_yolo/model.onnx models/YOLO.onnx')
os.system('rm -dfr models/downloads')

