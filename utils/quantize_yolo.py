"""quantize yolo model"""
import sys
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import onnx
import numpy as np
from quantize import get_quantized_model, get_quantized_model_by_const
from dataset import prepare_dataset_images

def measure_cos_err(lhs, rhs):
    """measure cos error"""
    assert lhs.shape == rhs.shape, 'shapes are  not the same!'
    return lhs / np.linalg.norm(lhs) @ rhs / np.linalg.norm(rhs)

def compare_yolo_output(orig_output, quant_output, _):
    """compare function for yolo"""
    return measure_cos_err(orig_output[0].flatten()[-102000:], quant_output[0].flatten()[-102000:])

def main():
    """ main """
    base_onnx = sys.argv[1]
    new_model_path = os.path.join(sys.argv[2])
    if len(sys.argv) > 4:
        distortion = float(sys.argv[3])
        dataset_calibration_path = sys.argv[4] + '/*.jpg'
        dataset_calibration = prepare_dataset_images(dataset_calibration_path, base_onnx)
        quant_model = get_quantized_model(base_onnx, dataset_calibration, distortion=distortion,
                                          compare_function=compare_yolo_output)
    else:
        constant = float(sys.argv[3])
        quant_model = get_quantized_model_by_const(base_onnx, constant)
    onnx.save(quant_model.m, new_model_path)

main()
