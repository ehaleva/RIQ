"""quantize bert model"""
import sys
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import onnx
import numpy as np
from onnx_bridge import OnnxBridge
from quantize import get_quantized_model
from dataset import prepare_dataset_text

def measure_cos_err(lhs, rhs):
    """measure cos error"""
    assert lhs.shape == rhs.shape, 'shapes are  not the same!'
    return lhs / np.linalg.norm(lhs) @ rhs / np.linalg.norm(rhs)

def bert_compare_function(orig_output, quant_output, inputs):
    """compare function for bert"""
    size = np.sum(inputs[1])
    return measure_cos_err(orig_output[0].flatten()[:size], quant_output[0].flatten()[:size])

def main():
    """ main """
    base_onnx = sys.argv[1]
    new_model_path = os.path.join(sys.argv[2])
    distortion = float(sys.argv[3])
    dataset_path = sys.argv[4]
    model = OnnxBridge(base_onnx)
    dims = model.get_inputs()
    size_input = dims[1][1]
    dataset_calibration = prepare_dataset_text(dataset_path, size_input)

    quant_model = get_quantized_model(
        base_onnx,
        dataset_calibration,
        distortion=distortion,
        compare_function=bert_compare_function)
    onnx.save(quant_model.m, new_model_path)


main()
