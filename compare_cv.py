"""compare between the accuracy of the quant model and the base model"""
import os
import sys
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torchvision
import numpy as np
from utils import presets
from utils.onnx_bridge import OnnxBridge
from utils.quantize import get_quantized_model, measure_cos_distance
from utils.dataset import prepare_dataset_images

def measure_cos_err(lhs, rhs):
    """measure cos error"""
    assert lhs.shape == rhs.shape, 'shapes are  not the same!'
    return lhs / np.linalg.norm(lhs) @ rhs / np.linalg.norm(rhs)

def compare_function(orig_output, quant_output, _):
    """compare function for CV accuracy models"""
    return measure_cos_err(orig_output[0].flatten()[:], quant_output[0].flatten()[:])

def data_loader(valdir):
    """ loading data """
    preprocessing = presets.ClassificationPresetEval(crop_size=224)
    dataset_test = torchvision.datasets.ImageFolder(valdir, preprocessing)
    return dataset_test


def eval_model(base_model, quant_model, testloader):
    """ eval the model and compute accuracy"""
    correct_origin = 0
    correct_quant = 0
    total = 0
    i = 0
    self_similarity = 0
    total_cos_error = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs_base = list()
            oputputs_quant = list()
            for img in images:
                img = np.array([img.numpy()])
                output_quant = quant_model(img)[0][0, :]
                oputputs_quant.append(output_quant)
                output_base = base_model(img)[0][0, :]
                outputs_base.append(output_base)
                total_cos_error += measure_cos_distance(output_quant, output_base)
            _, predicted_base = torch.max(torch.nn.Softmax(dim=-1)
                                          (torch.Tensor(np.array(outputs_base))), 1)
            _, predicted_quant = torch.max(torch.nn.Softmax(dim=-1)
                                           (torch.Tensor(np.array(oputputs_quant))), 1)
            total += labels.size(0)
            correct_quant += (torch.sum(predicted_quant == labels)).item()
            correct_origin += (torch.sum(predicted_base == labels)).item()
            self_similarity += (predicted_base == predicted_quant).sum().item()
            i += 1

            print("iteration ", i, "self similarity: ", self_similarity,
                  100 * correct_origin / total, 100 * correct_quant / total)

    correct_origin = 100 * correct_origin / total
    correct_quant = 100 * correct_quant / total
    self_similarity = 100 * self_similarity / total
    total_cos_error = 100 * total_cos_error /total
    print(f'Accuracy of the original network  on the 50000 test images: {correct_origin} %')
    print(f'Accuracy of the quantized network on the 50000 test images: {correct_quant} %')
    print(f'self similarity model { self_similarity }%')
    print(f'measure cos distance: {total_cos_error}%')


def main():
    """main"""
    model_name = sys.argv[1]
    base_onnx = sys.argv[2]
    data_path = sys.argv[3]
    dataset_calibration_path = sys.argv[4]
    dataset_calibration_path = dataset_calibration_path + '*.JPEG'
    distortion = float(sys.argv[5])
    batch_size = 128
    base_model = OnnxBridge(base_onnx)
    quant_onnx_fn = base_onnx.replace(".onnx", "_" + str(distortion) + ".onnx")
    dataset_calibration = prepare_dataset_images(dataset_calibration_path, base_onnx)
    print("Quantizing with distortion constraint", distortion)
    quant_model = get_quantized_model(base_onnx, dataset_calibration, distortion=distortion,
                                      compare_function=compare_function)
    quant_model.save(quant_onnx_fn)
    valdir = os.path.join(data_path, 'val')
    dataset_test = data_loader(valdir)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                             num_workers=4, shuffle=True)
    print("Measuring Accuracy on validation dataset")
    eval_model(base_model, quant_model, testloader)

main()
