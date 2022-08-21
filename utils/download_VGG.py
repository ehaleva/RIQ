""" download VGG model"""
import torch
from torchvision.models import vgg16, VGG16_Weights

model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model.eval()
inp = torch.randn(1, 3, 224, 224)
in_names = ["actual_input"]
out_name = ["output"]
torch.onnx.export(model, inp, "models/VGG.onnx", verbose=False, input_names=in_names, output_names=out_name,
                  export_params=True)
