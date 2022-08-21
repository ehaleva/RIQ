"""download resnet model"""
import torch
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()
inp = torch.randn(1, 3, 224, 224)
in_names = ["actual_input"]
out_name = ["output"]
torch.onnx.export(model, inp, "models/resnet.onnx", verbose=False, input_names=in_names,
                  output_names=out_name, export_params=True)
