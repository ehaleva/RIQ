"""download alexnet model"""
from torchvision.models import alexnet, AlexNet_Weights
import torch

model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
model.eval()
inputs = torch.randn(1, 3, 224, 224)
in_names = ["actual_input"]
out_name = ["output"]
torch.onnx.export(model, inputs, "models/alexnet.onnx", verbose=False, input_names=in_names,
                  output_names=out_name, export_params=True)
