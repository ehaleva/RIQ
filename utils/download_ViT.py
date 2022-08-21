"""download ViT model"""
import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights

model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
model.eval()
inp = torch.randn(1, 3, 224, 224)
in_names = ["actual_input"]
out_name = ["output"]
torch.onnx.export(model, inp, "models/ViT.onnx", verbose=False, input_names=in_names,
                  output_names=out_name, export_params=True)
