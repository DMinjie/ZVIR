from thop import profile
import torch
import ZVIR_NET
model = ZVIR_NET.ZVIR_NET()
input_tensor = torch.randn(1, 1, 224, 224)
flops, params = profile(model, inputs=(input_tensor, input_tensor))
print(f"FLOPs: {flops/1e9:.2f} GFLOPs")
print(f"Params: {params/1e6:.2f} M")
