from models.model import UNetMV2DConditionModel, UNetMV2DConditionOutput
import torch
from torchviz import make_dot
import sys
import pickle

sys.setrecursionlimit(10000)

unet = UNetMV2DConditionModel.from_pretrained(
    "flamehaze1115/wonder3d-v1.0",
    subfolder="unet",
)
assert isinstance(unet, UNetMV2DConditionModel)
unet.to(torch.float16)
unet.to("cuda")
unet.enable_xformers_memory_efficient_attention()

out: UNetMV2DConditionOutput = unet(
    torch.randn(6, 8, 32, 32).to("cuda").to(torch.float16),
    torch.tensor(143).to("cuda"),
    torch.randn(6, 1, 768).to("cuda").to(torch.float16),
    torch.randn(6, 10).to("cuda").to(torch.float16),
)

dot = make_dot(out.sample, params=dict(unet.named_parameters()), show_attrs=True, show_saved=True)
pickle.dump(dot, open("unet.dot", "wb"))