from datasets import MVImageDepthDataset
import torch
import os
from tqdm import tqdm

train = MVImageDepthDataset(
    root="../autodl-tmp/new_renderings", split="train", mix_rgb_depth=False
)

val = MVImageDepthDataset(
    root="../autodl-tmp/new_renderings", split="val", mix_rgb_depth=False
)

cpu_count = min(os.cpu_count() or 2, 8)
train_dataloader = torch.utils.data.DataLoader(
    train,
    batch_size=32,
    shuffle=True,
    num_workers=cpu_count,
)
val_dataloader = torch.utils.data.DataLoader(
    val,
    batch_size=32,
    shuffle=False,
    num_workers=cpu_count,
)

for i, data in tqdm(enumerate(train_dataloader)):
  batch = data

for i, data in tqdm(enumerate(val_dataloader)):
  batch = data