import gradio as gr
import torch
import numpy as np
from datasets import MVImageDepthDataset
from torchvision.transforms import ToPILImage

dataset = MVImageDepthDataset(root="../new_renderings", mix_rgb_depth=False)


def visualize_dataset(index: int):
    out = dataset[index - 1]
    # merge the images
    images = torch.cat(tuple(out["imgs_out"]), dim=2)
    depths = torch.cat(tuple(out["depths_out"]), dim=2)

    return ToPILImage()(images), ToPILImage()(depths)


with gr.Blocks(title="COMP4801 Demo WebUI") as demo:
    with gr.Column(variant="panel"):
        images = gr.Image(type="pil")
        depths = gr.Image(type="pil")
    with gr.Column():
        slider = gr.Slider(minimum=1, maximum=len(dataset), step=1)
        btn = gr.Button("Show")
        btn.click(visualize_dataset, inputs=[slider], outputs=[images, depths])


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
    )
