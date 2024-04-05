import argparse
import gradio as gr
import torch
import numpy as np
from omegaconf import OmegaConf
from datasets import MVImageDepthDataset
from torchvision.transforms import ToPILImage
from config import Config
from diffusers import PNDMScheduler, AutoencoderKL
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from utils.cls import retrieve_class_from_string

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="./configs/inference/mv_image.yaml",
)
args = parser.parse_args()
schema = OmegaConf.structured(Config)
conf = OmegaConf.load(args.config)
conf = OmegaConf.merge(schema, conf)

dataset = MVImageDepthDataset(root="../new_renderings", mix_rgb_depth=False)

def load_pipeline():
    # Load all the components
    noise_scheduler = PNDMScheduler.from_pretrained(
        pretrained_model_name_or_path=conf.model.pretrained,
        subfolder="scheduler",
    )
    assert isinstance(noise_scheduler, PNDMScheduler)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path=conf.model.pretrained,
        subfolder="image_encoder",
    )
    assert isinstance(image_encoder, CLIPVisionModelWithProjection)
    feature_extractor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name_or_path=conf.model.pretrained,
        subfolder="feature_extractor",
    )
    assert isinstance(feature_extractor, CLIPImageProcessor)
    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path=conf.model.pretrained,
        subfolder="vae",
    )
    assert isinstance(vae, AutoencoderKL)
    unet_cls = retrieve_class_from_string(conf.model.unet.target)
    if conf.model.pretrained_unet:
        unet = unet_cls.from_pretrained_2d(
            pretrained_model_name_or_path=conf.model.pretrained_unet,
            subfolder="unet",
            **conf.model.unet.params,  # type: ignore
        )
    else:
        unet = unet_cls.from_pretrained_2d(
            pretrained_model_name_or_path=conf.model.pretrained,
            subfolder="unet",
            **conf.model.unet.params,  # type: ignore
        )
    assert isinstance(unet, unet_cls)

load_pipeline()


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
