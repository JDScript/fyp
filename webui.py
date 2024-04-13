import argparse
import gradio as gr
import torch
import os
import PIL.Image
import rembg
import numpy as np
from omegaconf import OmegaConf
from datasets import MVImageDepthDataset, SingleImageDataset
from torchvision.transforms import ToPILImage
from config import Config
from diffusers import PNDMScheduler, AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from utils.cls import retrieve_class_from_string
from models.pipeline import MVDiffusionImagePipeline
from torch.utils.data import DataLoader
from einops import rearrange
from utils.depth import depth2normal

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count() or 1)

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
remove_bg_session = rembg.new_session("isnet-general-use")

pipeline: None | MVDiffusionImagePipeline = None


def save_image(tensor):
    ndarr = (
        tensor.mul(255)
        .add_(0.5)
        .clamp_(0, 255)
        .permute(1, 2, 0)
        .to("cpu", torch.uint8)
        .numpy()
    )
    return ndarr


def lazy_load_pipeline():
    global pipeline
    if pipeline is not None:
        return
    gr.Info("Model not loaded yet, may take some time to load the model")
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
    image_encoder.to(dtype=torch.float16)
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
    vae.to(dtype=torch.float16)
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
    unet.to(dtype=torch.float16)

    pipeline = MVDiffusionImagePipeline(
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        vae=vae,
        unet=unet,
        safety_checker=None,
        scheduler=PNDMScheduler.from_pretrained(
            conf.model.pretrained, subfolder="scheduler"
        ),
    )

    if torch.cuda.is_available():
        pipeline.to("cuda:0")
        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()

    gr.Info("Model loaded, starting inferencing!")


def visualize_dataset(index: int):
    out = dataset[index - 1]
    # merge the images
    images = torch.cat(tuple(out["imgs_out"]), dim=2)
    depths = torch.cat(tuple(out["depths_out"]), dim=2)

    return (
        ToPILImage()(images),
        ToPILImage()(depths),
    )


def run_inference(image: PIL.Image.Image, denoising_step: int, guidance_scale: int):
    global pipeline

    data = SingleImageDataset(image)
    loader = DataLoader(data, batch_size=1)
    batch = next(iter(loader))

    # repeat  (2B, Nv, 3, H, W)
    imgs_in = torch.cat([batch["imgs_in"]] * 2, dim=0)
    # (2B, Nv, Nce)
    camera_embeddings = torch.cat([batch["camera_embeddings"]] * 2, dim=0)
    task_embeddings = torch.cat(
        [batch["depth_task_embeddings"], batch["color_task_embeddings"]], dim=0
    )
    camera_task_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

    # (B*Nv, 3, H, W)
    imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")
    # (B*Nv, Nce)
    camera_task_embeddings = rearrange(camera_task_embeddings, "B Nv Nce -> (B Nv) Nce")

    assert isinstance(pipeline, MVDiffusionImagePipeline)

    out = pipeline(
        imgs_in,  # type: ignore
        camera_task_embeddings,  # type: ignore
        guidance_scale=guidance_scale,
        output_type="pt",
        num_images_per_prompt=1,
        eta=1.0,
        num_inference_steps=denoising_step,
    ).images

    bsz = out.shape[0] // 2
    depths_pred = out[:bsz]
    images_pred = out[bsz:]

    images_pred = [save_image(images_pred[i]) for i in range(bsz)]
    depths_pred = [
        np.mean(save_image(depths_pred[i]), axis=2).astype(np.uint8) for i in range(bsz)
    ]
    normal_pred = [
        depth2normal(
            np.mean(
                out[i]
                .mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to("cpu")
                .numpy(),
                axis=2,
            )
        )
        for i in range(bsz)
    ]

    out = images_pred + depths_pred + normal_pred

    return out


def remove_bg(data: PIL.Image.Image):
    data = data.resize((256, 256))
    mask = rembg.remove(
        data,
        session=remove_bg_session,
        bgcolor=(255, 255, 255, 255),
        alpha_matting=True,
        # sam_prompt=[{"type": "point", "data": [0, 0], "label": 1}],
    )
    return mask


with gr.Blocks(title="FYP23041 Demo") as demo:
    gr.Markdown("# FYP23041 Demo")
    with gr.Tab("Model Inference"):
        with gr.Row(equal_height=True):
            input_image = gr.Image(
                type="pil",
                label="Input Image",
                sources=["upload"],
            )
            masked_image = gr.Image(
                type="pil",
                label="Masked Image",
                interactive=False,
            )
            final_model = gr.Model3D(label="Reconstructed Model")
        with gr.Row():
            inference_button = gr.Button("Run", size="sm")
        with gr.Row():
            rgb1 = gr.Image(height=256, interactive=False, label="rgb_front")
            rgb2 = gr.Image(height=256, interactive=False, label="rgb_front_right")
            rgb3 = gr.Image(height=256, interactive=False, label="rgb_right")
            rgb4 = gr.Image(height=256, interactive=False, label="rgb_back")
            rgb5 = gr.Image(height=256, interactive=False, label="rgb_left")
            rgb6 = gr.Image(height=256, interactive=False, label="rgb_front_left")

            depth1 = gr.Image(height=256, interactive=False, label="depth_front")
            depth2 = gr.Image(height=256, interactive=False, label="depth_front_right")
            depth3 = gr.Image(height=256, interactive=False, label="depth_right")
            depth4 = gr.Image(height=256, interactive=False, label="depth_back")
            depth5 = gr.Image(height=256, interactive=False, label="depth_left")
            depth6 = gr.Image(height=256, interactive=False, label="depth_front_left")

            normal1 = gr.Image(
                height=256, interactive=False, label="computed_normal_front"
            )
            normal2 = gr.Image(
                height=256, interactive=False, label="computed_normal_front_right"
            )
            normal3 = gr.Image(
                height=256, interactive=False, label="computed_normal_right"
            )
            normal4 = gr.Image(
                height=256, interactive=False, label="computed_normal_back"
            )
            normal5 = gr.Image(
                height=256, interactive=False, label="computed_normal_left"
            )
            normal6 = gr.Image(
                height=256, interactive=False, label="computed_normal_front_left"
            )

    with gr.Tab("Dataset Visualization"):
        images = gr.Image(
            type="pil",
            height=256,
            width=6 * 256,
            interactive=False,
            label="Multi-view RGB Images",
        )
        depths = gr.Image(
            type="pil",
            height=256,
            width=6 * 256,
            interactive=False,
            label="Multi-view Depth Images",
        )
        slider = gr.Slider(
            minimum=1,
            maximum=len(dataset),
            step=1,
            label="Data Index",
            randomize=True,
        )
        btn = gr.Button("Show", size="sm")
        btn.click(visualize_dataset, inputs=[slider], outputs=[images, depths])
    with gr.Tab("Settings"):
        denoising_step = gr.Slider(
            minimum=20,
            maximum=100,
            step=1,
            label="Denoising Step",
            value=50,
        )
        guidance_scale = gr.Slider(
            minimum=1,
            maximum=15,
            step=1,
            label="Guidance Scale",
            value=3,
        )

    inference_button.click(
        remove_bg,
        inputs=[input_image],
        outputs=[masked_image],
    ).success(lazy_load_pipeline).success(
        run_inference,
        inputs=[masked_image, denoising_step, guidance_scale],
        outputs=[
            rgb1,
            rgb2,
            rgb3,
            rgb4,
            rgb5,
            rgb6,
            depth1,
            depth2,
            depth3,
            depth4,
            depth5,
            depth6,
            normal1,
            normal2,
            normal3,
            normal4,
            normal5,
            normal6,
        ],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
    )
