import argparse
import gradio as gr
import torch
import os
import PIL.Image
import rembg
import numpy as np
import subprocess
import glob
from datetime import datetime
from omegaconf import OmegaConf
from datasets import MVImageDepthDataset, SingleImageDataset
from torchvision.transforms import ToPILImage
from config import Config
from diffusers import PNDMScheduler, AutoencoderKL
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from utils.cls import retrieve_class_from_string
from models.pipeline import MVDiffusionImagePipeline
from torch.utils.data import DataLoader
from einops import rearrange
from utils.depth import depth2normal
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from functools import partial

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
remove_bg_session = rembg.new_session()

pipeline: None | MVDiffusionImagePipeline = None
pipelineLoading = False


def add_margin(pil_img, color=0, size=256):
    width, height = pil_img.size
    result = PIL.Image.new(pil_img.mode, (size, size), color)
    result.paste(pil_img, ((size - width) // 2, (size - height) // 2))
    return result


def tensor2ndarr(tensor: torch.Tensor):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu").numpy()

    assert isinstance(ndarr, np.ndarray)
    return ndarr


def lazy_load_pipeline():
    global pipeline
    global pipelineLoading
    if pipeline is not None or pipelineLoading:
        return
    pipelineLoading = True
    gr.Info("Model not loaded yet, may take some time to load the model")
    # Load all the components
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
        scheduler=DPMSolverMultistepScheduler.from_pretrained(
            conf.model.pretrained, subfolder="scheduler"
        ),
    )

    if torch.cuda.is_available():
        pipeline.to("cuda:0")
        if is_xformers_available():
            pipeline.enable_xformers_memory_efficient_attention()

    gr.Info("Model loaded, starting inferencing!")
    pipelineLoading = False


def visualize_dataset(index: int):
    out = dataset[index - 1]
    # merge the images
    images = torch.cat(tuple(out["imgs_out"]), dim=2)
    depths = torch.cat(tuple(out["depths_out"]), dim=2)

    return (
        ToPILImage()(images),
        ToPILImage()(depths),
    )


def run_inference(image: np.ndarray, denoising_step: int, guidance_scale: int):
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

    mid = out.shape[0] // 2
    depths_pred: list[torch.Tensor] = out[:mid]  # type: ignore
    images_pred: list[torch.Tensor] = out[mid:]  # type: ignore

    normals = [
        depth2normal(np.mean(tensor2ndarr(depths_pred[i]), axis=2)) for i in range(mid)
    ]
    images = [tensor2ndarr(images_pred[i]).astype(np.uint8) for i in range(mid)]
    depths = [tensor2ndarr(depths_pred[i]).astype(np.uint8) for i in range(mid)]

    # Save to disk
    cur_dir = Path(__file__).parent
    scene = "scene" + datetime.now().strftime("@%Y%m%d-%H%M%S")
    scene_dir = cur_dir / "outputs" / scene
    os.makedirs(scene_dir, exist_ok=True)
    for view_idx in range(6):
        view = SingleImageDataset.VIEWS[view_idx]
        rgb_filename = f"rgb_{view}.png"
        depth_filename = f"depth_{view}.png"
        normal_filename = f"normal_{view}.png"

        # Mask shape (256, 256)
        rgb_mask = rembg.remove(
            images[view_idx], only_mask=True, session=remove_bg_session
        )  # (256, 256)
        depth_mask = rembg.remove(
            depths[view_idx], only_mask=True, session=remove_bg_session
        )  # (256, 256)
        assert isinstance(rgb_mask, np.ndarray)
        assert isinstance(depth_mask, np.ndarray)

        # Apply mask with added alpha channel
        images[view_idx] = np.dstack([images[view_idx], rgb_mask])
        depths[view_idx] = np.dstack([depths[view_idx], depth_mask])
        normals[view_idx] = np.dstack([normals[view_idx], depth_mask])

        # Save
        PIL.Image.fromarray(images[view_idx]).save(scene_dir / rgb_filename)
        PIL.Image.fromarray(depths[view_idx]).save(scene_dir / depth_filename)
        PIL.Image.fromarray(normals[view_idx]).save(scene_dir / normal_filename)

    out = images + depths + normals + [scene]

    return out


def remove_bg(data: np.ndarray):
    image = rembg.remove(
        data,
        session=remove_bg_session,
    )

    assert isinstance(image, np.ndarray)

    alpha = image[:, :, 3]
    coords: np.ndarray[torch.Any, np.dtype[np.signedinteger[torch.Any]]] = np.stack(
        np.nonzero(alpha), 1
    )[:, (1, 0)]
    min_x, min_y = np.min(coords, 0)
    max_x, max_y = np.max(coords, 0)

    ref_image = PIL.Image.fromarray(image).crop((min_x, min_y, max_x, max_y))
    h, w = ref_image.height, ref_image.width
    scale = 144 / max(h, w)
    h_, w_ = int(scale * h), int(scale * w)
    ref_image = ref_image.resize((w_, h_))
    ref_image = add_margin(ref_image, size=256)
    img = np.array(ref_image)  # np.uint8

    alpha = img[..., 3:4]
    # White background to img
    img = img[..., :3] * (alpha / 255) + 255 * (1 - (alpha / 255))

    return img.astype(np.uint8)


def reconstruct3d(scene, need_reconstruct):
    if not need_reconstruct:
        return None
    cmds = [
        "python",
        "launch.py",
        "--config",
        "configs/neuralangelo-mvimage-depth.yaml",
        "--train",
        "--gpu",
        "0",
        f"dataset.scene={scene}",
    ]

    ret = subprocess.run(cmds, cwd=Path(__file__).parent / "instant-nsr-pl")
    if ret.returncode != 0:
        gr.Error(str(ret.stderr))
        return

    objs = glob.glob(
        f"{Path(__file__).parent}/instant-nsr-pl/exp/{scene}/*/save/*.obj",
        recursive=True,
    )
    if objs:
        return objs[0]
    gr.Error("No reconstructed obj file found")
    return None


with gr.Blocks(title="FYP23041 Demo") as demo:
    gr.Markdown("# FYP23041 Demo")
    with gr.Tab("Model Inference"):
        with gr.Row(equal_height=True):
            input_image = gr.Image(
                type="numpy",
                label="Input Image",
                sources=["upload"],
            )
            masked_image = gr.Image(
                type="numpy",
                label="Masked Image",
                interactive=False,
            )
            final_model = gr.Model3D(
                label="Reconstructed Model",
                interactive=False,
            )
        with gr.Row():
            need_reconstruct = gr.Checkbox(label="Reconstruct 3D Model", value=True)
            inference_button = gr.Button("Run", size="sm")
        with gr.Row():
            scene = gr.Textbox(value="scene", label="Scene Name", interactive=False)
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
            scene,
        ],
    ).success(
        reconstruct3d, inputs=[scene, need_reconstruct], outputs=[final_model]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
    )
    lazy_load_pipeline()
