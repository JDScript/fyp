from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import argparse
import os
import torch
import math
import logging
from omegaconf import OmegaConf
from diffusers import PNDMScheduler, AutoencoderKL
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from config import Config
from typing import cast
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from utils.cls import retrieve_class_from_string
from tqdm import tqdm
from einops import rearrange
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from models.pipeline import MVDiffusionImagePipeline
from torchvision.utils import make_grid, save_image
from collections import defaultdict
import diffusers
import transformers


logger = get_logger(__name__, log_level="INFO")

torch.backends.cuda.matmul.allow_tf32 = True


def log_validation(
    dataloader,
    vae,
    feature_extractor,
    image_encoder,
    unet,
    cfg: Config,
    accelerator,
    weight_dtype,
    global_step,
    name,
    save_dir,
):
    logger.info(f"Running {name} ... ")

    pipeline = MVDiffusionImagePipeline(
        image_encoder=image_encoder,
        feature_extractor=feature_extractor,
        vae=vae,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        scheduler=PNDMScheduler.from_pretrained(
            cfg.model.pretrained, subfolder="scheduler"
        ),
    )

    pipeline.set_progress_bar_config(disable=True)

    if is_xformers_available and torch.cuda.is_available():
        pipeline.enable_xformers_memory_efficient_attention()

    images_cond, images_gt, images_pred = [], [], defaultdict(list)
    for i, batch in tqdm(enumerate(dataloader), desc=f"{name} steps"):
        # (B, Nv, 3, H, W)
        imgs_in, colors_out, depths_out = (
            batch["imgs_in"],
            batch["imgs_out"],
            batch["depths_out"],
        )

        # repeat  (2B, Nv, 3, H, W)
        imgs_in = torch.cat([imgs_in] * 2, dim=0)
        imgs_out = torch.cat([depths_out, colors_out], dim=0)
        # (B, Nv, Nce)
        camera_embeddings = torch.cat([batch["camera_embeddings"]] * 2, dim=0)
        task_embeddings = torch.cat(
            [batch["depth_task_embeddings"], batch["color_task_embeddings"]], dim=0
        )
        camera_task_embeddings = torch.cat([camera_embeddings, task_embeddings], dim=-1)

        # (B*Nv, 3, H, W)
        imgs_in, imgs_out = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W"), rearrange(
            imgs_out, "B Nv C H W -> (B Nv) C H W"
        )
        # (B*Nv, Nce)
        camera_task_embeddings = rearrange(
            camera_task_embeddings, "B Nv Nce -> (B Nv) Nce"
        )

        images_cond.append(imgs_in)
        images_gt.append(imgs_out)
        with torch.autocast("cuda"):
            # B*Nv images
            for guidance_scale in [1.0, 3.0]:
                out = pipeline(
                    imgs_in,  # type: ignore
                    camera_task_embeddings,  # type: ignore
                    guidance_scale=guidance_scale,
                    output_type="pt",
                    num_images_per_prompt=1,
                    eta=1.0,
                ).images  # type: ignore
                shape = out.shape  # type:ignore
                out0, out1 = out[: shape[0] // 2], out[shape[0] // 2 :]
                out = []
                for ii in range(shape[0] // 2):
                    out.append(out0[ii])
                    out.append(out1[ii])
                out = torch.stack(out, dim=0)
                images_pred[f"{name}-sample_cfg{guidance_scale:.1f}"].append(out)
    images_cond_all = torch.cat(images_cond, dim=0)
    images_gt_all = torch.cat(images_gt, dim=0)
    images_pred_all = {}
    for k, v in images_pred.items():
        images_pred_all[k] = torch.cat(v, dim=0)

    nrow = 12
    images_cond_grid = make_grid(
        images_cond_all, nrow=nrow, padding=0, value_range=(0, 1)
    )
    images_gt_grid = make_grid(images_gt_all, nrow=nrow, padding=0, value_range=(0, 1))
    images_pred_grid = {}
    for k, v in images_pred_all.items():
        images_pred_grid[k] = make_grid(v, nrow=nrow, padding=0, value_range=(0, 1))
    save_image(
        images_cond_grid, os.path.join(save_dir, f"{global_step}-{name}-cond.jpg")
    )
    save_image(images_gt_grid, os.path.join(save_dir, f"{global_step}-{name}-gt.jpg"))
    for k, v in images_pred_grid.items():
        save_image(v, os.path.join(save_dir, f"{global_step}-{k}.jpg"))
    torch.cuda.empty_cache()


def train(conf: Config):
    accelerator_project_config = ProjectConfiguration(
        project_dir=conf.training.output_dir,
        logging_dir=os.path.join(conf.training.output_dir, "logs"),
    )

    vis_dir = os.path.join(conf.training.output_dir, "vis")

    accelerator = Accelerator(
        project_config=accelerator_project_config,
        mixed_precision=conf.training.mixed_precision,
        log_with=conf.training.log_with,
        gradient_accumulation_steps=conf.training.gradient_accumulation_steps,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if not os.path.exists(conf.training.output_dir):
            os.makedirs(conf.training.output_dir, exist_ok=True)
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir, exist_ok=True)
        OmegaConf.save(conf, os.path.join(conf.training.output_dir, "config.yaml"))

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
        logger.info(f"Loading pretrained UNet from {conf.model.pretrained_unet}")
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

    if conf.training.use_ema:
        ema_unet = EMAModel(
            parameters=unet.parameters(),
            model_cls=unet_cls,
            model_config=unet.config,
        )

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # Disable gradient for vae and image_encoder
    vae.requires_grad_(False)
    image_encoder.requires_grad_(False)

    if conf.training.trainable_modules is None:
        unet.requires_grad_(True)
    else:
        unet.requires_grad_(False)
        for name, module in unet.named_modules():
            if name.endswith(tuple(conf.training.trainable_modules)):
                for params in module.parameters():
                    params.requires_grad = True

    # Enable XFormers
    if is_xformers_available() and torch.cuda.is_available():
        print("using xformers")
        unet.enable_xformers_memory_efficient_attention()

    # Gradient checkpointing
    if conf.training.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Accelerator hooks
    def save_model_hook(models, weights, output_dir):
        if conf.training.use_ema:
            ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

        for i, model in enumerate(models):
            model.save_pretrained(os.path.join(output_dir, "unet"))

            # make sure to pop weight so that corresponding model is not saved again
            weights.pop()

    def load_model_hook(models, input_dir):
        if conf.training.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "unet_ema"), unet_cls
            )
            ema_unet.load_state_dict(load_model.state_dict())
            ema_unet.to(accelerator.device)
            del load_model

        for _ in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = unet_cls.from_pretrained(input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Optimizer
    params, params_class_embedding = [], []
    for name, param in unet.named_parameters():
        if "class_embedding" in name:
            params_class_embedding.append(param)
        else:
            params.append(param)
    optimizer = retrieve_class_from_string(conf.training.optimizer.target)(
        params=[
            {"params": params, "lr": conf.training.learning_rate},
            {"params": params_class_embedding, "lr": conf.training.learning_rate * 10},
        ],
        **conf.training.optimizer.params,
    )

    # Scheduler
    scheduler_params = conf.training.lr_scheduler.params or {}
    scheduler_params["num_training_steps"] = (
        conf.training.max_train_steps * accelerator.num_processes
    )
    scheduler_params["num_warmup_steps"] = (
        scheduler_params.get("num_warmup_steps", 100) * accelerator.num_processes
    )
    lr_scheduler = retrieve_class_from_string(conf.training.lr_scheduler.target)(
        optimizer=optimizer,
        **scheduler_params,
    )

    # Dataset and Dataloader
    train_dataset = retrieve_class_from_string(conf.datasets.train_dataset.target)(
        **conf.datasets.train_dataset.params
    )
    val_dataset = retrieve_class_from_string(conf.datasets.val_dataset.target)(
        **conf.datasets.val_dataset.params
    )
    train_val_dataset = retrieve_class_from_string(
        conf.datasets.train_val_dataset.target
    )(**conf.datasets.train_val_dataset.params)

    cpu_count = min(os.cpu_count() or 2, 8)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=conf.training.batch_size,
        shuffle=True,
        num_workers=cpu_count,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=conf.training.batch_size,
        shuffle=False,
        num_workers=cpu_count,
    )
    val_train_dataloader = torch.utils.data.DataLoader(
        train_val_dataset,
        batch_size=conf.training.batch_size,
        shuffle=False,
        num_workers=cpu_count,
    )

    # Prepare the model
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Convert all non-trainable weights to half-precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        conf.training.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        conf.training.mixed_precision = accelerator.mixed_precision
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if conf.training.use_ema:
        ema_unet.to(accelerator.device)

    clip_image_mean = torch.as_tensor(feature_extractor.image_mean)[:, None, None].to(
        accelerator.device, dtype=torch.float32
    )
    clip_image_std = torch.as_tensor(feature_extractor.image_std)[:, None, None].to(
        accelerator.device, dtype=torch.float32
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / conf.training.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(
        conf.training.max_train_steps / num_update_steps_per_epoch
    )

    if accelerator.is_main_process:
        tracker_config = {}
        accelerator.init_trackers(conf.name, tracker_config)

    total_batch_size = (
        conf.training.batch_size
        * accelerator.num_processes
        * conf.training.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {conf.training.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {conf.training.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {conf.training.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if conf.training.resume_from_checkpoint:
        if conf.training.resume_from_checkpoint != "latest":
            path = os.path.basename(conf.training.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            if os.path.exists(os.path.join(conf.training.output_dir, "checkpoint")):
                path = "checkpoint"
            else:
                dirs = os.listdir(conf.training.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{conf.training.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            conf.training.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(conf.training.output_dir, path))
            # global_step = int(path.split("-")[1])
            global_step = 0

            resume_global_step = global_step * conf.training.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * conf.training.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, conf.training.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                conf.training.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % conf.training.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # (B, Nv, 3, H, W), should be mixed out
                imgs_in, colors_out, depths_out = (
                    batch["imgs_in"],
                    batch["imgs_out"],
                    batch["depths_out"],
                )
                B, number_of_views = imgs_in.shape[0], imgs_in.shape[1]

                # repeat  (2B, Nv, 3, H, W)
                imgs_in = torch.cat([imgs_in] * 2, dim=0)
                imgs_out = torch.cat([depths_out, colors_out], dim=0)

                # (2B, Nv, Nce)
                camera_embeddings = torch.cat([batch["camera_embeddings"]] * 2, dim=0)
                task_embeddings = torch.cat(
                    [batch["depth_task_embeddings"], batch["color_task_embeddings"]],
                    dim=0,
                )
                camera_task_embeddings = torch.cat(
                    [camera_embeddings, task_embeddings], dim=-1
                )

                # (B*Nv, 3, H, W)
                imgs_in, imgs_out = rearrange(
                    imgs_in, "B Nv C H W -> (B Nv) C H W"
                ), rearrange(imgs_out, "B Nv C H W -> (B Nv) C H W")
                # (B*Nv, Nce)
                camera_task_embeddings = rearrange(
                    camera_task_embeddings, "B Nv Nce -> (B Nv) Nce"
                )
                # camera embbeddings e_de_da_sincos
                camera_task_embeddings = torch.cat(
                    [
                        torch.sin(camera_task_embeddings),
                        torch.cos(camera_task_embeddings),
                    ],
                    dim=-1,
                )

                imgs_in, imgs_out, camera_task_embeddings = (
                    imgs_in.to(weight_dtype),
                    imgs_out.to(weight_dtype),
                    camera_task_embeddings.to(weight_dtype),
                )

                # (B*Nv, 4, Hl, Wl)
                cond_vae_embeddings = vae.encode(imgs_in * 2.0 - 1.0).latent_dist.mode()  # type: ignore
                # scale input latents
                cond_vae_embeddings = cond_vae_embeddings * vae.config.scaling_factor  # type: ignore
                latents = vae.encode(imgs_out * 2.0 - 1.0).latent_dist.sample() * vae.config.scaling_factor  # type: ignore
                imgs_in_proc = TF.resize(
                    imgs_in,
                    [
                        feature_extractor.crop_size["height"],
                        feature_extractor.crop_size["width"],
                    ],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
                # do the normalization in float32 to preserve precision
                imgs_in_proc = (
                    (imgs_in_proc.float() - clip_image_mean) / clip_image_std
                ).to(weight_dtype)

                # (B*Nv, 1, 768)
                image_embeddings = image_encoder(imgs_in_proc).image_embeds.unsqueeze(1)

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # same noise for different views of the same object
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,  # type: ignore
                    (bsz // conf.model.num_views,),
                    device=latents.device,
                ).repeat_interleave(conf.model.num_views)
                timesteps = timesteps.long()

                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  # type: ignore

                # Conditioning dropout to support classifier-free guidance during inference.
                if conf.training.condition_drop_type == "drop_as_a_whole":
                    # drop a group of normals and colors as a whole
                    random_p = torch.rand(B, device=latents.device)

                    # Sample masks for the conditioning images.
                    image_mask_dtype = cond_vae_embeddings.dtype
                    image_mask = 1 - (
                        (random_p >= conf.training.condition_drop_rate).to(
                            image_mask_dtype
                        )
                        * (random_p < 3 * conf.training.condition_drop_rate).to(
                            image_mask_dtype
                        )
                    )
                    image_mask = image_mask.reshape(B, 1, 1, 1, 1).repeat(
                        1, number_of_views, 1, 1, 1
                    )
                    image_mask = rearrange(image_mask, "B Nv C H W -> (B Nv) C H W")
                    # Final image conditioning.
                    cond_vae_embeddings = image_mask * cond_vae_embeddings

                    # Sample masks for the conditioning images.
                    clip_mask_dtype = image_embeddings.dtype
                    clip_mask = 1 - (
                        (random_p < 2 * conf.training.condition_drop_rate).to(
                            clip_mask_dtype
                        )
                    )
                    clip_mask = clip_mask.reshape(B, 1, 1, 1).repeat(
                        1, number_of_views, 1, 1
                    )
                    clip_mask = rearrange(clip_mask, "B Nv M C -> (B Nv) M C")
                    # Final image conditioning.
                    image_embeddings = clip_mask * image_embeddings
                elif conf.training.condition_drop_type == "drop_independent":
                    random_p = torch.rand(bsz, device=latents.device)

                    # Sample masks for the conditioning images.
                    image_mask_dtype = cond_vae_embeddings.dtype
                    image_mask = 1 - (
                        (random_p >= conf.training.condition_drop_rate).to(
                            image_mask_dtype
                        )
                        * (random_p < 3 * conf.training.condition_drop_rate).to(
                            image_mask_dtype
                        )
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    cond_vae_embeddings = image_mask * cond_vae_embeddings

                    # Sample masks for the conditioning images.
                    clip_mask_dtype = image_embeddings.dtype
                    clip_mask = 1 - (
                        (random_p < 2 * conf.training.condition_drop_rate).to(
                            clip_mask_dtype
                        )
                    )
                    clip_mask = clip_mask.reshape(bsz, 1, 1)
                    # Final image conditioning.
                    image_embeddings = clip_mask * image_embeddings
                elif conf.training.condition_drop_type == "drop_joint":
                    # randomly drop all independently
                    random_p = torch.rand(bsz // 2, device=latents.device)

                    # Sample masks for the conditioning images.
                    image_mask_dtype = cond_vae_embeddings.dtype
                    image_mask = 1 - (
                        (random_p >= conf.training.condition_drop_rate).to(
                            image_mask_dtype
                        )
                        * (random_p < 3 * conf.training.condition_drop_rate).to(
                            image_mask_dtype
                        )
                    )
                    image_mask = torch.cat([image_mask] * 2, dim=0)
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    cond_vae_embeddings = image_mask * cond_vae_embeddings

                    # Sample masks for the conditioning images.
                    clip_mask_dtype = image_embeddings.dtype
                    clip_mask = 1 - (
                        (random_p < 2 * conf.training.condition_drop_rate).to(
                            clip_mask_dtype
                        )
                    )
                    clip_mask = torch.cat([clip_mask] * 2, dim=0)
                    clip_mask = clip_mask.reshape(bsz, 1, 1)
                    # Final image conditioning.
                    image_embeddings = clip_mask * image_embeddings

                # (B*Nv, 8, Hl, Wl)
                latent_model_input = torch.cat(
                    [noisy_latents, cond_vae_embeddings], dim=1
                )
                model_pred = unet(
                    latent_model_input,
                    timesteps,
                    encoder_hidden_states=image_embeddings,
                    class_labels=camera_task_embeddings,
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":  # type: ignore
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":  # type: ignore
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"  # type: ignore
                    )

                snr = compute_snr(timesteps)
                mse_loss_weights = (
                    torch.stack(
                        [snr, conf.training.snr_gamma * torch.ones_like(timesteps)],
                        dim=1,
                    ).min(dim=1)[0]
                    / snr
                )
                # We first calculate the original loss. Then we mean over the non-batch dimensions and
                # rebalance the sample-wise losses with their respective loss weights.
                # Finally, we take the mean of the rebalanced loss.
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(conf.training.batch_size)).mean()  # type: ignore
                train_loss += (
                    avg_loss.item() / conf.training.gradient_accumulation_steps
                )

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet.parameters(), conf.training.max_grad_norm
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if conf.training.use_ema:
                    ema_unet.step(unet.parameters())
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss}, step=global_step)
                    train_loss = 0.0
                if global_step % conf.training.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            conf.training.output_dir, f"checkpoint"
                        )
                        accelerator.save_state(save_path)
                        try:
                            unet.module.save_pretrained(
                                os.path.join(
                                    conf.training.output_dir, f"unet-{global_step}"
                                )
                            )
                        except:
                            unet.save_pretrained(
                                os.path.join(
                                    conf.training.output_dir, f"unet-{global_step}"
                                )
                            )
                        logger.info(f"Saved state to {save_path}")
                if global_step % conf.training.validation_steps == 0 or (
                    conf.training.validation_sanity_check and global_step == 1
                ):
                    if accelerator.is_main_process:
                        if conf.training.use_ema:
                            # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                            ema_unet.store(unet.parameters())
                            ema_unet.copy_to(unet.parameters())
                        log_validation(
                            val_dataloader,
                            vae,
                            feature_extractor,
                            image_encoder,
                            unet,
                            conf,
                            accelerator,
                            weight_dtype,
                            global_step,
                            "validation",
                            vis_dir,
                        )
                        log_validation(
                            val_train_dataloader,
                            vae,
                            feature_extractor,
                            image_encoder,
                            unet,
                            conf,
                            accelerator,
                            weight_dtype,
                            global_step,
                            "validation_train",
                            vis_dir,
                        )
                        if conf.training.use_ema:
                            # Switch back to the original UNet parameters.
                            ema_unet.restore(unet.parameters())

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= conf.training.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if conf.training.use_ema:
            ema_unet.copy_to(unet.parameters())
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
        os.makedirs(os.path.join(conf.training.output_dir, "pipeckpts"), exist_ok=True)
        pipeline.save_pretrained(os.path.join(conf.training.output_dir, "pipeckpts"))

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/training/mv_joint.yaml",
    )
    args = parser.parse_args()
    schema = OmegaConf.structured(Config)
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(schema, conf)
    print(OmegaConf.to_yaml(conf))
    train(cast(Config, conf))
