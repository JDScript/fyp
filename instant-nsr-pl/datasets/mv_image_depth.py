import numpy as np
import pytorch_lightning as pl
import PIL.Image
from torch.utils.data import Dataset, IterableDataset, DataLoader
from utils.misc import get_rank
from pathlib import Path
from models.ray_utils import get_ortho_ray_directions_origins

import datasets
import os
import torch


def camNormal2worldNormal(rot_c2w, camNormal):
    H, W, _ = camNormal.shape
    normal_img = np.matmul(
        rot_c2w[None, :, :], camNormal.reshape(-1, 3)[:, :, None]
    ).reshape([H, W, 3])

    return normal_img


def img2normal(img):
    return (img / 255.0) * 2 - 1


def RT_opengl2opencv(RT):
    # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    R = RT[:3, :3]
    t = RT[:3, 3]

    R_bcam2cv = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)

    R_world2cv = R_bcam2cv @ R
    t_world2cv = R_bcam2cv @ t

    RT = np.concatenate([R_world2cv, t_world2cv[:, None]], 1)

    return RT


def normal_opengl2opencv(normal):
    H, W, C = np.shape(normal)
    # normal_img = np.reshape(normal, (H*W,C))
    R_bcam2cv = np.array([1, -1, -1], np.float32)
    normal_cv = normal * R_bcam2cv[None, None, :]

    print(np.shape(normal_cv))

    return normal_cv


def inv_RT(RT):
    RT_h = np.concatenate([RT, np.array([[0, 0, 0, 1]])], axis=0)
    RT_inv = np.linalg.inv(RT_h)

    return RT_inv[:3, :]


class MVImageDepthDatasetBase:
    VIEWS = ["front", "front_right", "right", "back", "left", "front_left"]

    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()

        # Inside configurations
        self.result_dir = Path(config.result_dir)
        self.cam_poses_dir = Path(config.cam_poses_dir)
        self.img_size: tuple[int, int] = self.config.img_size
        self.scene: str = self.config.scene

        self.w = self.img_size[0]
        self.h = self.img_size[1]

        self.view_weights = self.view_weights = (
            torch.from_numpy(np.array(self.config.view_weights))
            .float()
            .to(self.rank)
            .view(-1)
        )
        self.view_weights = self.view_weights.view(-1, 1, 1).repeat(1, self.h, self.w)

        (
            images_np,
            depths_np,
            masks_np,
            normals_cam_np,
            normals_world_np,
            poses_all_np,
            w2cs_np,
            origins_np,
            directions_np,
            rgb_masks_np,
        ) = self.load_diffusion_result()

        self.has_mask = True
        self.apply_mask = self.config.apply_mask

        self.all_c2w = torch.from_numpy(poses_all_np)
        self.all_images = torch.from_numpy(images_np) / 255.0
        self.all_depths = torch.from_numpy(depths_np)
        self.all_fg_masks = torch.from_numpy(masks_np)
        self.all_rgb_masks = torch.from_numpy(rgb_masks_np)
        self.all_normals_world = torch.from_numpy(normals_world_np)
        self.origins = torch.from_numpy(origins_np)
        self.directions = torch.from_numpy(directions_np)

        self.directions = self.directions.float().to(self.rank)
        self.origins = self.origins.float().to(self.rank)
        self.all_rgb_masks = self.all_rgb_masks.float().to(self.rank)
        (
            self.all_c2w,
            self.all_images,
            self.all_depths,
            self.all_fg_masks,
            self.all_normals_world,
        ) = (
            self.all_c2w.float().to(self.rank),
            self.all_images.float().to(self.rank),
            self.all_depths.float().to(self.rank),
            self.all_fg_masks.float().to(self.rank),
            self.all_normals_world.float().to(self.rank),
        )

    def load_cam_poses(self):
        poses = {}
        for face in MVImageDepthDatasetBase.VIEWS:
            RT = np.loadtxt(self.cam_poses_dir / f"_{face}_RT.txt")
            poses[face] = RT

        return poses

    def load_diffusion_result(self):
        all_images = []
        all_depths = []
        all_normals = []
        all_normals_world = []
        all_masks = []
        all_color_masks = []
        all_poses = []
        all_w2cs = []
        directions = []
        ray_origins = []

        RT_front = np.loadtxt(self.cam_poses_dir / "_front_RT.txt")
        RT_front_cv = RT_opengl2opencv(RT_front)

        for view in MVImageDepthDatasetBase.VIEWS:
            image_path = self.result_dir / self.scene / f"rgb_{view}.png"
            depth_path = self.result_dir / self.scene / f"depth_{view}.png"
            normal_path = self.result_dir / self.scene / f"normal_{view}.png"

            image = np.array(PIL.Image.open(image_path).resize(self.img_size))
            depth = np.array(PIL.Image.open(depth_path).resize(self.img_size))
            normal = np.array(PIL.Image.open(normal_path).resize(self.img_size))

            mask = depth[:, :, 3]
            color_mask = image[:, :, 3]
            invalid_color_mask = color_mask < 255 * 0.5
            threshold = np.ones_like(image[:, :, 0]) * 250
            invalid_white_mask = (
                (image[:, :, 0] > threshold)
                & (image[:, :, 1] > threshold)
                & (image[:, :, 2] > threshold)
            )
            invalid_color_mask_final = invalid_color_mask & invalid_white_mask
            color_mask = (1 - invalid_color_mask_final) > 0

            image = image[:, :, :3]

            depth = np.mean(depth[:, :, :3], axis=2, keepdims=True)
            depth = depth / 255.0

            RT = np.loadtxt(self.cam_poses_dir / f"_{view}_RT.txt")
            normal = normal[:, :, :3]
            normal = img2normal(normal)
            normal[mask == 0] = [0, 0, 0]
            mask = mask > (0.5 * 255)

            all_images.append(image)
            all_depths.append(depth)
            all_masks.append(mask)
            all_color_masks.append(color_mask)
            RT_cv = RT_opengl2opencv(RT)  # convert normal from opengl to opencv
            all_poses.append(inv_RT(RT_cv))  # cam2world
            all_w2cs.append(RT_cv)

            normal_cam_cv = normal_opengl2opencv(normal)
            normal_world = camNormal2worldNormal(
                inv_RT(RT_front_cv)[:3, :3], normal_cam_cv
            )
            all_normals.append(normal_cam_cv)
            all_normals_world.append(normal_world)

            origins, dirs = get_ortho_ray_directions_origins(
                W=self.img_size[0], H=self.img_size[1]
            )
            ray_origins.append(origins)
            directions.append(dirs)

        return (
            np.stack(all_images),
            np.stack(all_depths),
            np.stack(all_masks),
            np.stack(all_normals),
            np.stack(all_normals_world),
            np.stack(all_poses),
            np.stack(all_w2cs),
            np.stack(ray_origins),
            np.stack(directions),
            np.stack(all_color_masks),
        )


class MVImageDepthDataset(Dataset, MVImageDepthDatasetBase):
    def __init__(self, config, split) -> None:
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        return {"index": index}


class MVImageDepthIterableDataset(IterableDataset, MVImageDepthDatasetBase):
    def __init__(self, config, split) -> None:
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register("mv_image_depth")
class MVImageDepthDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        if stage in [None, "fit"]:
            self.train_dataset = MVImageDepthIterableDataset(self.config, "train")
        if stage in [None, "fit", "validate"]:
            self.val_dataset = MVImageDepthDataset(
                self.config, self.config.get("val_split", "train")
            )
        if stage in [None, "test"]:
            self.test_dataset = MVImageDepthDataset(
                self.config, self.config.get("test_split", "test")
            )
        if stage in [None, "predict"]:
            self.predict_dataset = MVImageDepthDataset(self.config, "train")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset,
            num_workers=os.cpu_count() or 1,
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler,
        )

    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)
