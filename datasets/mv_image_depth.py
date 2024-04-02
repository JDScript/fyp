from typing import Any
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, ToTensor
import torch
from torchvision.datasets.utils import (
    verify_str_arg,
)
import numpy as np
import torch
import random
import math


class MVImageDepthDataset(Dataset):
    VIEWS = ["front", "front_right", "right", "back", "left", "front_left"]

    def __init__(
        self,
        root="./data",
        split: str = "train",
        validation_samples=32,  # number of samples used for validation
        mix_rgb_depth=False,  # mix rgb and depth image for training
        bg_color="three_choices",
        size=(256, 256),
        num_views: int = 6,
        backup_scene: int = 0,
    ) -> None:
        self.root = Path(__file__).parent.parent / root
        self.split = verify_str_arg(split, "split", ("train", "val", "train_val"))
        self.uids_path = self.root / "valid_uids"
        self.uids = open(self.uids_path).read().splitlines()
        self.fixed_camera_poses = self.load_fixed_camera_poses()
        self.mix_rgb_depth = mix_rgb_depth
        self.bg_color = verify_str_arg(
            bg_color, "bg_color", ("three_choices", "white", "black", "gray")
        )
        self.size = size
        self.num_views = num_views

        if split == "train":
            self.uids = self.uids[:-validation_samples]
        if split == "val":
            self.uids = self.uids[-validation_samples:]
        if split == "train_val":
            self.uids = self.uids[:validation_samples]
        
        if self.mix_rgb_depth:
            self.backup = self.__getitem_mix__(backup_scene)
        else:
            self.backup = self.__getitem_joint__(backup_scene)

    def load_fixed_camera_poses(self):
        poses_dir = Path(__file__).parent / "./poses"
        poses = {}
        for face in MVImageDepthDataset.VIEWS:
            RT = np.loadtxt(poses_dir / f"_{face}_RT.txt")
            poses[face] = RT

        return poses

    def get_bg_color(self):
        white = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        black = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        gray = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        if self.bg_color == "white":
            return white
        elif self.bg_color == "black":
            return black
        elif self.bg_color == "gray":
            return gray
        else:
            return random.choice([white, black, gray])

    def load_image(self, img_path: Path, bg_color: np.ndarray):
        img = np.array(Image.open(img_path).resize(self.size))
        img = img.astype(np.float32) / 255.0

        assert img.shape[-1] == 4, "Should be RGBA image"

        alpha = img[:, :, 3:]
        img = img[:, :, :3]
        img = img[..., :3] * alpha + bg_color * (1 - alpha)

        return torch.from_numpy(img)

    def load_depth(self, depth_path: Path):
        depth = np.array(Image.open(depth_path).resize(self.size))
        depth = depth.astype(np.float32) / 255.0

        assert depth.shape[-1] == 4, "Should be RGBA image"

        alpha = depth[:, :, 3:]
        depth = depth[:, :, :3]
        depth = depth[..., :3] * alpha + np.array([1.0, 1.0, 1.0]) * (1 - alpha)

        return torch.from_numpy(depth)

    def cartesian_to_spherical(self, xyz):
        ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
        xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
        z = np.sqrt(xy + xyz[:, 2] ** 2)
        theta = np.arctan2(
            np.sqrt(xy), xyz[:, 2]
        )  # for elevation angle defined from Z-axis down
        # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
        azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
        return np.array([theta, azimuth, z])

    def get_T(self, target_RT, cond_RT):
        R, T = target_RT[:3, :3], target_RT[:, -1]
        T_target = -R.T @ T  # change to cam2world

        R, T = cond_RT[:3, :3], cond_RT[:, -1]
        T_cond = -R.T @ T

        theta_cond, azimuth_cond, z_cond = self.cartesian_to_spherical(T_cond[None, :])
        theta_target, azimuth_target, z_target = self.cartesian_to_spherical(
            T_target[None, :]
        )

        d_theta = theta_target - theta_cond
        d_azimuth = (azimuth_target - azimuth_cond) % (2 * math.pi)
        d_z = z_target - z_cond

        # d_T = torch.tensor([d_theta.item(), math.sin(d_azimuth.item()), math.cos(d_azimuth.item()), d_z.item()])
        return d_theta, d_azimuth

    def __getitem_mix__(self, index: int) -> dict[str, torch.Tensor]:
        uid = self.uids[index]
        condition_view = "front"  # thinking that may do not need to augment, since already done in data preprocessing

        if random.random() < 0.5:
            read_color, read_depth = True, False
        else:
            read_color, read_depth = False, True

        condition_w2c = self.fixed_camera_poses[condition_view]
        target_w2cs = [
            self.fixed_camera_poses[view] for view in MVImageDepthDataset.VIEWS
        ]

        elevations = []
        azimuths = []

        bg_color = self.get_bg_color()
        img_tensors_in = [
            self.load_image(
                self.root / uid / f"rgb_{condition_view}.webp",
                bg_color,
            ).permute(2, 0, 1)
        ] * self.num_views
        img_tensors_out = []

        for view, tgt_w2c in zip(MVImageDepthDataset.VIEWS, target_w2cs):
            if read_color:
                img_tensor = self.load_image(
                    self.root / uid / f"rgb_{view}.webp",
                    bg_color,
                ).permute(2, 0, 1)
                img_tensors_out.append(img_tensor)
            if read_depth:
                depth_tensor = self.load_depth(
                    self.root / uid / f"depth_{view}.webp",
                ).permute(2, 0, 1)
                img_tensors_out.append(depth_tensor)
            elevation, azimuth = self.get_T(tgt_w2c, condition_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        img_tensors_in = torch.stack(
            img_tensors_in, dim=0
        ).float()  # (number_of_views, 3, H, W)
        img_tensors_out = torch.stack(img_tensors_out, dim=0).float()

        elevations = torch.as_tensor(np.array(elevations)).float().squeeze(1)
        azimuths = torch.as_tensor(np.array(azimuths)).float().squeeze(1)
        elevations_cond = torch.as_tensor(np.array([0] * self.num_views)).float()
        camera_embeddings = torch.stack(
            [elevations_cond, elevations, azimuths], dim=-1
        )  # (number_of_views, 3)

        depth_class = torch.tensor([1, 0]).float()
        depth_task_embeddings = torch.stack(
            [depth_class] * self.num_views, dim=0
        )  # (number_of_views, 2)

        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack(
            [color_class] * self.num_views, dim=0
        )  # (number_of_views, 2)

        if read_color:
            task_embeddings = color_task_embeddings
        if read_depth:
            task_embeddings = depth_task_embeddings

        return {
            "elevations_cond": elevations_cond,
            "elevations_cond_deg": torch.rad2deg(elevations_cond),
            "elevations": elevations,
            "azimuths": azimuths,
            "elevations_deg": torch.rad2deg(elevations),
            "azimuths_deg": torch.rad2deg(azimuths),
            "imgs_in": img_tensors_in,
            "imgs_out": img_tensors_out,
            "camera_embeddings": camera_embeddings,
            "task_embeddings": task_embeddings,
        }

    def __getitem_joint__(self, index: int) -> dict[str, torch.Tensor]:
        uid = self.uids[index]
        condition_view = "front"

        condition_w2c = self.fixed_camera_poses[condition_view]
        target_w2cs = [
            self.fixed_camera_poses[view] for view in MVImageDepthDataset.VIEWS
        ]

        elevations = []
        azimuths = []

        bg_color = self.get_bg_color()

        img_tensors_in = [
            self.load_image(
                self.root / uid / f"rgb_{condition_view}.webp",
                bg_color,
            ).permute(2, 0, 1)
        ] * self.num_views
        img_tensors_out = []
        depth_tensors_out = []

        for view, tgt_w2c in zip(MVImageDepthDataset.VIEWS, target_w2cs):
            img_tensor = self.load_image(
                self.root / uid / f"rgb_{view}.webp",
                bg_color,
            ).permute(2, 0, 1)
            img_tensors_out.append(img_tensor)

            depth_tensor = self.load_depth(
                self.root / uid / f"depth_{view}.webp",
            ).permute(2, 0, 1)
            depth_tensors_out.append(depth_tensor)

            elevation, azimuth = self.get_T(tgt_w2c, condition_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        img_tensors_in = torch.stack(img_tensors_in, dim=0).float()
        img_tensors_out = torch.stack(img_tensors_out, dim=0).float()
        depth_tensors_out = torch.stack(depth_tensors_out, dim=0).float()

        elevations = torch.as_tensor(np.array(elevations)).float().squeeze(1)
        azimuths = torch.as_tensor(np.array(azimuths)).float().squeeze(1)
        elevations_cond = torch.as_tensor(np.array([0] * self.num_views)).float()
        camera_embeddings = torch.stack([elevations_cond, elevations, azimuths], dim=-1)

        depth_class = torch.tensor([1, 0]).float()
        depth_task_embeddings = torch.stack(
            [depth_class] * self.num_views, dim=0
        )  # (number_of_views, 2)

        color_class = torch.tensor([0, 1]).float()
        color_task_embeddings = torch.stack(
            [color_class] * self.num_views, dim=0
        )  # (number_of_views, 2)

        return {
            "elevations_cond": elevations_cond,
            "elevations_cond_deg": torch.rad2deg(elevations_cond),
            "elevations": elevations,
            "azimuths": azimuths,
            "elevations_deg": torch.rad2deg(elevations),
            "azimuths_deg": torch.rad2deg(azimuths),
            "imgs_in": img_tensors_in,
            "imgs_out": img_tensors_out,
            "depths_out": depth_tensors_out,
            "camera_embeddings": camera_embeddings,
            "depth_task_embeddings": depth_task_embeddings,
            "color_task_embeddings": color_task_embeddings,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        try:
            if self.mix_rgb_depth:
                return self.__getitem_mix__(index)
            return self.__getitem_joint__(index)
        except:
            uid = self.uids[index]
            print("load error ", uid)
            return self.backup

    def __len__(self) -> int:
        return len(self.uids)


if __name__ == "__main__":
    dataset = MVImageDepthDataset(root="../new_renderings", split="train_val", mix_rgb_depth=False)
    out = dataset[0]
    for key in out:
        print(key, out[key].shape)
