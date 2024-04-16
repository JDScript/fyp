import torch
import math
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class SingleImageDataset(Dataset):
    VIEWS = ["front", "front_right", "right", "back", "left", "front_left"]

    def __init__(self, image: np.ndarray, num_views: int = 6) -> None:
        super().__init__()
        self.image = image
        self.fixed_camera_poses = self.load_fixed_camera_poses()
        self.num_views = num_views

    def load_fixed_camera_poses(self):
        poses_dir = Path(__file__).parent / "./poses"
        poses = {}
        for face in SingleImageDataset.VIEWS:
            RT = np.loadtxt(poses_dir / f"_{face}_RT.txt")
            poses[face] = RT

        return poses

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

    def load_image(self):
        img = np.array(self.image)
        img = img.astype(np.float32) / 255.0

        # img = img[:, :, :3]  # RGB
        # # apply mask to image
        # img = img * self.mask[:, :, None]
        # # add background to image
        # img[self.mask == 0] = bg_color

        return torch.from_numpy(img)

    def __getitem__(self, index):
        condition_view = "front"
        condition_w2c = self.fixed_camera_poses[condition_view]
        target_w2cs = [
            self.fixed_camera_poses[view] for view in SingleImageDataset.VIEWS
        ]

        elevations = []
        azimuths = []

        bg_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        img_tensors_in = [self.load_image().permute(2, 0, 1)] * self.num_views

        for view, tgt_w2c in zip(SingleImageDataset.VIEWS, target_w2cs):
            elevation, azimuth = self.get_T(tgt_w2c, condition_w2c)
            elevations.append(elevation)
            azimuths.append(azimuth)

        img_tensors_in = torch.stack(
            img_tensors_in, dim=0
        ).float()  # (number_of_views, 3, H, W)

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

        return {
            "elevations_cond": elevations_cond,
            "elevations_cond_deg": torch.rad2deg(elevations_cond),
            "elevations": elevations,
            "azimuths": azimuths,
            "elevations_deg": torch.rad2deg(elevations),
            "azimuths_deg": torch.rad2deg(azimuths),
            "imgs_in": img_tensors_in,
            "camera_embeddings": camera_embeddings,
            "depth_task_embeddings": depth_task_embeddings,
            "color_task_embeddings": color_task_embeddings,
        }

    def __len__(self):
        return 1
