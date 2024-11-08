"""
single scale dataloader
"""
import torch
import torchvision.transforms as transforms

from torch.utils.data import Dataset


class TrainSS(Dataset):
    def __init__(self, 
        img_h: torch.Tensor,
        crop_size: list,
        upscale_factor: int,
        n_patches: int,
        noise_ratio: float,
        d_trans: str,
    ):
        self.dat = img_h
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.n_patches = n_patches
        self.noise_ratio = noise_ratio
        self.d_trans = d_trans
        self.n = self.dat.shape[0]
        del img_h, crop_size, upscale_factor, n_patches, noise_ratio, d_trans

    def __len__(self):
        return self.dat.shape[0]*self.n_patches

    def __getitem__(self, idx):
        img = self.dat[idx%self.n,:,:,:]
        transform = transforms.RandomCrop(self.crop_size)
        img_h = transform(img)
        img_l = self._get_img_low(img_h)
        del img
        return img_l, img_h

    def _get_img_low(self, img_h):
        if self.d_trans == "uniform":
            img_l = img_h[:, ::self.upscale_factor, ::self.upscale_factor]
        elif self.d_trans == "noisy_uniform":
            img_l = img_h[:, ::self.upscale_factor, ::self.upscale_factor]
            img_l = img_l + self.noise_ratio * torch.randn(img_l.shape)
        elif self.d_trans == "bilinear":
            img_h = torch.unsqueeze(img_h, 0)
            img_l = torch.nn.functional.interpolate(img_h, [int(self.crop_size/self.upscale_factor), int(self.crop_size/self.upscale_factor)], mode="bilinear")
            img_l = torch.squeeze(img_l, 0)
        elif self.d_trans == "bicubic":
            img_h = torch.unsqueeze(img_h, 0)
            img_l = torch.nn.functional.interpolate(img_h, [int(self.crop_size/self.upscale_factor), int(self.crop_size/self.upscale_factor)], mode="bicubic")
            img_l = torch.squeeze(img_l, 0)
        elif self.d_trans == "nearest":
            img_h = torch.unsqueeze(img_h, 0)
            img_l = torch.nn.functional.interpolate(img_h, [int(self.crop_size/self.upscale_factor), int(self.crop_size/self.upscale_factor)], mode="nearest")
            img_l = torch.squeeze(img_l, 0)
        else:
            raise ValueError(f"Invalid method: {self.d_trans}")
        return img_l


class ValSS(Dataset):
    def __init__(self, 
        img_h: torch.Tensor,
        crop_size: list,
        upscale_factor: int,
        n_patches: int,
        noise_ratio: float,
        d_trans: str,
    ):
        self.dat = img_h
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.n_patches = n_patches
        self.noise_ratio = noise_ratio
        self.d_trans = d_trans
        self.n = self.dat.shape[0]
        del img_h, crop_size, upscale_factor, n_patches, noise_ratio, d_trans

    def __len__(self):
        return self.dat.shape[0]
            
    def __getitem__(self, idx):
        img = self.dat[idx%self.n,:,:,:]
        transform = transforms.RandomCrop(self.crop_size)
        img_h = transform(img)
        img_l = self._get_img_low(img_h)
        del img
        return img_l, img_h

    def _get_img_low(self, img_h):
        if self.d_trans == "uniform":
            img_l = img_h[:, ::self.upscale_factor, ::self.upscale_factor]
        elif self.d_trans == "noisy_uniform":
            img_l = img_h[:, ::self.upscale_factor, ::self.upscale_factor]
            img_l = img_l + self.noise_ratio * torch.randn(img_l.shape)
        elif self.d_trans == "bilinear":
            img_h = torch.unsqueeze(img_h, 0)
            img_l = torch.nn.functional.interpolate(img_h, [int(self.crop_size/self.upscale_factor), int(self.crop_size/self.upscale_factor)], mode="bilinear")
            img_l = torch.squeeze(img_l, 0)
        elif self.d_trans == "bicubic":
            img_h = torch.unsqueeze(img_h, 0)
            img_l = torch.nn.functional.interpolate(img_h, [int(self.crop_size/self.upscale_factor), int(self.crop_size/self.upscale_factor)], mode="bicubic")
            img_l = torch.squeeze(img_l, 0)
        elif self.d_trans == "nearest":
            img_h = torch.unsqueeze(img_h, 0)
            img_l = torch.nn.functional.interpolate(img_h, [int(self.crop_size/self.upscale_factor), int(self.crop_size/self.upscale_factor)], mode="nearest")
            img_l = torch.squeeze(img_l, 0)
        else:
            raise ValueError(f"Invalid method: {self.d_trans}")
        return img_l


class TestSS(Dataset):
    def __init__(self, 
        img_h: torch.Tensor,
        upscale_factor: int,
        noise_ratio: float,
        d_trans: str,
    ):
        self.dat = img_h
        self.h, self.w = img_h.shape[-2], img_h.shape[-1]
        self.upscale_factor = upscale_factor
        self.noise_ratio = noise_ratio
        self.d_trans = d_trans
        self.n = self.dat.shape[0]
        del img_h, upscale_factor, noise_ratio, d_trans

    def __len__(self):
        return self.dat.shape[0]

    def __getitem__(self, idx):
        img_h = self.dat[idx,:,:,:]
        img_l = self._get_img_low(img_h)
        return img_l, img_h

    def _get_img_low(self, img_h):
        if self.d_trans == "uniform":
            img_l = img_h[:, ::self.upscale_factor, ::self.upscale_factor]
        elif self.d_trans == "noisy_uniform":
            img_l = img_h[:, ::self.upscale_factor, ::self.upscale_factor]
            img_l = img_l + self.noise_ratio * torch.randn(img_l.shape)
        elif self.d_trans == "bilinear":
            img_h = torch.unsqueeze(img_h, 0)
            img_l = torch.nn.functional.interpolate(img_h, [int(self.h/self.upscale_factor), int(self.w/self.upscale_factor)], mode="bilinear")
            img_l = torch.squeeze(img_l, 0)
        elif self.d_trans == "bicubic":
            img_h = torch.unsqueeze(img_h, 0)
            img_l = torch.nn.functional.interpolate(img_h, [int(self.h/self.upscale_factor), int(self.w/self.upscale_factor)], mode="bicubic")
            img_l = torch.squeeze(img_l, 0)
        elif self.d_trans == "nearest":
            img_h = torch.unsqueeze(img_h, 0)
            img_l = torch.nn.functional.interpolate(img_h, [int(self.h/self.upscale_factor), int(self.w/self.upscale_factor)], mode="nearest")
            img_l = torch.squeeze(img_l, 0)
        else:
            raise ValueError(f"Invalid method: {self.d_trans}")
        return img_l