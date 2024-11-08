"""
arbitrary scale dataloader
"""
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import math

from torch.utils.data import Dataset


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(1, -1).permute(1, 0)
    return coord, rgb


class TrainAS(Dataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None, augment=False, d_trans=None, upscale_factor=None, n_patches=8):
        self.dataset = dataset
        self.inp_size = inp_size
        self.upscale_factor = upscale_factor
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.d_trans = d_trans
        self.n_patches = n_patches
        self.n = self.dataset.shape[0]

    def __len__(self):
        return self.dataset.shape[0]*self.n_patches

    def _get_img_low(self, img_h, h_lr, w_lr):
        img_h = torch.unsqueeze(img_h, 0)
        if self.d_trans == "uniform":
            img_l = img_h[:, :, ::self.upscale_factor, ::self.upscale_factor]
        elif self.d_trans == "noisy_uniform":
            img_l = img_h[:, :, ::self.upscale_factor, ::self.upscale_factor]
            img_l = img_l + self.noise_ratio * torch.randn(img_l.shape)
        elif self.d_trans == "bilinear":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="bilinear")
        elif self.d_trans == "bicubic":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="bicubic")
        elif self.d_trans == "nearest":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="nearest")
        else:
            raise NotImplementedError("This functionality has not been implemented yet")
        img_l = torch.squeeze(img_l, 0)
        return img_l

    def __getitem__(self, idx):
        img = self.dataset[idx%self.n,:,:,:]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)]
            img_down = self._get_img_low(img, h_lr, w_lr)
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = self._get_img_low(crop_hr, w_lr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        h_hr, w_hr = crop_hr.shape[1], crop_hr.shape[2]
        h_lr, w_lr = crop_lr.shape[1], crop_lr.shape[2]
        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr

        if self.inp_size is not None:
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr*w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(h_lr*w_lr, crop_hr.shape[0])

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        del img
        return {
            'img_low': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'img_high': hr_rgb
        }
    

class ValAS(Dataset):
    def __init__(self, dataset, crop_size=None, upscale_factor=None, d_trans=None):
        self.dataset = dataset
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.d_trans = d_trans 

    def __len__(self):
        return len(self.dataset)

    def _get_img_low(self, img_h, h_lr, w_lr):
        img_h = torch.unsqueeze(img_h, 0)
        if self.d_trans == "uniform":
            img_l = img_h[:, :, ::self.upscale_factor, ::self.upscale_factor]
        elif self.d_trans == "noisy_uniform":
            img_l = img_h[:, :, ::self.upscale_factor, ::self.upscale_factor]
            img_l = img_l + self.noise_ratio * torch.randn(img_l.shape)
        elif self.d_trans == "bilinear":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="bilinear")
        elif self.d_trans == "bicubic":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="bicubic")
        elif self.d_trans == "nearest":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="nearest")
        else:
            raise NotImplementedError("This functionality has not been implemented yet")
        img_l = torch.squeeze(img_l, 0)
        return img_l

    def __getitem__(self, idx):
        img = self.dataset[idx,:,:,:]
        transform = transforms.RandomCrop(self.crop_size)
        crop_hr = transform(img)
        crop_lr = self._get_img_low(crop_hr, int(self.crop_size/self.upscale_factor), int(self.crop_size/self.upscale_factor))

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        del img
        return {
            'img_low': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'img_high': hr_rgb
        }


class TestAS(Dataset):
    def __init__(self, dataset, upscale_factor=None, d_trans=None):
        self.dataset = dataset
        self.upscale_factor = upscale_factor
        self.d_trans = d_trans 

    def __len__(self):
        return len(self.dataset)

    def _get_img_low(self, img_h, h_lr, w_lr):
        img_h = torch.unsqueeze(img_h, 0)
        if self.d_trans == "uniform":
            img_l = img_h[:, :, ::self.upscale_factor, ::self.upscale_factor]
        elif self.d_trans == "noisy_uniform":
            img_l = img_h[:, :, ::self.upscale_factor, ::self.upscale_factor]
            img_l = img_l + self.noise_ratio * torch.randn(img_l.shape)
        elif self.d_trans == "bilinear":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="bilinear")
        elif self.d_trans == "bicubic":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="bicubic")
        elif self.d_trans == "nearest":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="nearest")
        else:
            raise NotImplementedError("This functionality has not been implemented yet")
        img_l = torch.squeeze(img_l, 0)
        return img_l

    def __getitem__(self, idx):
        crop_hr = self.dataset[idx,:,:,:]
        crop_lr = self._get_img_low(crop_hr, int(crop_hr.shape[1]/self.upscale_factor), int(crop_hr.shape[2]/self.upscale_factor))

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        
        return {
            'img_low': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'img_high': hr_rgb
        }


class TrainAS_E(Dataset):
    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None, augment=False, d_trans=None, upscale_factor=None, n_patches=8):
        self.dataset = dataset
        self.inp_size = inp_size
        self.upscale_factor = upscale_factor
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.d_trans = d_trans
        self.n_patches = n_patches
        self.n = self.dataset.shape[0]

    def __len__(self):
        return self.dataset.shape[0]*self.n_patches

    def _get_img_low(self, img_h, h_lr, w_lr):
        img_h = torch.unsqueeze(img_h, 0)
        if self.d_trans == "uniform":
            img_l = img_h[:, :, ::self.upscale_factor, ::self.upscale_factor]
        elif self.d_trans == "noisy_uniform":
            img_l = img_h[:, :, ::self.upscale_factor, ::self.upscale_factor]
            img_l = img_l + self.noise_ratio * torch.randn(img_l.shape)
        elif self.d_trans == "bilinear":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="bilinear")
        elif self.d_trans == "bicubic":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="bicubic")
        elif self.d_trans == "nearest":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="nearest")
        else:
            raise NotImplementedError("This functionality has not been implemented yet")
        img_l = torch.squeeze(img_l, 0)
        return img_l

    def __getitem__(self, idx):
        img = self.dataset[idx%self.n,:,:,:]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)]
            img_down = self._get_img_low(img, h_lr, w_lr)
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = self._get_img_low(crop_hr, w_lr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        h_hr, w_hr = crop_hr.shape[1], crop_hr.shape[2]
        h_lr, w_lr = crop_lr.shape[1], crop_lr.shape[2]
        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr

        if self.inp_size is not None:
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)

        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)
        del img
        return {
            'img_low': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'img_high': hr_rgb
        }
        
    
class ValAS_E(Dataset):
    def __init__(self, dataset, crop_size=None, upscale_factor=None, d_trans=None):
        self.dataset = dataset
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.d_trans = d_trans 

    def __len__(self):
        return len(self.dataset)

    def _get_img_low(self, img_h, h_lr, w_lr):
        img_h = torch.unsqueeze(img_h, 0)
        if self.d_trans == "uniform":
            img_l = img_h[:, :, ::self.upscale_factor, ::self.upscale_factor]
        elif self.d_trans == "noisy_uniform":
            img_l = img_h[:, :, ::self.upscale_factor, ::self.upscale_factor]
            img_l = img_l + self.noise_ratio * torch.randn(img_l.shape)
        elif self.d_trans == "bilinear":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="bilinear")
        elif self.d_trans == "bicubic":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="bicubic")
        elif self.d_trans == "nearest":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="nearest")
        else:
            raise NotImplementedError("This functionality has not been implemented yet")
        img_l = torch.squeeze(img_l, 0)
        return img_l

    def __getitem__(self, idx):
        img = self.dataset[idx,:,:,:]
        transform = transforms.RandomCrop(self.crop_size)
        crop_hr = transform(img)
        crop_lr = self._get_img_low(crop_hr, int(self.crop_size/self.upscale_factor), int(self.crop_size/self.upscale_factor))

        h_hr, w_hr = crop_hr.shape[1], crop_hr.shape[2]
        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr

        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        del img

        return {
            'img_low': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'img_high': hr_rgb
        }


class TestAS_E(Dataset):
    def __init__(self, dataset, upscale_factor=None, d_trans=None):
        self.dataset = dataset
        self.upscale_factor = upscale_factor
        self.d_trans = d_trans 

    def __len__(self):
        return len(self.dataset)

    def _get_img_low(self, img_h, h_lr, w_lr):
        img_h = torch.unsqueeze(img_h, 0)
        if self.d_trans == "uniform":
            img_l = img_h[:, :, ::self.upscale_factor, ::self.upscale_factor]
        elif self.d_trans == "noisy_uniform":
            img_l = img_h[:, :, ::self.upscale_factor, ::self.upscale_factor]
            img_l = img_l + self.noise_ratio * torch.randn(img_l.shape)
        elif self.d_trans == "bilinear":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="bilinear")
        elif self.d_trans == "bicubic":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="bicubic")
        elif self.d_trans == "nearest":
            img_l = torch.nn.functional.interpolate(img_h, [h_lr, w_lr], mode="nearest")
        else:
            raise NotImplementedError("This functionality has not been implemented yet")
        img_l = torch.squeeze(img_l, 0)
        return img_l

    def __getitem__(self, idx):
        crop_hr = self.dataset[idx,:,:,:]
        crop_lr = self._get_img_low(crop_hr, int(crop_hr.shape[1]/self.upscale_factor), int(crop_hr.shape[2]/self.upscale_factor))

        h_hr, w_hr = crop_hr.shape[1], crop_hr.shape[2]
        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr

        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'img_low': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'img_high': hr_rgb
        }