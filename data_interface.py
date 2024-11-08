import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from typing import Optional
from dataloaders.data_utils import get_Data
from dataloaders.SSdataloder import (TrainSS, ValSS, TestSS)
from dataloaders.ASdataloader import (TrainAS, ValAS, TestAS, TrainAS_E, ValAS_E, TestAS_E)


class DataModule(pl.LightningDataModule):
    def __init__(self, 
        name: str = "vorticity_Re_16000",
        data_dir: str = "data/vorticity_Re_16000.npy",
        model: str = "HiNOTE",
        pre_method: str = "zscore",
        d_trans: str = "uniform",
        reduce_dim: list = [],
        n_train_val_test: list = [800, 100, 100],
        b_train_val_test: list = [10, 10, 10],
        crop_size: int = 128,
        upscale_factor: int = 4,
        n_patches: int = 8,
        noise_ratio: float = 0.1,
        img_low_res: int = 32,
        scale_min: int = 1,
        scale_max: int = 4,
        augment: bool = False,
        sample_q: int = 1024,
        viz_size: list = [128, 128, 1024, 1024],
        num_workers: int = 4,
    ):
        super().__init__()

        self.name = name
        self.data_dir = data_dir
        self.model = model
        self.pre_method = pre_method
        self.reduce_dim = reduce_dim
        self.d_trans = d_trans
        self.n_train_val_test = n_train_val_test
        self.b_train, self.b_val, self.b_test = b_train_val_test
        self.crop_size = crop_size
        self.upscale_factor = upscale_factor
        self.n_patches = n_patches
        self.noise_ratio = noise_ratio
        self.img_low_res = img_low_res
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.viz_size = viz_size
        self.num_workers = num_workers
        self.d_singlescale = set(["DFNO", "EDSR", "ESPCN", "SRCNN", "SwinIR", "WDSR"])
        self.grid_arbitraryscale = set(["LIIF", "MetaSR", "LTE"])
        self.operator_arbitraryscale = set(["SRNO", "HiNOTE"])

    def setup(self, stage: Optional[str] = None):
        d_train, d_val, d_test, normalizer = get_Data(self.data_dir, self.n_train_val_test, self.pre_method, self.reduce_dim)
        if self.model in self.d_singlescale:
            self.train_data = TrainSS(d_train, self.crop_size, self.upscale_factor, self.n_patches, self.noise_ratio, self.d_trans)
            self.val_data   = ValSS(d_val, self.crop_size, self.upscale_factor, self.n_patches, self.noise_ratio, self.d_trans)
            self.test_data  = TestSS(d_test, self.upscale_factor, self.noise_ratio, self.d_trans)
        elif self.model in self.grid_arbitraryscale:
            self.train_data = TrainAS(d_train, self.img_low_res, self.scale_min, self.scale_max, self.augment, self.d_trans, self.upscale_factor, self.n_patches)
            self.val_data   = ValAS(d_val, self.crop_size, self.upscale_factor, self.d_trans)
            self.test_data  = TestAS(d_test, self.upscale_factor, self.d_trans)
        elif self.model in self.operator_arbitraryscale:
            self.train_data = TrainAS_E(d_train, self.img_low_res, self.scale_min, self.scale_max, self.augment, self.d_trans, self.upscale_factor, self.n_patches)
            self.val_data   = ValAS_E(d_val, self.crop_size, self.upscale_factor, self.d_trans)
            self.test_data  = TestAS_E(d_test, self.upscale_factor, self.d_trans)
        else:
            raise NotImplementedError("This functionality has not been implemented yet")

        del d_train, d_val, d_test
        return normalizer

    def train_dataloader(self):
        return DataLoader(dataset=self.train_data, batch_size=self.b_train, num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_data, batch_size=self.b_val, num_workers=self.num_workers, pin_memory=True, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test_data, batch_size=self.b_test, num_workers=self.num_workers, pin_memory=True, shuffle=False)