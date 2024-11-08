import torch
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from omegaconf import DictConfig
from models.SRCNN import SRCNN_net
from models.ESPCN import ESPCN_net
from models.EDSR import EDSR_net
from models.WDSR import WDSR_net
from models.SwinIR import SwinIR_net
from models.MetaSR import MetaSR_net
from models.LIIF import LIIF_net
from models.LTE import LTE_net
from models.DFNO import DFNO_net
from models.SRNO import SRNO_net
from models.HiNOTE import HiNOTE_net

from utils import (get_optimizer, get_scheduler, get_loss, toNumpy)
from torchmetrics.regression import MeanSquaredError
from torchmetrics.image import (PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure)


#---------------------------------------------------------
# get model
#---------------------------------------------------------
def get_model(cfg):
    """
    Set model.
    Args:
        cfg: Model configuration.
    Returns:
        Model will be use for modeling.
    """
    if cfg.name == "SRCNN":
        model = SRCNN_net(cfg.in_channels,
                         cfg.upscale_factor)
    elif cfg.name == "ESPCN":
        model = ESPCN_net(cfg.in_channels, 
                         cfg.upscale_factor, 
                         cfg.width)
    elif cfg.name == "EDSR":
        model = EDSR_net(cfg.in_channels, 
                         cfg.hidden_channels, 
                         cfg.n_res_blocks, 
                         cfg.upscale_factor)
    elif cfg.name == "WDSR":
        model = WDSR_net(cfg.in_channels,
                         cfg.out_channels, 
                         cfg.hidden_channels, 
                         cfg.n_res_blocks, 
                         cfg.upscale_factor)
    elif cfg.name == "MetaSR":
        model = MetaSR_net(cfg.edsr_n_resblocks,
                         cfg.edsr_n_feats,
                         cfg.edsr_res_scale,
                         cfg.edsr_scale,
                         cfg.edsr_no_upsampling,
                         cfg.edsr_rgb_range,
                         cfg.mlp_in_dim,
                         cfg.mlp_hidden_list)
    elif cfg.name == "LIIF":
        model = LIIF_net(cfg.edsr_n_resblocks,
                         cfg.edsr_n_feats,
                         cfg.edsr_res_scale,
                         cfg.edsr_scale,
                         cfg.edsr_no_upsampling,
                         cfg.edsr_rgb_range,
                         cfg.mlp_out_dim,
                         cfg.mlp_hidden_list,
                         cfg.local_ensemble,
                         cfg.feat_unfold,
                         cfg.cell_decode)
    elif cfg.name == "LTE":
        model = LTE_net(cfg.edsr_n_resblocks,
                         cfg.edsr_n_feats,
                         cfg.edsr_res_scale,
                         cfg.edsr_scale,
                         cfg.edsr_no_upsampling,
                         cfg.edsr_rgb_range,
                         cfg.mlp_out_dim,
                         cfg.mlp_hidden_list,
                         cfg.hidden_dim)        
    elif cfg.name == "SwinIR":
        h = (cfg.resol_h // cfg.upscale_factor // cfg.window_size + 1) * cfg.window_size
        w = (cfg.resol_w // cfg.upscale_factor // cfg.window_size + 1) * cfg.window_size
        model = SwinIR_net(upscale=cfg.upscale_factor, 
                           in_chans=cfg.in_channels, 
                           img_size=(h, w),
                           window_size=cfg.window_size, 
                           img_range=cfg.img_range, 
                           depths=cfg.depths, 
                           embed_dim=cfg.embed_dim, 
                           num_heads=cfg.num_heads, 
                           mlp_ratio=cfg.mlp_ratio, 
                           upsampler=cfg.upsampler, 
                           resi_connection=cfg.resi_connection)
    elif cfg.name == "DFNO":
        model = DFNO_net(cfg.in_feats,
                         cfg.n_feats,
                         cfg.out_feats,
                         cfg.n_res_blocks,
                         cfg.upscale_factor,
                         cfg.modes1,
                         cfg.modes2,
                         cfg.width)
    elif cfg.name == "SRNO":
        model = SRNO_net(cfg.n_resblocks,
                         cfg.n_feats,
                         cfg.res_scale,
                         cfg.scale,
                         cfg.no_upsampling,
                         cfg.rgb_range,
                         cfg.width,
                         cfg.blocks)
    elif cfg.name == "HiNOTE":
        model = HiNOTE_net(cfg.feature_up_ratio,
                           cfg.fourier_up,
                           cfg.feature_combine,
                           cfg.non_act,
                           cfg.liif_up_method,
                           cfg.attention_width,
                           cfg.attention_head,
                           cfg.hierarchical_levels)
    return model


#---------------------------------------------------------
# visualization
#---------------------------------------------------------
def plotSample(yhat, yref, dir_save, sample_name):
    """
    Args:
        yhat (numpy.array): (b, lat, lon)
        yref (numpy.array): (b, lat, lon)
    """
    cmap = plt.get_cmap("RdBu_r")
    plt.close("all")
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    # Reference
    ax1.set_title("Reference")
    cset1 = ax1.imshow(yref, cmap=cmap)
    ax1.set_xticks([], [])
    ax1.set_yticks([], [])
    fig.colorbar(cset1, ax=ax1)
    # Prediction
    ax2.set_title("Prediction")
    cset2 = ax2.imshow(yhat, cmap=cmap)
    ax2.set_xticks([], [])
    ax2.set_yticks([], [])
    fig.colorbar(cset2, ax=ax2)
    # Error
    ax3.set_title("Error")
    cset3 = ax3.imshow(yhat-yref, cmap=cmap)
    ax3.set_xticks([], [])
    ax3.set_yticks([], [])
    fig.colorbar(cset3, ax=ax3)
    plt.savefig(dir_save + "/" + sample_name + ".png", bbox_inches="tight")


#---------------------------------------------------------
# Model pytorch lightningmodule single-scale
#---------------------------------------------------------
class SingleScaleModule(pl.LightningModule):
    def __init__(self,
        normalizer,
        params_model: DictConfig,
        params_optim: DictConfig,
        params_scheduler: DictConfig,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.cfg_model     = params_model
        self.cfg_optim     = params_optim
        self.cfg_scheduler = params_scheduler

        self.model      = get_model(self.cfg_model)
        self.optimizer  = get_optimizer(list(self.model.parameters()), self.cfg_optim)
        self.scheduler  = get_scheduler(self.optimizer, self.cfg_scheduler)
        self.criterion  = get_loss(self.cfg_optim.loss)

        self.normalizer = normalizer
        self.sync_dist = torch.cuda.device_count() > 1
        self.validation_step_yhat = []
        self.validation_step_yref = []
        self.test_step_yhat = []
        self.test_step_yref = []
        self.m_MSE = MeanSquaredError()
        self.m_PSNR = PeakSignalNoiseRatio()
        self.m_SSIM = StructuralSimilarityIndexMeasure()

    def step(self, batch: Any):
        """
        Args:
        input_x, output_y, idx
            x: yref (torch.tensor) - (b, c, h_l, w_l)
            yref: img_h (torch.tensor) - (b, c, h_h, w_h)
        Returns:
            loss (torch.tensor) - (1)
            yhat (torch.tensor) - (b, c, h_h, w_h)
            yref (torch.tensor) - (b, c, h_h, w_h)
        """
        x, yref = batch
        yhat = self.model(x)
        if self.cfg_optim.loss == "FPLoss":
            loss = self.criterion(yhat, yref, x)
        else:
            loss = self.criterion(yhat, yref)
        return loss, yhat, yref

    def training_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log("train/mse", self.m_MSE(yhat, yref), sync_dist=self.sync_dist)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        _, yhat, yref = self.step(batch)
        self.validation_step_yhat.append(yhat)
        self.validation_step_yref.append(yref)
        return {"yref": yref, "yhat": yhat}

    def on_validation_epoch_end(self):
        yhats = torch.cat(self.validation_step_yhat, dim=0)
        yrefs = torch.cat(self.validation_step_yref, dim=0)
        self.log("validation/mse", self.m_MSE(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("validation/psnr", self.m_PSNR(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("validation/ssim", self.m_SSIM(yhats, yrefs), sync_dist=self.sync_dist)
        # visualization
        b_size = 3
        if (self.current_epoch+1)%50 == 0:
            for idx in range(b_size): 
                plotSample(toNumpy(torch.squeeze(yhats[idx,:,:,:])), toNumpy(torch.squeeze(yrefs[idx,:,:,:])), self.cfg_model.save_dir, f"val_epoch_{self.current_epoch}_idx_{idx}")
        # free memory
        self.validation_step_yhat.clear()
        self.validation_step_yref.clear()

    def test_step(self, batch: Any, batch_idx: int):
        _, yhat, yref = self.step(batch)
        self.test_step_yhat.append(yhat)
        self.test_step_yref.append(yref)
        self.log("test/mse", self.m_MSE(yhat, yref))
        return {"yref": yref, "yhat": yhat}

    def on_test_epoch_end(self):
        yhats = torch.cat(self.test_step_yhat, dim=0)
        yrefs = torch.cat(self.test_step_yref, dim=0)
        self.log("test/mse", self.m_MSE(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("test/psnr", self.m_PSNR(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("test/ssim", self.m_SSIM(yhats, yrefs), sync_dist=self.sync_dist)
        # visualization
        b_size = 3
        for idx in range(b_size): 
            plotSample(toNumpy(torch.squeeze(yhats[idx,:,:,:])), toNumpy(torch.squeeze(yrefs[idx,:,:,:])), self.cfg_model.save_dir, f"test_epoch_{self.current_epoch}_idx_{idx}")
        # free memory
        self.test_step_yhat.clear()
        self.test_step_yref.clear()

        np.save(self.cfg_model.save_dir+"/predictions.npy", toNumpy(yhats))
        np.save(self.cfg_model.save_dir+"/targets.npy", toNumpy(yrefs))

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]


#---------------------------------------------------------
# Model pytorch lightningmodule baseline arbitrary-scale
#---------------------------------------------------------
class ArbitraryScaleModule(pl.LightningModule):
    def __init__(self,
        normalizer,
        params_data: DictConfig,
        params_model: DictConfig,
        params_optim: DictConfig,
        params_scheduler: DictConfig,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.cfg_data      = params_data
        self.cfg_model     = params_model
        self.cfg_optim     = params_optim
        self.cfg_scheduler = params_scheduler

        self.model      = get_model(self.cfg_model)
        self.optimizer  = get_optimizer(list(self.model.parameters()), self.cfg_optim)
        self.scheduler  = get_scheduler(self.optimizer, self.cfg_scheduler)
        self.criterion  = get_loss(self.cfg_optim.loss)

        self.normalizer = normalizer
        self.sync_dist = torch.cuda.device_count() > 1
        self.validation_step_yhat = []
        self.validation_step_yref = []
        self.test_step_yhat = []
        self.test_step_yref = []
        self.viz_size = self.cfg_data.viz_size

        self.m_MSE = MeanSquaredError()
        self.m_PSNR = PeakSignalNoiseRatio()
        self.m_SSIM = StructuralSimilarityIndexMeasure()

    def step(self, batch: Any):
        img_low, coord, cell, yref = batch['img_low'], batch['coord'], batch['cell'], batch['img_high']
        yhat = self.model(img_low, coord, cell)
        if self.cfg_optim.loss == "FPLoss":
            loss = self.criterion(yhat, yref, img_low)
        else:
            loss = self.criterion(yhat, yref)
        return loss, yhat, yref

    def training_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log("train/mse", self.m_MSE(yhat, yref), sync_dist=self.sync_dist)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        _, yhat, yref = self.step(batch)
        yhat = yhat.view(-1, 1, self.viz_size[0], self.viz_size[1])
        yref = yref.view(-1, 1, self.viz_size[0], self.viz_size[1])
        self.validation_step_yhat.append(yhat)
        self.validation_step_yref.append(yref)
        return {"yref": yref, "yhat": yhat}

    def on_validation_epoch_end(self):
        yhats = torch.cat(self.validation_step_yhat, dim=0)
        yrefs = torch.cat(self.validation_step_yref, dim=0)
        self.log("validation/mse", self.m_MSE(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("validation/psnr", self.m_PSNR(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("validation/ssim", self.m_SSIM(yhats, yrefs), sync_dist=self.sync_dist)
        # visualization
        b_size = 3
        if (self.current_epoch+1)%50 == 0:
            for idx in range(b_size): 
                plotSample(toNumpy(torch.squeeze(yhats[idx,:,:,:])), toNumpy(torch.squeeze(yrefs[idx,:,:,:])), self.cfg_model.save_dir, f"val_epoch_{self.current_epoch}_idx_{idx}")
        # free memory
        self.validation_step_yhat.clear()
        self.validation_step_yref.clear()

    def test_step(self, batch: Any, batch_idx: int):
        _, yhat, yref = self.step(batch)
        yhat = yhat.view(-1, 1, self.viz_size[2], self.viz_size[3])
        yref = yref.view(-1, 1, self.viz_size[2], self.viz_size[3])
        self.test_step_yhat.append(yhat)
        self.test_step_yref.append(yref)
        self.log("test/mse", self.m_MSE(yhat, yref))
        return {"yref": yref, "yhat": yhat}

    def on_test_epoch_end(self):
        yhats = torch.cat(self.test_step_yhat, dim=0)
        yrefs = torch.cat(self.test_step_yref, dim=0)
        self.log("test/mse", self.m_MSE(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("test/psnr", self.m_PSNR(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("test/ssim", self.m_SSIM(yhats, yrefs), sync_dist=self.sync_dist)
        # visualization
        b_size = 3
        for idx in range(b_size): 
            plotSample(toNumpy(torch.squeeze(yhats[idx,:,:,:])), toNumpy(torch.squeeze(yrefs[idx,:,:,:])), self.cfg_model.save_dir, f"test_epoch_{self.current_epoch}_idx_{idx}")
        # free memory
        self.test_step_yhat.clear()
        self.test_step_yref.clear()

        np.save(self.cfg_model.save_dir+"/predictions.npy", toNumpy(yhats))
        np.save(self.cfg_model.save_dir+"/targets.npy", toNumpy(yrefs))

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]


class HiNOTEModule(pl.LightningModule):
    def __init__(self,
        normalizer,
        params_data: DictConfig,
        params_model: DictConfig,
        params_optim: DictConfig,
        params_scheduler: DictConfig,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.cfg_data      = params_data
        self.cfg_model     = params_model
        self.cfg_optim     = params_optim
        self.cfg_scheduler = params_scheduler

        self.model      = get_model(self.cfg_model)
        self.optimizer  = get_optimizer(list(self.model.parameters()), self.cfg_optim)
        self.scheduler  = get_scheduler(self.optimizer, self.cfg_scheduler)
        self.criterion  = get_loss(self.cfg_optim.loss)

        self.normalizer = normalizer
        self.sync_dist = torch.cuda.device_count() > 1
        self.validation_step_yhat = []
        self.validation_step_yref = []
        self.test_step_yhat = []
        self.test_step_yref = []
        self.viz_size = self.cfg_data.viz_size

        self.m_MSE = MeanSquaredError()
        self.m_PSNR = PeakSignalNoiseRatio()
        self.m_SSIM = StructuralSimilarityIndexMeasure()

    def step(self, batch: Any):
        """
        Args:
        (1) efficient implementation
            img_low (torch.tensor) - (b, c, h_l, w_l)
            coord (torch.tensor) - (b, n_points, 2)
            cell (torch.tensor) - (b, n_points, 2)
            yref (torch.tensor) - (b, n_points, c)
        (2) efficient implementation
            img_low (torch.tensor) - (b, c, h_l, w_l)
            coord (torch.tensor) - (b, h_h, w_h, 2)
            cell (torch.tensor) - (b, 2)
            yref (torch.tensor) - (b, c, h_h, w_h)
        Returns:
            loss (torch.tensor) - (1)
            yhat (torch.tensor) - (b, c, h_h, w_h)
            yref (torch.tensor) - (b, c, h_h, w_h)
        """
        img_low, coord, cell, yref = batch['img_low'], batch['coord'], batch['cell'], batch['img_high']
        yhat = self.model(img_low, coord, cell)
        if self.cfg_optim.loss == "FPLoss":
            loss = self.criterion(yhat, yref, img_low)
        else:
            loss = self.criterion(yhat, yref)
        return loss, yhat, yref

    def training_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        self.log("train/mse", self.m_MSE(yhat, yref), sync_dist=self.sync_dist)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, yhat, yref = self.step(batch)
        self.log("validation/loss", loss, on_step=False, on_epoch=True, sync_dist=self.sync_dist)
        yhat = yhat.view(-1, 1, self.viz_size[0], self.viz_size[1])
        yref = yref.view(-1, 1, self.viz_size[0], self.viz_size[1])
        self.validation_step_yhat.append(yhat)
        self.validation_step_yref.append(yref)
        return {"yref": yref, "yhat": yhat}

    def on_validation_epoch_end(self):
        yhats = torch.cat(self.validation_step_yhat, dim=0)
        yrefs = torch.cat(self.validation_step_yref, dim=0)
        self.log("validation/mse", self.m_MSE(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("validation/psnr", self.m_PSNR(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("validation/ssim", self.m_SSIM(yhats, yrefs), sync_dist=self.sync_dist)
        # visualization
        b_size = 3
        if (self.current_epoch+1)%50 == 0:
            for idx in range(b_size): 
                plotSample(toNumpy(torch.squeeze(yhats[idx,:,:,:])), toNumpy(torch.squeeze(yrefs[idx,:,:,:])), self.cfg_model.save_dir, f"val_epoch_{self.current_epoch}_idx_{idx}")
        # free memory
        self.validation_step_yhat.clear()
        self.validation_step_yref.clear()

    def test_step(self, batch: Any, batch_idx: int):
        _, yhat, yref = self.step(batch)
        yhat = yhat.view(-1, 1, self.viz_size[2], self.viz_size[3])
        yref = yref.view(-1, 1, self.viz_size[2], self.viz_size[3])
        self.test_step_yhat.append(yhat)
        self.test_step_yref.append(yref)
        self.log("test/mse", self.m_MSE(yhat, yref))
        return {"yref": yref, "yhat": yhat}

    def on_test_epoch_end(self):
        yhats = torch.cat(self.test_step_yhat, dim=0)
        yrefs = torch.cat(self.test_step_yref, dim=0)
        self.log("test/mse", self.m_MSE(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("test/psnr", self.m_PSNR(yhats, yrefs), sync_dist=self.sync_dist)
        self.log("test/ssim", self.m_SSIM(yhats, yrefs), sync_dist=self.sync_dist)
        # visualization
        b_size = 3
        for idx in range(b_size): 
            plotSample(toNumpy(torch.squeeze(yhats[idx,:,:,:])), toNumpy(torch.squeeze(yrefs[idx,:,:,:])), self.cfg_model.save_dir, f"test_epoch_{self.current_epoch}_idx_{idx}")
        # free memory
        self.test_step_yhat.clear()
        self.test_step_yref.clear()

        np.save(self.cfg_model.save_dir+"/predictions.npy", toNumpy(yhats))
        np.save(self.cfg_model.save_dir+"/targets.npy", toNumpy(yrefs))

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]