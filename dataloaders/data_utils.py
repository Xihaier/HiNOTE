import torch
import numpy as np


class ZscoreStandardizer(object):
    """  
    Normalization transformation
    if reduce_dim = [0]: The mean is computed over different time volumes.
    if reduce_dim = []:  The mean is computed over all data points.
    """  
    def __init__(self, x, reduce_dim=[0]):
        self.mean = torch.mean(x, reduce_dim, keepdim=True).squeeze()
        self.std  = torch.std(x, reduce_dim, keepdim=True).squeeze()
        self.epsilon = 1e-10
        assert self.mean.shape == self.std.shape

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.epsilon)

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.epsilon) + self.mean


class MinMaxStandardizer(object):
    """  
    Min-Max transformation
    if reduce_dim = [0]: The min/max is computed over different time volumes.
    if reduce_dim = []:  The min/max is computed over all data points.
    """  
    def __init__(self, x, reduce_dim=[0]):
        if reduce_dim:
            self.minVal = torch.min(x, reduce_dim[0], keepdim=True)[0].squeeze()
            self.maxVal = torch.max(x, reduce_dim[0], keepdim=True)[0].squeeze()
        else:
            self.minVal = torch.min(x)
            self.maxVal = torch.max(x)

        self.epsilon = 1e-10

        assert self.minVal.shape == self.maxVal.shape

    def do(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.minVal) / (self.maxVal - self.minVal) + self.epsilon

    def undo(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.epsilon) * (self.maxVal - self.minVal) + self.minVal


import torchvision.transforms.functional as TF

def rotate_and_flip_images(images, rotation_angle=90, horizontal_flip=True, vertical_flip=False):
    """
    Rotates and flips a batch of images.

    Args:
    images (torch.Tensor): Tensor of images of shape (b, c, h, w).
    rotation_angle (int): Angle of rotation in degrees.
    horizontal_flip (bool): If True, apply horizontal flip.
    vertical_flip (bool): If True, apply vertical flip.

    Returns:
    torch.Tensor: The batch of transformed images.
    """
    transformed_images = []

    for img in images:
        # Apply rotation
        img = TF.rotate(img, rotation_angle)

        # Apply horizontal flip
        if horizontal_flip:
            img = TF.hflip(img)

        # Apply vertical flip
        if vertical_flip:
            img = TF.vflip(img)

        transformed_images.append(img)

    # Stack the images back into a batch
    return torch.stack(transformed_images)


def get_Data(data_dir, n_train_val_test, preprocessing, reduce_dim):
    """
    Args:
        data_dir (string): dataset file path
        n_train_val_test (list): number of training, validation, and test data points 
        preprocessing (string): data preprocessing method
        reduce_dim (int): data preprocessing reduced dimension
    Returns:
        data_train, data_val, data_test, normalizer
    """
    # load data: (n, c, h, w)
    dat = torch.unsqueeze(torch.from_numpy(np.load(data_dir)).to(torch.float32), 1)

    # pre-process data
    if preprocessing == "zscore":
        normalizer = ZscoreStandardizer(dat, reduce_dim)       
    elif preprocessing == "minmax":
        normalizer = MinMaxStandardizer(dat, reduce_dim)
    dat = normalizer.do(dat)

    # training, validation, and test data points 
    n_train, n_val, _ = n_train_val_test
    indices   = torch.randperm(sum(n_train_val_test))

    # # this is for making plots generate the last 100 for visulization
    # indices   = [num for num in range(sum(n_train_val_test))]

    idx_train = indices[:n_train]
    idx_val   = indices[n_train:(n_train+n_val)]
    idx_test  = indices[(n_train+n_val):]
    data_train = dat[idx_train,:,:,:]
    data_val   = dat[idx_val,:,:,:]
    data_test  = dat[idx_test,:,:,:]
    del dat

    # # data augmentation
    # data_train1 = rotate_and_flip_images(data_train, rotation_angle=90, horizontal_flip=True, vertical_flip=False)
    # data_train2 = rotate_and_flip_images(data_train, rotation_angle=180, horizontal_flip=False, vertical_flip=True)
    # data_train3 = rotate_and_flip_images(data_train, rotation_angle=270, horizontal_flip=True, vertical_flip=True)
    # data_train = torch.cat((data_train, data_train1, data_train2, data_train3), dim=0)

    return data_train, data_val, data_test, normalizer