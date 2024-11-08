"""
SRCNN: Image Super-Resolution Using Deep Convolutional Networks
Ref: https://arxiv.org/pdf/1501.00092v3.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F 


class SRCNN_net(nn.Module):
    """
    Parameters
    ----------
    upscale_factor : int
        Super-Resolution scale factor. Determines Low-Resolution downsampling.

    """
    def __init__(self, in_feats, upscale_factor):

        super(SRCNN_net, self).__init__()

        self.upsacle_factor = upscale_factor
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_feats, out_channels=64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=in_feats, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input Low-Resolution image as tensor

        Returns
        -------
        torch.Tensor
            Super-Resolved image as tensor

        """
        # CNN extracting features
        # bicubic upsampling
        x = F.interpolate(x, scale_factor=[self.upsacle_factor,self.upsacle_factor], 
                                      mode='bicubic', align_corners=True)

        x = self.model(x)
        return x


if __name__ == "__main__":
    model = SRCNN_net(in_feats=1, upscale_factor=4)

    input_x = torch.rand((16, 1, 32, 32))
    output_y = model(input_x)
    print(output_y.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {pytorch_total_params}")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {pytorch_total_params}")