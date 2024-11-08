"""
WDSR: Wide Activation for Efficient and Accurate Image Super-Resolution
Ref: https://arxiv.org/abs/1808.08718
"""
import torch

from torch import nn
from torch.nn.utils import weight_norm


class ResBlock(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=1.0):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.module = nn.Sequential(
            weight_norm(nn.Conv2d(n_feats, n_feats * expansion_ratio, kernel_size=3, padding=1, padding_mode='circular')),
            nn.ReLU(inplace=True),
            weight_norm(nn.Conv2d(n_feats * expansion_ratio, n_feats, kernel_size=3, padding=1, padding_mode='circular'))
        )

    def forward(self, x):
        return x + self.module(x) * self.res_scale


class WDSR_net(nn.Module):
    def __init__(self, in_feats, out_feats, n_feats, n_res_blocks, upscale_factor):
        super(WDSR_net, self).__init__()
        
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.n_feats = n_feats
        self.expansion_ratio = 4
        self.res_scale = 0.1
        self.n_res_blocks = n_res_blocks
        self.scale = upscale_factor
        
        head = [weight_norm(nn.Conv2d(self.in_feats , self.n_feats, kernel_size=3, padding=1, padding_mode='circular'))]
        body = [ResBlock(self.n_feats, self.expansion_ratio, self.res_scale) for _ in range(self.n_res_blocks)]
        tail = [weight_norm(nn.Conv2d(self.n_feats, self.out_feats * (self.scale ** 2), kernel_size=3, padding=1, padding_mode='circular')),
                nn.PixelShuffle(self.scale)]
        skip = [weight_norm(nn.Conv2d(self.in_feats , self.out_feats * (self.scale ** 2), kernel_size=5, padding=2, padding_mode='circular')), 
                nn.PixelShuffle(self.scale)]

        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

        self.subtract_mean = True

    def forward(self, x):
        # input size: [N,C,H,W]
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        return x


if __name__ == "__main__":
    model = WDSR_net(in_feats=1, out_feats=1, n_feats=32, n_res_blocks=18, upscale_factor=4)

    input_x = torch.rand((16, 1, 32, 32))
    output_y = model(input_x)
    print(output_y.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {pytorch_total_params}")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {pytorch_total_params}")