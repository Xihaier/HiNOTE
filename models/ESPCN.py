"""
Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
Ref: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf
"""
import torch
import torch.nn as nn
import torch.nn.init as init


class ESPCN_net(nn.Module):
    def __init__(self, in_feats=1, upscale_factor=4, width=1):
        super(ESPCN_net, self).__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_feats, 128*width, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(128*width, 128*width, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(128*width, 64*width, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(64*width, in_feats * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)
    

if __name__ == "__main__":
    model = ESPCN_net(in_feats=1, upscale_factor=4, width=1)

    input_x = torch.rand((16, 1, 32, 32))
    output_y = model(input_x)
    print(output_y.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {pytorch_total_params}")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {pytorch_total_params}")