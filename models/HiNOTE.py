import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


non_act_d = {"ReLU": nn.ReLU(True),
             "LeakyReLU": nn.LeakyReLU(True),
             "ELU": nn.ELU(True),
             "SELU": nn.SELU(True),
             "GELU": nn.GELU(),
             "RReLU": nn.RReLU(True)}

def default_conv(in_channels, out_channels, kernel_size, bias=False):
    """
    Standard convolutional layer with padding to retain input shape.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the kernel.
        bias (bool): If True, adds a learnable bias.
    
    Returns:
        nn.Conv2d: Configured convolutional layer.
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class BasicBlock(nn.Module):
    """
    A basic convolutional block with BatchNorm, ReLU activation, and 2D convolution.

    Args:
        in_planes (int): Number of input feature planes.
        out_planes (int): Number of output feature planes.
    """
    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    """
    Dense block with multiple layers, where outputs are concatenated for deeper connections.

    Args:
        n_layers (int): Number of layers in the block.
        in_planes (int): Number of input channels.
        growth_rate (int): Growth rate for increasing channels at each layer.
        block (nn.Module): Block type to use for each layer.
    """
    def __init__(self, n_layers, in_planes, growth_rate, block):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, n_layers)

    def _make_layer(self, block, in_planes, growth_rate, n_layers):
        layers = []
        for i in range(n_layers):
            layers.append(block(in_planes + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class freup_Areadinterpolation(nn.Module):
    """
    Frequency domain upsampling with area-based interpolation.

    Args:
        channels (int): Number of channels in the input.
    """
    def __init__(self, channels):
        super(freup_Areadinterpolation, self).__init__()

        self.amp_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0), 
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0), 
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        N, C, H, W = x.shape

        # Perform FFT to obtain magnitude and phase
        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        # Apply amplitude and phase fusions
        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        # Upsample the fused amplitude and phase
        amp_fuse = Mag.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        pha_fuse = Pha.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

        # Construct complex signal and perform inverse FFT
        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        # Crop and interpolate to double size
        crop = torch.zeros_like(x)
        crop[:, :, 0:int(H/2), 0:int(W/2)] = output[:, :, 0:int(H/2), 0:int(W/2)]
        crop[:, :, int(H/2):H, 0:int(W/2)] = output[:, :, int(H*1.5):2*H, 0:int(W/2)]
        crop[:, :, 0:int(H/2), int(W/2):W] = output[:, :, 0:int(H/2), int(W*1.5):2*W]
        crop[:, :, int(H/2):H, int(W/2):W] = output[:, :, int(H*1.5):2*H, int(W*1.5):2*W]
        crop = F.interpolate(crop, (2 * H, 2 * W))

        return self.post(crop)

class freup_Periodicpadding(nn.Module):
    """
    Frequency domain upsampling with periodic padding.

    Args:
        channels (int): Number of channels in the input.
    """
    def __init__(self, channels):
        super(freup_Periodicpadding, self).__init__()
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.post = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        N, C, H, W = x.shape

        # Apply FFT to get magnitude and phase
        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        # Apply amplitude and phase fusions
        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        # Tile the amplitude and phase to double the spatial size
        amp_fuse = torch.tile(Mag, (2, 2))
        pha_fuse = torch.tile(Pha, (2, 2))

        # Construct complex result and apply inverse FFT
        real = amp_fuse * torch.cos(pha_fuse)
        imag = amp_fuse * torch.sin(pha_fuse)
        out = torch.complex(real, imag)
        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return self.post(output)

class freup_Cornerdinterpolation(nn.Module):
    """
    Frequency domain upsampling using corner-based interpolation.

    Args:
        channels (int): Number of channels in the input.
    """
    def __init__(self, channels):
        super(freup_Cornerdinterpolation, self).__init__()
        self.amp_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )
        self.pha_fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, 1, 0), nn.LeakyReLU(0.1, inplace=False),
            nn.Conv2d(channels, channels, 1, 1, 0)
        )

    def forward(self, x):
        N, C, H, W = x.shape

        # Perform FFT and extract magnitude and phase
        fft_x = torch.fft.fft2(x)
        mag_x = torch.abs(fft_x)
        pha_x = torch.angle(fft_x)

        Mag = self.amp_fuse(mag_x)
        Pha = self.pha_fuse(pha_x)

        r, c = H, W
        # I_Mup = torch.zeros((N, C, 2 * H, 2 * W)).cuda()
        # I_Pup = torch.zeros((N, C, 2 * H, 2 * W)).cuda()
        I_Mup = torch.zeros((N, C, 2 * H, 2 * W))
        I_Pup = torch.zeros((N, C, 2 * H, 2 * W))

        # Define indexing for odd/even cases
        ir1, ir2 = (r // 2 + 1, r // 2 + 1) if r % 2 else (r // 2 + 1, r // 2)
        ic1, ic2 = (c // 2 + 1, c // 2 + 1) if c % 2 else (c // 2 + 1, c // 2)

        # Populate interpolated magnitudes and phases
        I_Mup[:, :, :ir1, :ic1] = Mag[:, :, :ir1, :ic1]
        I_Mup[:, :, :ir1, ic2 + c:] = Mag[:, :, :ir1, ic2:]
        I_Mup[:, :, ir2 + r:, :ic1] = Mag[:, :, ir2:, :ic1]
        I_Mup[:, :, ir2 + r:, ic2 + c:] = Mag[:, :, ir2:, ic2:]

        # Handle boundary conditions
        if r % 2 == 0:
            I_Mup[:, :, ir2, :] *= 0.5
            I_Mup[:, :, ir2 + r, :] *= 0.5
        if c % 2 == 0:
            I_Mup[:, :, :, ic2] *= 0.5
            I_Mup[:, :, :, ic2 + c] *= 0.5

        # Populate phases
        I_Pup[:, :, :ir1, :ic1] = Pha[:, :, :ir1, :ic1]
        I_Pup[:, :, :ir1, ic2 + c:] = Pha[:, :, :ir1, ic2:]
        I_Pup[:, :, ir2 + r:, :ic1] = Pha[:, :, ir2:, :ic1]
        I_Pup[:, :, ir2 + r:, ic2 + c:] = Pha[:, :, ir2:, ic2:]

        # Apply boundary conditions to phases
        if r % 2 == 0:
            I_Pup[:, :, ir2, :] *= 0.5
            I_Pup[:, :, ir2 + r, :] *= 0.5
        if c % 2 == 0:
            I_Pup[:, :, :, ic2] *= 0.5
            I_Pup[:, :, :, ic2 + c] *= 0.5

        # Construct complex FFT and apply inverse FFT
        real = I_Mup * torch.cos(I_Pup)
        imag = I_Mup * torch.sin(I_Pup)
        out = torch.complex(real, imag)
        output = torch.fft.ifft2(out)
        output = torch.abs(output)

        return output

class fresadd(nn.Module):
    """
    Additive frequency upsampling module.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        method (str): Interpolation method to use.
        act (nn.Module): Activation function to apply.
    """
    def __init__(self, in_channels=32, out_channels=16, method="Corner", act=None):
        super(fresadd, self).__init__()

        # Define frequency upsampling method
        if method == "Area":
            self.Fup = freup_Areadinterpolation(in_channels)
        elif method == "Periodic":
            self.Fup = freup_Periodicpadding(in_channels)
        elif method == "Corner":
            self.Fup = freup_Cornerdinterpolation(in_channels)
        else:
            raise NotImplementedError("Method not implemented")        

        self.fuse = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.act = act

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x3 = self.Fup(x1)
        xm = x2 + x3
        xn = self.fuse(xm)
        return self.act(xn)

class frescat(nn.Module):
    """
    Concatenative frequency upsampling module.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        method (str): Interpolation method to use.
        act (nn.Module): Activation function to apply.
    """
    def __init__(self, in_channels=32, out_channels=16, method="Corner", act=None):
        super(frescat, self).__init__()
        
        # Define frequency upsampling method
        if method == "Area":
            self.Fup = freup_Areadinterpolation(in_channels)
        elif method == "Periodic":
            self.Fup = freup_Periodicpadding(in_channels)
        elif method == "Corner":
            self.Fup = freup_Cornerdinterpolation(in_channels)
        else:
            raise NotImplementedError("Method not implemented")   
        
        self.fuse = nn.Conv2d(2 * in_channels, out_channels, 1, 1, 0, bias=False)
        self.act = act

    def forward(self, x):
        x1 = x
        x2 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x3 = self.Fup(x1)
        xn = self.fuse(torch.cat([x2, x3], dim=1))
        return self.act(xn)

def make_coord(shape, ranges=None, flatten=True):
    """
    Generates coordinates at grid centers for a given shape, optionally within specified ranges.

    Args:
        shape (tuple): The shape (height, width, etc.) of the grid to create coordinates for.
        ranges (tuple of tuples, optional): Ranges for each dimension in the form ((min1, max1), (min2, max2), ...). 
                                            If None, defaults to (-1, 1) for each dimension.
        flatten (bool, optional): If True, flattens the coordinates into a single dimension. Default is True.

    Returns:
        torch.Tensor: A tensor of coordinates with shape (prod(shape), len(shape)) if flattened, 
                      otherwise of shape (*shape, len(shape)).
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
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class LIIFEnsemble(nn.Module):
    """
    An ensemble upsampling layer using the Local Implicit Image Function (LIIF)-based structure.

    Args:
        up_method (str): The upsampling method to use in grid sampling, such as "bilinear" or "nearest".
    """
    def __init__(self, up_method):
        super(LIIFEnsemble, self).__init__()
        self.up_method = up_method

    def forward(self, x, coord, cell):
        """
        Forward pass for ensemble upsampling.

        Args:
            x (torch.Tensor): Input feature tensor with shape (batch, channels, height, width).
            coord (torch.Tensor): Coordinates for upsampling, with shape (batch, h_out, w_out, 2).
            cell (torch.Tensor): Cell size information for relative scaling of coordinates.

        Returns:
            torch.Tensor: The upsampled feature tensor with additional spatial and feature information.
        """
        feat = x
        # pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
        #     .permute(2, 0, 1) \
        #     .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])
        pos_lr = make_coord(feat.shape[-2:], flatten=False) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                feat_ = F.grid_sample(feat, coord_.flip(-1), mode=self.up_method, align_corners=False)

                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode=self.up_method, align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)
                
        rel_cell = cell.clone()
        rel_cell[:, 0] *= feat.shape[-2]
        rel_cell[:, 1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)

        feat_ensemble = torch.cat([*rel_coords, *feat_s, 
                                   rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, coord.shape[1], coord.shape[2])], dim=1)
        
        return feat_ensemble

class LayerNorm(nn.Module):
    """
    Custom Layer Normalization layer.

    Args:
        d_model (int): The dimensionality of the input features.
        eps (float, optional): Small constant to prevent division by zero. Default is 1e-5.
    """
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        """
        Forward pass for layer normalization.

        Args:
            x (torch.Tensor): Input tensor with shape (..., d_model).

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        out = (x - mean) / (std + self.eps)
        out = self.weight * out + self.bias
        return out
    
class Galerkin_attn(nn.Module):
    """
    Galerkin multi-head attention layer.

    Args:
        midc (int): Number of intermediate channels.
        heads (int): Number of attention heads.
        act (nn.Module): Activation function to use.
    """
    def __init__(self, midc, heads, act):
        super(Galerkin_attn, self).__init__()

        self.headc = midc // heads
        self.heads = heads
        self.midc = midc

        self.qkv_proj = nn.Conv2d(midc, 3 * midc, 1, bias=False)
        self.o_proj1 = nn.Conv2d(midc, midc, 1, bias=False)
        self.o_proj2 = nn.Conv2d(midc, midc, 1, bias=False)

        self.kln = LayerNorm((self.heads, 1, self.headc))
        self.vln = LayerNorm((self.heads, 1, self.headc))

        self.act1 = act
        self.act2 = act

    def forward(self, x):
        """
        Forward pass for the attention layer.

        Args:
            x (torch.Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with the same shape as input after applying attention.
        """
        B, C, H, W = x.shape
        bias = x
        qkv = self.qkv_proj(x).permute(0, 2, 3, 1).reshape(B, H * W, self.heads, 3 * self.headc)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)

        k = self.kln(k)
        v = self.vln(v)
        v = torch.matmul(k.transpose(-2, -1), v) / (H * W)
        v = torch.matmul(q, v)

        v = v.permute(0, 2, 1, 3).reshape(B, H, W, C)
        ret = v.permute(0, 3, 1, 2) + bias
        bias = self.o_proj2(self.act1(self.o_proj1(ret))) + bias
        return self.act2(bias)

def sinc_filter(cutoff_freq, filter_size):
    """
    Constructs a 2D low-pass sinc filter for attenuating high-frequency components.
    The cutoff frequency should be less than or equal to 0.5 (Nyquist frequency).

    Args:
    - cutoff_freq: Cutoff frequency (float) as a fraction of the Nyquist frequency (0.5).
    - filter_size: The size of the filter kernel (should be an odd number).

    Returns:
    - A 2D tensor representing the sinc-based filter kernel.
    """
    t = torch.linspace(-filter_size // 2, filter_size // 2, filter_size)
    x, y = torch.meshgrid(t, t, indexing='ij')
    r = torch.sqrt(x**2 + y**2) + 1e-10
    filter = torch.sin(2 * np.pi * cutoff_freq * r) / (np.pi * r)
    filter /= torch.sum(filter)

    return filter

class AliasFreeActivation(torch.nn.Module):
    def __init__(self, activation_fn, upsample_factor=2, filter_size=31):
        """
        Args:
        - activation_fn: Non-linear activation function to apply after upsampling.
        - upsample_factor: Factor by which to upsample before applying the activation.
        - filter_size: Size of the sinc filter kernel (should be odd).
        """
        super(AliasFreeActivation, self).__init__()
        self.activation_fn = activation_fn
        self.upsample_factor = upsample_factor
        self.filter_size = filter_size
        self.cutoff_freq = 0.5 / upsample_factor
        self.register_buffer("sinc_filter", sinc_filter(self.cutoff_freq, filter_size).view(1, 1, filter_size, filter_size))

    def forward(self, x):
        # Step 1: Upsample the input to increase frequency resolution
        x_upsampled = F.interpolate(x, scale_factor=self.upsample_factor, mode='bilinear', align_corners=True)

        # Step 2: Apply the non-linear activation function
        x_activated = self.activation_fn(x_upsampled)

        # Step 3: Apply the sinc-based low-pass filter to attenuate high-frequency components
        sinc_filter_expanded = self.sinc_filter.expand(x_activated.shape[1], 1, -1, -1)
        x_filtered = F.conv2d(x_activated, sinc_filter_expanded, padding=self.filter_size // 2, groups=x_activated.shape[1])

        # Step 4: Downsample to the original resolution
        x_downsampled = F.interpolate(x_filtered, size=x.shape[-2:], mode='bilinear', align_corners=True)

        return x_downsampled

class HierarchicalDecoder(nn.Module):
    """
    Optimized Hierarchical Decoder with multi-resolution processing, using a unique kernel per level.
    Upsampling includes AliasFreeActivation, upsampling, concatenation, and integration.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        levels (int): Number of hierarchical levels.
        activation_fn (nn.Module): Non-linear activation function to use.
        kernel_cls (callable): Callable that returns a new kernel instance for each level.
    """
    def __init__(self, in_channels, out_channels, levels=3, activation_fn=nn.GELU(), kernel_cls=None):
        super(HierarchicalDecoder, self).__init__()

        # Downsampling layers
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                AliasFreeActivation(activation_fn=activation_fn),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)
            ) for _ in range(levels - 1)
        ])
        
        # Unique kernel for each level
        self.kernels = nn.ModuleList([kernel_cls() for _ in range(levels)])

        # Upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                AliasFreeActivation(activation_fn=activation_fn),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
            ) for _ in range(levels - 1)
        ])

        # Integration layers
        self.integrate_layers = nn.ModuleList([nn.Conv2d(in_channels*2, in_channels, kernel_size=3, padding=1) for _ in range(levels - 1)])

        # Final projection to reduce the integrated features to the output channels
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass for hierarchical decoder with multi-resolution integration.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch, in_channels, height, width).
        
        Returns:
            torch.Tensor: Output tensor with the same spatial dimensions as input.
        """
        # Downsampling to create hierarchical levels
        features = [x]
        for down in self.downsample_layers:
            x = down(x)
            features.append(x)

        # Apply kernel at each resolution level
        processed_features = [self.kernels[i](feat) for i, feat in enumerate(features)]

        # Reverse processed_features for hierarchical upsampling integration
        processed_features = processed_features[::-1]

        # Start integration with the smallest (last) feature
        x = processed_features[0]

        # Upsampling and integration
        for i in range(len(self.upsample_layers)):
            x_low = self.upsample_layers[i](x)
            x_high = processed_features[i + 1]
            x = self.integrate_layers[i](torch.cat((x_high, x_low), dim=1))

        # Final projection to match output channels
        output = self.project(x)

        return output

class HiNOTE_net(nn.Module):
    def __init__(self, feature_up_ratio=4, fourier_up="Corner", feature_combine="add", 
                 non_act="GELU", liif_up_method="nearest", attention_width=128, 
                 attention_head=4, hierarchical_levels=1):
        super(HiNOTE_net, self).__init__()

        # Encoder network
        self.encoder = nn.Sequential()
        if feature_combine == "add":
            for idx in range(int(math.log(feature_up_ratio, 2))):
                self.encoder.add_module(
                    f"featUp{idx}", 
                    fresadd(in_channels=1, out_channels=1, method=fourier_up, act=non_act_d[non_act])
                )
        elif feature_combine == "cat":
            for idx in range(int(math.log(feature_up_ratio, 2))):
                self.encoder.add_module(
                    f"featUp{idx}", 
                    frescat(in_channels=1, out_channels=1, method=fourier_up, act=non_act_d[non_act])
                )
        else:
            raise NotImplementedError("Unsupported feature combination method specified.")

        # LIIF ensemble layer for dynamic upsampling based on input coordinates
        self.sampler = LIIFEnsemble(up_method=liif_up_method)

        # Feature lifting layer
        self.lift = default_conv(in_channels=14, out_channels=attention_width, kernel_size=5)

        # Hierarchical Decoder with unique kernels at each level
        self.hierarchical_decoder = HierarchicalDecoder(
            in_channels=attention_width,
            out_channels=1,
            levels=hierarchical_levels,
            activation_fn=non_act_d[non_act],
            kernel_cls=lambda: Galerkin_attn(midc=attention_width, heads=attention_head, act=non_act_d[non_act])
        )

    def forward(self, x, coord, cell):
        # Encode and sample
        y = self.encoder(x)
        y = self.sampler(y, coord, cell)
        
        # Lift and decode hierarchically
        y = self.lift(y)
        y = self.hierarchical_decoder(y)
        
        # Align features with original coordinates
        y += F.grid_sample(x, coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
        
        return y

if __name__ == "__main__":
    model = HiNOTE_net()

    input_x = torch.rand((16, 1, 32, 32))
    coord = torch.rand((16, 128, 128, 2))    
    cell = torch.tensor([2 / 128, 2 / 128], dtype=torch.float32).repeat(16,1)
    output_y = model(input_x, coord, cell)
    print(output_y.shape)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {pytorch_total_params}")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {pytorch_total_params}")