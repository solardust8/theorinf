import torch
import torch.nn as nn

class Conv_Layer_k3_hwdiv2(nn.Module):
    def __init__(self, in_channels, out_channels, activation = nn.ReLU(), norm=False):
        super(Conv_Layer_k3_hwdiv2, self).__init__()
        self.normalize = norm
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.batch_norm(x)
        x = self.activation(x)
        return x

class Conv_Layer_k7_norm(nn.Module):
    def __init__(self, in_channels, out_channels, activation = nn.ReLU()):
        super(Conv_Layer_k7_norm, self).__init__()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

class TConv_Layer_k3_hwx2(nn.Module):
    def __init__(self, in_channels, out_channels, activation = nn.ReLU()):
        super(TConv_Layer_k3_hwx2, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            activation
        )

    def forward(self, x):
        x = self.layer(x)
        return x
    
class TConv_Layer_k7_hwinit(nn.Module):
    def __init__(self, in_channels, out_channels, activation = nn.ReLU()):
        super(TConv_Layer_k7_hwinit, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, output_padding=0, bias=False),
            activation
        )

    def forward(self, x):
        x = self.layer(x)
        return x
    
class ResidualBlock_k3(nn.Module):
    def __init__(self, channels, activation = nn.ReLU()):
        super(ResidualBlock_k3, self).__init__()
        self.activation = activation
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        skip_x = x
        x = self.block(x)
        return self.activation(x)
    
class ResidualAE(nn.Module):
    def __init__(self, in_channels=128, hid_channels=16, hw_start = 128, hw_end = 16, activation=nn.ReLU()):
        super(ResidualAE, self).__init__()
        
        assert ((in_channels & (in_channels-1) == 0) and in_channels != 0)
        assert ((hid_channels & (hid_channels-1) == 0) and hid_channels != 0)
        assert (hid_channels < in_channels)
        
        assert ((hw_start & (hw_start-1) == 0) and hw_start != 0)
        assert ((hw_end & (hw_end-1) == 0) and hw_end != 0)
        assert (hw_end < hw_start)
        
        self.enc_layers = self._initialize_enc_layers(in_channels, 
                                                      hid_channels, 
                                                      hw_start, 
                                                      hw_end, 
                                                      activation)
        self.dec_layers = self._initialize_dec_layers(in_channels, 
                                                      hid_channels,
                                                      hw_start, 
                                                      hw_end, 
                                                      activation)
        self.Encoder = nn.Sequential(*self.enc_layers)
        self.Decoder = nn.Sequential(*self.dec_layers)
        
    def _initialize_enc_layers(self, 
                               in_channels, 
                               hid_channels, 
                               hw_start, 
                               hw_end, 
                               activation):
        modules = []
        modules.append(Conv_Layer_k7_norm(in_channels=3, out_channels=in_channels, activation=activation))
        cur_channels = in_channels
        cur_hw = hw_start
        while cur_channels != hid_channels and cur_hw != hw_end:
            modules.append(ResidualBlock_k3(cur_channels, activation))
            modules.append(Conv_Layer_k3_hwdiv2(cur_channels, cur_channels//2, activation, norm=True))
            cur_channels //= 2
            cur_hw //= 2
        return modules
    
    def _initialize_dec_layers(self, 
                               in_channels, 
                               hid_channels, 
                               hw_start, 
                               hw_end, 
                               activation):
        modules = []
        cur_channels = hid_channels
        cur_hw = hw_end
        while cur_channels != in_channels and cur_hw != hw_start:
            modules.append(TConv_Layer_k3_hwx2(cur_channels, cur_channels*2, activation))
            modules.append(ResidualBlock_k3(cur_channels*2, activation))
            cur_channels *= 2
            cur_hw *= 2
        modules.append(TConv_Layer_k7_hwinit(in_channels=cur_channels, out_channels=3, activation=activation))
        return modules
    
    def forward(self, x, b_t=None):
        x = self.Encoder(x)
        if self.training and b_t is not None:
            max_val = x.max() / (2 ** (b_t + 1))
            noise = torch.rand_like(x) * max_val
            x = x + noise
        x = self.Decoder(x)
        return x