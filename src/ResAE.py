import torch
import torch.nn as nn

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
        x += skip_x
        return self.activation(x)


class ResAE(nn.Module):
    def __init__(self, activation=nn.ReLU()):
        super(ResAE, self).__init__()
        self.activation = activation
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(128),
            self.activation,
            ResidualBlock_k3(128, activation),
            nn.Conv2d(128, 32, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(32),
            self.activation,
            ResidualBlock_k3(32, activation),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            self.activation,
        )

        self.decoder = nn.Sequential(
            ResidualBlock_k3(16, activation),
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(32),
            self.activation,
            ResidualBlock_k3(32, activation),
            nn.ConvTranspose2d(32, 128, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            self.activation,
            ResidualBlock_k3(128, activation),
            nn.ConvTranspose2d(128, 3, kernel_size=8, stride=2, padding=3, output_padding=0),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self, x, b_t=None):
        x = self.encoder(x)
        if self.training and b_t is not None:
            max_val = x.max() / (2 ** (b_t + 1))
            noise = torch.rand_like(x) * max_val
            x = x + noise
        x = self.decoder(x)
        return x
    
class BaseAutoEncoder(nn.Module):
    def __init__(self):
        super(BaseAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 128, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 3, kernel_size=7, stride=2, padding=3, output_padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x, b_t=None):
        x = self.encoder(x)
        if self.training and b_t is not None:
            max_val = x.max() / (2 ** (b_t + 1))
            noise = torch.rand_like(x) * max_val
            x = x + noise
        x = self.decoder(x)
        return x