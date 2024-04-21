import torch.nn as nn
class Generator(nn.Module):
    def __init__(self,input_size ,size_feature_map, image_channels = 3):
        super(Generator, self).__init__()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( input_size, size_feature_map * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(size_feature_map * 8),
            nn.ReLU(True),
            # state size. ``(size_feature_map*8) x 4 x 4``
            nn.ConvTranspose2d(size_feature_map * 8, size_feature_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_feature_map * 4),
            nn.ReLU(True),
            # state size. ``(size_feature_map*4) x 8 x 8``
            nn.ConvTranspose2d( size_feature_map * 4, size_feature_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_feature_map * 2),
            nn.ReLU(True),
            # state size. ``(size_feature_map*2) x 16 x 16``
            nn.ConvTranspose2d( size_feature_map * 2, size_feature_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(size_feature_map),
            nn.ReLU(True),
            # state size. ``(size_feature_map) x 32 x 32``
            nn.ConvTranspose2d( size_feature_map, image_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, input):
        return self.main(input)