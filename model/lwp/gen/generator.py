import torch
from torch import nn
from torch.nn import functional as F

class Generator(nn.Module):
    def __init__(self, z_size, image_size, out_c_size, g_size=128):
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.out_c_size = out_c_size
        self.g_size = g_size

        self.init_size = image_size // 4
        self.l1 = nn.Sequential(nn.Linear(z_size, g_size*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(g_size),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(g_size, g_size, 3, stride=1, padding=1),
            nn.BatchNorm2d(g_size, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(g_size, g_size//2, 3, stride=1, padding=1),
            nn.BatchNorm2d(g_size//2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(g_size//2, out_c_size, 3, stride=1, padding=1),
            # nn.Tanh(),
            # nn.BatchNorm2d(out_c_size, affine=False) 
        )

    def forward(self, z):
        if len(z.size()) > 2:
            z = z.view(z.shape[:2])
        out = self.l1(z)
        out = out.view(out.shape[0], self.g_size, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = F.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = F.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img

    def sample(self, size):
        
        # sample z
        z = torch.randn(size, self.z_size)
        z = z.cuda()
        X = self.forward(z)
        return X

def create_generator(z_size, g_channel_size, gen_output_size, out_c_size=128):
    generator = Generator(
        z_size=z_size, image_size=gen_output_size,
        out_c_size=out_c_size, g_size=g_channel_size)
    return generator