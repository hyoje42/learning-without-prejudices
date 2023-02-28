from typing import Tuple, Union
import torch
from torch import Tensor, nn, autograd, optim
from torch.nn import functional as F

from .generator import create_generator


EPSILON = 1e-8

class Critic(nn.Module):
    def __init__(self, image_size, image_channel_size, channel_size):
        """
        Args:
            image_size: size of image(feature map)
            image_channel_size: channel size of image(feature map)
            channel_size: channel size of critic network
        """
        # configurations
        super().__init__()
        self.image_size = image_size
        self.image_channel_size = image_channel_size
        self.channel_size = channel_size

        # layers
        self.conv1 = nn.Conv2d(
            image_channel_size, channel_size,
            kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            channel_size, channel_size*2,
            kernel_size=4, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            channel_size*2, channel_size*4,
            kernel_size=4, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            channel_size*4, channel_size*8,
            kernel_size=4, stride=2, padding=1,
        )
        
        self.fc = nn.Linear((image_size//16)**2 * channel_size*8, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = torch.flatten(x, 1)
        return self.fc(x)

class WGAN(nn.Module):
    def __init__(self, z_size,
                 c_channel_size, g_channel_size,
                 gen_output_size, gen_out_c_size):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.c_channel_size = c_channel_size
        self.g_channel_size = g_channel_size

        self.num_update_critic = 5
        self.lamda = 10.0

        # output size of generator
        self.gen_output_size = gen_output_size
        self.gen_out_c_size = gen_out_c_size

        self._build_generator()
        self._build_critic()
        self._set_optimizer()

        self.encode = None # Encoder

    def _build_generator(self):
        self.generator = create_generator(
            self.z_size, self.g_channel_size, 
            self.gen_output_size, self.gen_out_c_size)

    def _build_critic(self):
        self.critic = Critic(
            image_size=self.gen_output_size, 
            image_channel_size=self.gen_out_c_size,
            channel_size=self.c_channel_size)

    def _set_optimizer(self):
        self.optimizer_gen    = optim.Adam(self.generator.parameters(), lr=5e-5, betas=(0.5, 0.999), weight_decay=5e-4)
        self.optimizer_critic = optim.Adam(self.critic.parameters(),    lr=2e-4, betas=(0.5, 0.999), weight_decay=5e-4)

    def train_a_batch(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        """
        self.train()
        ## Update Discriminator
        for _ in range(self.num_update_critic):
            generated_x = self.sample(x.size(0))
            loss_critic, feat_fake = self._c_loss(x, generated_x, return_fake=True)
            loss_critic_gp = loss_critic + self._gradient_penalty(x, feat_fake, self.lamda)
            self.optimizer_critic.zero_grad()
            loss_critic_gp.backward()
            self.optimizer_critic.step()

        ## Update Generator
        generated_x = self.sample(x.size(0))
        loss_gen = self._g_loss(generated_x)
        self.optimizer_gen.zero_grad()
        loss_gen.backward()
        self.optimizer_gen.step()

        return loss_critic, loss_gen

    def sample(self, size) -> Tensor:
        return self.generator(self._get_noise(size))

    def _get_noise(self, size) -> Tensor:
        device = next(self.parameters()).device
        return torch.randn(size, self.z_size, 1, 1, device=device)

    def _c_loss(self, x: Tensor, fake: Tensor, return_fake=False) -> Union[Tuple[Tensor], Tensor]:
        assert x.size() == fake.size(), print(x.size(), fake.size())
        c_x = self.critic(x).mean()
        c_g = self.critic(fake).mean()
        l = -(c_x-c_g)
        return (l, fake) if return_fake else l

    def _g_loss(self, fake, return_fake=False) -> Union[Tuple[Tensor], Tensor]:
        loss = -self.critic(fake).mean()
        return (loss, fake) if return_fake else loss

    def _gradient_penalty(self, x: Tensor, g: Tensor, lamda) -> Tensor:
        assert x.size() == g.size(), print(x.size(), g.size())
        a = torch.rand(x.size(0), 1)
        a = a.cuda() if self._is_on_cuda() else a
        a = a\
            .expand(x.size(0), x.nelement()//x.size(0))\
            .contiguous()\
            .view(x.size())
        interpolated = (a*x.data + (1-a)*g.data).requires_grad_(True)
        c = self.critic(interpolated)
        gradients = autograd.grad(
            c, interpolated, grad_outputs=(
                torch.ones(c.size()).cuda() if self._is_on_cuda() else
                torch.ones(c.size())
            ),
            create_graph=True,
            retain_graph=True,
        )[0]
        return lamda * ((1-(gradients+EPSILON).norm(2, dim=1))**2).mean()

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda

    def print_net(self):
        def _count_parameters(*models: nn.Module):
            num = 0
            for model in models:
                num += sum(p.numel() for p in model.parameters() if p.requires_grad)
            msg = f"{num/(1000000):.4f} M"
            return msg
        msg = (
            f"[PARAMETER:G] {_count_parameters(self.generator)} " + 
            f"[PARAMETER:Critic] {_count_parameters(self.critic)}"
        )
        return msg

