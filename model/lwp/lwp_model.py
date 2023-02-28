"""
Learning without Prejudices
"""
from typing import Dict, List, Tuple
from tqdm import tqdm

import torch
from torch import Tensor, nn

from model import SimpleConvNet
from model.lwp import WGAN, BYOL
from model.buffer import Buffer
from option import Config

class LwP_Base(nn.Module):
    def __init__(self, option: Config, classifier: SimpleConvNet) -> None:
        super().__init__()
        self.option = option
        self.classifier = classifier
        self.num_classes = option.num_class
        self.z_size = option.z_size
        self.d_channel = option.d_channel
        self.g_channel = option.g_channel
        self.gen_output_size = option.gen_output_size
        self.gen_out_c_size = option.gen_out_c_size
        self.layer_idx = option.layer_idx
        
        self.replay = option.replay

        self.device = torch.device('cpu')
        if self.replay:
            self.buffer = Buffer(buffer_size=200, device=self.device)

        ## Define Modules
        self._build_ssl()
        self._build_generator()

        ## Define Losses and Optimizers
        self._set_loss_and_optimizer()

    def _build_ssl(self):
        # net for SSL
        self.netS = BYOL(self.option, self.classifier)

    def _build_generator(self):
        self.netG = WGAN(
            z_size=self.z_size,
            c_channel_size=self.d_channel, g_channel_size=self.g_channel,
            gen_output_size=self.gen_output_size, gen_out_c_size=self.gen_out_c_size
        )
    
    def _set_loss_and_optimizer(self):
        # loss
        self.loss = nn.CrossEntropyLoss()
        # optimizers
        self.optim = self.netS.optim_ssl

    def train_one_epoch(self, data_loader) -> Tuple[Dict[str, float], str]:
        pass

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)

    def forward_encode(self, x: Tensor) -> Tensor:
        return self.netS.forward_encode(x)

    def sample(self, size):
        return self.netG.sample(size)

    def cuda(self, device=None):
        res = super().cuda(device)
        self.device = next(self.parameters()).device
        if self.replay:
            self.buffer.device = self.device
        return res

    def _to(self, *inputs: Tensor) -> List[Tensor]:
        return list(map(lambda x: x.to(self.device), inputs))

class LwP_Step(LwP_Base):
    def __init__(self, option: Config, classifier: SimpleConvNet) -> None:
        super().__init__(option, classifier)

    def train_one_epoch(self, data_loader) -> Tuple[Dict[str, float], str]:
        self.train()

        loss_sum = 0.
        loss_buf_sum = 0.
        loss_dis_sum = 0.
        loss_gen_sum = 0.
        loss_contra_real_sum = 0.
        loss_contra_fake_sum = 0.
        total_num_train = 0
        for images, labels in tqdm(data_loader, desc="[Train][Step Mode]"):
            images, labels = self._to(images, labels)
            bsize = images.shape[0]
            total_num_train += bsize
            
            ################################################
            # (1) Train GAN
            # Update generator and critic
            ################################################
            with torch.no_grad():
                feats_real = self.forward_encode(images)
            loss_dis, loss_gen = self.netG.train_a_batch(feats_real)
            loss_dis_sum += bsize*loss_dis.item()
            loss_gen_sum += bsize*loss_gen.item()

            ################################################
            # (2) Train OnlineNet and TargetNet(momentum update) by Contrastive Learning
            # Update onlineNet, targetNet
            ################################################
            ## generate fake and real features
            # Contrastive Learning with Real Features
            loss_contra_real = self.netS.train_a_batch(feats_real)
            loss_contra_real_sum += bsize*loss_contra_real.item()

            # Contrastive Learning with Fake Features
            self.netG.eval()
            with torch.no_grad():
                feats_fake = self.sample(images.size(0))
            loss_contra_fake = self.netS.train_a_batch(feats_fake)
            loss_contra_fake_sum += bsize*loss_contra_fake.item()

            ################################################
            # (3) Training Classifier with Labels
            # Update encoder, onlineNet.projector, classifier
            ################################################
            # Replay mode
            if self.replay:
                if not self.buffer.is_empty():
                    images_buf, labels_buf = self.buffer.get_data(
                        bsize, transform=None)
                    pred_buf = self.forward(images_buf)
                    loss_buf: Tensor = self.loss(pred_buf, labels_buf)
                    self.optim.zero_grad()
                    loss_buf.backward()
                    self.optim.step()
                    loss_buf_sum += bsize*loss_buf.item()
                self.buffer.add_data(examples=images.clone(), labels=labels)

            pred_label = self.forward(images)
            loss: Tensor = self.loss(pred_label, labels)
            loss_sum += bsize*loss.item()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        self.netS.global_step += 1
        losses = {
            'loss': loss_sum/total_num_train,
            'loss_gan_gen': loss_gen_sum/total_num_train, 
            'loss_gan_dis': loss_dis_sum/total_num_train,
            'loss_contra_real': loss_contra_real_sum/total_num_train,
            'loss_contra_fake': loss_contra_fake_sum/total_num_train,
            'loss_buf': loss_buf_sum/total_num_train,
            }
        
        msg = (
            f"BASE LOSS : {losses['loss']:.4f} " + 
            f"GAN(G) LOSS: {losses['loss_gan_gen']:.4f} " + 
            f"GAN(D) LOSS: {losses['loss_gan_dis']:.4f} " + 
            f"CONTRA(REAL) LOSS : {losses['loss_contra_real']:.4f} " + 
            f"CONTRA(FAKE) LOSS : {losses['loss_contra_fake']:.4f} " +
            f"BUFFER LOSS : {losses['loss_buf']:.4f} "
        )

        return losses, msg

