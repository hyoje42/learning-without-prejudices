from typing import Tuple
from collections import OrderedDict
from copy import deepcopy
import math

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F

from model import SimpleConvNet
from .base import SSL_Base
from option import Config

class Contra_mse_loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        pred / target: (B, C)
        """
        pred_norm = F.normalize(pred, dim=1, p=2)
        target_norm = F.normalize(target, dim=1, p=2)

        return self.mse_loss(pred_norm, target_norm) / pred.size(0)

class BYOL(SSL_Base):
    def __init__(self, option: Config, classifier: SimpleConvNet):
        super().__init__(option, classifier)
        self.onlineNet = self.backbone
        self.targetNet = deepcopy(
            nn.Sequential(OrderedDict(list(self.onlineNet.named_children())[:-1]))
        )
        self.targetNet.requires_grad_(False)
        
        self.byol_loss = Contra_mse_loss()
        self.optim_ssl = optim.Adam(self._get_params(), lr=0.0001, weight_decay=0.0005)

    def forward_for_contra(self, t1, t2) -> Tuple[Tensor, Tensor]:
        pred_online = self.onlineNet(t1)
        with torch.no_grad():
            proj_target = self.targetNet(t2)

        return pred_online, proj_target

    def train_a_batch(self, feats: Tensor) -> Tensor:
        """
        Train OnlineNet and TargetNet(momentum update) by Contrastive Learning (BYOL style).
        Update onlineNet, targetNet.
        """
        self.train()
        # generate augmented features
        feats_1, feats_2 = self._get_aug_features(feats.detach(), mu=0., sigma=0.005, drop_rate=0.2)

        pred_online_1, proj_target_1 = self.forward_for_contra(feats_1, feats_2)
        pred_online_2, proj_target_2 = self.forward_for_contra(feats_2, feats_1)
        loss_contra: Tensor = (
            self.byol_loss(pred_online_1, proj_target_1.detach()) + 
            self.byol_loss(pred_online_2, proj_target_2.detach())
        )
        self.optim_ssl.zero_grad()
        loss_contra.backward()
        self.optim_ssl.step()

        # update weights of targetNet using Momentum
        self._momentum_update_targetNet()

        return loss_contra

    def _momentum_update_targetNet(self):
        """
        Momentum update of the target encoder & projector
        """
        # base tau : 0.996 / cosine annealing
        self.tau = 1. - (1. - 0.996)*(math.cos(math.pi*self.global_step / self.total_global_step)+1)/2.
        for param_o, param_t in zip(self.onlineNet.parameters(), self.targetNet.parameters()):
            param_t.data = self.tau * param_t.data + (1. - self.tau) * param_o.data

    def _get_params(self):
        modules = [self.classifier, self.onlineNet]
        params = []
        for m in modules:
            params += list(m.parameters())
        return set(params)

    def print_net(self):
        msg = (
            f"[PARAMETER:OnlineNet] {self._count_parameters(self.onlineNet)} " + 
            f"[PARAMETER:TargetNet] {self._count_parameters(self.targetNet)}"
        )
        return msg

    