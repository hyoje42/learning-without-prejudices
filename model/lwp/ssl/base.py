"""
Self-Supervised Learning Method
"""
from typing import List
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model import SimpleConvNet, MLP_Block
from option import Config


class SSL_Base(nn.Module):
    def __init__(self, option: Config, classifier: SimpleConvNet):
        super().__init__()
        self.option = option
        self.classifier = classifier
        self.num_classes = option.num_class
        self.layer_idx = option.layer_idx

        self.device = torch.device('cpu')
        self.global_step = 0
        self.total_global_step = option.epoch * option.num_task # epoch * num_tasks

        ## Define Modules
        self._build_model()
        self.generator = None

        self.augment = {
            'aug_dropout': option.aug_dropout,
            'aug_noise': option.aug_noise,
            'aug_relu': option.aug_relu,
        }

    def _build_model(self):
        classifier_name = self.classifier.__class__.__name__
        mlp_units = 128
        self.backbone = nn.Sequential(OrderedDict([
            ('conv', nn.Sequential(OrderedDict(
                list(self.classifier.encoder.encoder.named_children())[self.layer_idx:]))
            ),
            ('avg_pool', nn.AdaptiveAvgPool2d((1, 1))),
            ('flatten', nn.Flatten(1)),
            ('projector', self.classifier.mlp),
            ('predictor', MLP_Block(mlp_units, mlp_units)),
        ]))

    @torch.no_grad()
    def forward_encode(self, x: Tensor) -> Tensor:
        m = self.classifier
        if 'Simple' in str(m.__class__):
            out = x
            for name, layer in m.encoder.encoder.named_children():
                if name == str(self.layer_idx): break
                out = layer(out)

        elif 'ResNet' in str(m.__class__):
            out = x
            for idx, (name, layer) in enumerate(m.named_children()):
                if idx == self.layer_idx: break
                out = layer(out)
                if name == 'bn1':
                    out = F.relu(out)

        return out

    def train_a_batch(self, feats: Tensor) -> Tensor:
        pass

    def sample(self, size: int) -> Tensor:
        return self.generator.sample(size)

    def _get_aug_features(self, feat:Tensor, mu = 0.0, sigma = 0.005, drop_rate = 0.5):
        """
        Add gaussian noise before Global Average Pooling.
        Args:
            feat : (B, C, H, W) feature map (detached).
            mu : mean.
            sigma : std.
        Return:
            feat : feature vector of original feature vector.
            feat_aug : augmentation result of feature map.
        """
        feat = feat.detach()
        feat_aug = feat.detach()
        if self.augment['aug_noise'] > 0:
            feat_aug = aug_noise(feat_aug, mu=mu, sigma=sigma)
            feat = feat if self.augment['aug_noise'] == 1 else aug_noise(feat, mu=mu, sigma=sigma)
        if self.augment['aug_dropout'] > 0:
            feat_aug = aug_dropout(feat_aug, drop_rate=drop_rate)
            feat = feat if self.augment['aug_dropout'] == 1 else aug_dropout(feat, drop_rate=drop_rate)
        if self.augment['aug_relu']:
            feat, feat_aug = aug_relu(feat, feat_aug)
        return feat, feat_aug
    
    def _get_list_classify_net(self):
        return [self.classifier]

    def _to(self, *inputs: Tensor) -> List[Tensor]:
        device = next(self.parameters()).device
        return list(map(lambda x: x.to(device), inputs))
    
    def _count_parameters(self, *models):
        num = 0
        for model in models:
            num += sum(p.numel() for p in model.parameters() if p.requires_grad)
        msg = f"{num/(1000000):.4f} M"
        return msg

    def _get_params(self):
        pass

    def print_net(self):
        pass

    def cuda(self, device=None):
        res = super().cuda(device)
        self.device = next(self.parameters()).device
        return res

    def init_weight(self, module: nn.Module):
        module.apply(weight_module)

def weight_module(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def aug_noise(feat: Tensor, mu = 0.0, sigma = 0.005) -> Tensor:
    noise = torch.normal(mean=mu, std=sigma, size=feat.size(), device=feat.device)
    return feat + noise

def aug_dropout(feat: Tensor, drop_rate = 0.5) -> Tensor:
    return F.dropout2d(feat, p=drop_rate)
    # return F.dropout2d(feat, p=drop_rate) * (1 - drop_rate)

def aug_relu(*feats: Tensor) -> Tensor:
    return list(map(lambda x : torch.relu(x), feats))
    