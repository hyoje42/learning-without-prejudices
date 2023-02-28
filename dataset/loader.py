import os
import pickle
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from dataset.mnist import ColourBiasedMNIST_BG, ColourBiasedMNIST_FG

ROOT_PATH = "dataset"

def get_data_loader(
    batch_size,
    mode="BG",
    train=True,
    transform=None,
    data_label_correlation=1.0,
    data_indices=None,
    colormap_idxs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
):
    """
    mode: 'FG'(foreground) or 'BG'(background)
    """
    if mode == "FG":
        Colored_MNIST = ColourBiasedMNIST_FG
    elif mode == "BG":
        Colored_MNIST = ColourBiasedMNIST_BG
    else:
        raise NotImplemented

    dataset = Colored_MNIST(
        root="dataset/MNIST-BIASED/MNIST",
        train=train,
        transform=transform,
        download=True,
        data_label_correlation=data_label_correlation,
        data_indices=data_indices,
        colormap_idxs=colormap_idxs,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader

def get_data_loader_from_npy(
    data: str, batch_size=32, transform=None, biased_ratio=0.9
):
    dic_tr_dl = {}
    dic_val_dl = {}

    for task in range(1, 10 + 1):
        if data == "mnist-biased":
            path = os.path.join(
                ROOT_PATH,
                data.upper(),
                f"MNIST-BIASED-BG-Biased{biased_ratio}-Task{task:02d}",
            )

        dataset = Attribute_Dataset(path, split="train", transform=transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

        dic_tr_dl[task] = dataloader

        dataset = Attribute_Dataset(path, split="valid", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        dic_val_dl[task] = dataloader

    return dic_tr_dl, dic_val_dl


class Attribute_Dataset(Dataset):
    def __init__(self, path, split="train", transform=None) -> None:
        super().__init__()
        data_path = os.path.join(path, f"images_{split}.npy")
        self.data = np.load(data_path)

        target_path = os.path.join(path, f"targets_{split}.npy")
        self.targets = torch.LongTensor(np.load(target_path))

        if len(self.targets.shape) > 1:
            self.attr = self.targets[:, 1]
            self.targets = self.targets[:, 0]

        if split != "test":
            with open(os.path.join(path, "attr_names.pickle"), "rb") as f:
                self.attr_names = pickle.load(f)

        # self.num_attrs =  self.attr.size(1)
        # self.set_query_attr_idx(query_attr_idx)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        if self.transform is not None:
            image = self.transform(image)

        return image, target


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

