from typing import List
import os
import cv2
import random

import torch
from torch.utils.data import Dataset


class CelebA_HQ(Dataset):
    def __init__(
        self, root: str = "./dataset", txt_file: str = None, image_size: int = 224
    ) -> None:

        self._image_size: int = image_size
        self._root: str = root

        with open(txt_file) as txt_file:
            self._image_list: List[str] = txt_file.read().splitlines()

        random.shuffle(self._image_list)

        self.targets = list(map(lambda x: int(x.split(',')[-1]), self._image_list))
        self.targets = torch.LongTensor(self.targets)

    def __getitem__(self, index: int) -> torch.Tensor:

        _image_dir: str = self._image_list[index].split(",")[0]
        self.label: int = int(self._image_list[index].split(",")[1])

        assert self.label == self.targets[index].item()

        self.image = (
            cv2.resize(
                cv2.imread(os.path.join(self._root, _image_dir)),
                dsize=(self._image_size, self._image_size),
                interpolation=cv2.INTER_CUBIC,
            ).transpose(2, 0, 1)
            / 255.0
        )

        return torch.FloatTensor(self.image), self.label

    def __len__(self) -> int:

        return len(self._image_list)
