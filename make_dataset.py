"""
python make_dataset.py --data mnist-biased --split both --biased_ratio 0.9
"""

import argparse
import os
import pickle
import random
import numpy as np

import torch

from dataset.mnist import ColourBiasedMNIST_BG

COLORMAP_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

HELPER_MNIST = {'train': 50000, 'valid': 10000, 'root': 'MNIST', 
                'attr_idxs': COLORMAP_INDEX}

HELPER = {
    'mnist-biased': HELPER_MNIST,
}

def make_tasks(data: str = 'mnist-biased', num_task=10, biased_ratio=0.9, split='both'):
    """
    Args:
        data: 'mnist-biased'
    """
    data_dir = f"dataset/{data.upper()}"
    root = os.path.join(data_dir, f"{HELPER[data]['root']}")

    flag_train = False
    flag_test = False

    if split == 'train':
        flag_train = True
    elif split == 'test':
        flag_test = True
    elif split == 'both':
        flag_train = True
        flag_test = True
    
    # split indices
    assert HELPER[data]['train']%num_task == 0
    assert HELPER[data]['valid']%num_task == 0

    indices = np.random.permutation(np.arange(HELPER[data]['train'] + HELPER[data]['valid']))

    dic_tr_indices  = {}
    dic_val_indices = {}
    part_tr         = int(HELPER[data]['train']/num_task)
    part_val        = int(HELPER[data]['valid']/num_task)
    for i in range(1, num_task+1):
        dic_tr_indices[i]  = indices[(i-1)*part_tr + (i-1)*part_val:i*part_tr + (i-1)*part_val]
        dic_val_indices[i] = indices[i*part_tr + (i-1)*part_val:i*part_tr + i*part_val]

    attr_idxs = HELPER[data]['attr_idxs']

    ## TRAIN
    if flag_train:
        for task in range(1, 10+1):
            attr_idxs = attr_idxs[1:] + [attr_idxs[0]]

            if data == 'mnist-biased':
                save_dir = os.path.join(
                    data_dir, f"MNIST-BIASED-BG-Biased{biased_ratio}-Task{task:02d}"
                )
                dataset_train = ColourBiasedMNIST_BG(root, train=True, download=True, 
                            data_label_correlation=biased_ratio, 
                            data_indices=dic_tr_indices[task], 
                            colormap_idxs=attr_idxs)

                dataset_valid = ColourBiasedMNIST_BG(root, train=True, download=True, 
                            data_label_correlation=biased_ratio, 
                            data_indices=dic_val_indices[task], 
                            colormap_idxs=attr_idxs)

            os.makedirs(save_dir, exist_ok=True)
            print(save_dir)

            with open(os.path.join(save_dir, "attr_names.pickle"), "wb") as f:
                pickle.dump(attr_idxs, f)

            np.save(os.path.join(save_dir, "images_train.npy"), dataset_train.data)
            np.save(os.path.join(save_dir, "images_valid.npy"), dataset_valid.data)
            np.save(os.path.join(save_dir, "targets_train.npy"), dataset_train.targets)
            np.save(os.path.join(save_dir, "targets_valid.npy"), dataset_valid.targets)

    ## TEST
    if flag_test:
        if data == 'mnist-biased':
            dataset_test = ColourBiasedMNIST_BG(root, train=False, download=True, 
                                    data_label_correlation=0.1,
                                    data_indices=None, 
                                    colormap_idxs=attr_idxs)
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, "images_test.npy"), dataset_test.data)
        np.save(os.path.join(data_dir, "targets_test.npy"), dataset_test.targets)

        print("unbiased dataset")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed',  default=777, type=int)
    parser.add_argument('--data',  default='mnist-biased', type=str, choices=['mnist-biased'])
    parser.add_argument('--split', default='train', type=str, choices=['train', 'test', 'both'])
    parser.add_argument('--biased_ratio', default=0.85, type=float)
    parser.add_argument('--num_task',  default=10, type=int)

    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) and unknown_args[0] == '-f' and 'jupyter' in unknown_args[1]:
        unknown_args = unknown_args[2:]
    assert len(unknown_args) == 0, f"Invalid Arguments: {str(unknown_args)}"

    print(args)

    # fix seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    make_tasks(data=args.data, num_task=args.num_task, biased_ratio=args.biased_ratio, split=args.split)