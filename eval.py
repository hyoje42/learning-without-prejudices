"""
python eval.py --exp_path <path of experiment>
"""

import os
import json
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from dataset.loader import Attribute_Dataset, transform
import trainer as Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_path',   default='',  required=True, help='experiment path')
    parser.add_argument('--gpu',        default='0' ,               help='gpu')
    option, _ = parser.parse_known_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(option.gpu)
    exp_path = option.exp_path

    with open(os.path.join(exp_path, "options.json"), "r") as f:
        option_dict = json.load(f)

    for key in option_dict.keys():
        setattr(option, key, option_dict[key])

    # Load Dataset
    if option.data in ['mnist-biased']:
        print(f"[PREPARING LOADER] {option.data}")
        dataset = Attribute_Dataset(f"dataset/{option.data.upper()}", split='test', transform=transform)
        unbiased_te_dl = DataLoader(dataset, batch_size=option.batch_size, shuffle=False)

    # Define Trainer
    if option.method == 'LWP':
        trainer = Trainer.Trainer_LwP(option)
    else: # Baseline
        trainer = Trainer.Trainer(option)

    # Continual Learning
    print(f"[START TRAINING] {option.data}")
    acc_list = []
    for i in range(1, option.num_task+1):
        # Load weights
        ckpt = torch.load(os.path.join(option.exp_path, f"checkpoint_step_{option.epoch}_task_{i}.pth" ))
        trainer.net.load_state_dict(ckpt['net_state_dict'])
        # Eval
        acc_task_i = trainer._validate(data_loader=unbiased_te_dl, step=option.epoch, msg="[TEST]")
        acc_list.append(acc_task_i)

    for i in range(option.num_task):
        print(f"Accuracy of Task {i+1} : {acc_list[i]:.4f}")
    print(f"the Average of Accuracy all along the tasks : {np.mean(acc_list):.4f}")

if __name__ == "__main__": main()
