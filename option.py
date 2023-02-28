# -*- coding: utf-8 -*-
import os
import json
import argparse
import random
import shutil
import torch
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--exp_name',   default='',                 help='experiment name')
parser.add_argument('--method',           default='base',             help='method', choices=['base', 'lwp'])
parser.add_argument('--data',             default='mnist-biased',    type=str,   help='type of dataset', choices=['mnist-biased'])
parser.add_argument('--batch_size',       default=32,        type=int,   help='mini-batch size')
parser.add_argument('--epoch',            default=10,        type=int,   help='epoch of each task')

parser.add_argument('--log_step',         default=50,     type=int,   help='step for logging in iteration')
parser.add_argument('--save_step',        default=1,      type=int,   help='step for saving in epoch')
parser.add_argument('--data_dir',         default='./',               help='data directory')
parser.add_argument('--save_dir',         default='./exps',           help='save directory for checkpoint')

parser.add_argument('--seed',             default=777,    type=int,   help='random seed')
parser.add_argument('--num_workers',      default=4,      type=int,   help='number of workers in data loader')
parser.add_argument('--cpu',              action='store_true',        help='enables cuda')
parser.add_argument('--gpu',              default='0',                help='which number of gpu used')

parser.add_argument('--biased_r',         default=0.85,   type=float, help='biased ratio(defualt: 0.85)')
parser.add_argument('--replay',           action='store_true',              help='use replay or not')


class Config():
    def __init__(self, opt) -> None:
        self.exp_name: str = opt.exp_name
        self.method: str = opt.method.upper()
        self.data:str = opt.data
        self.batch_size: int = opt.batch_size
        self.epoch: int = opt.epoch

        self.log_step: int = opt.log_step
        self.save_step: int = opt.save_step
        self.data_dir: str = opt.data_dir
        self.save_dir: str = opt.save_dir

        self.seed: int = opt.seed
        self.num_workers: int = opt.num_workers

        self.cuda: bool = not opt.cpu
        self.gpu: str = opt.gpu

        self.biased_r: float = opt.biased_r

        # LwP
        self.replay: bool = opt.replay

        assert len(self.__dict__) == len(opt.__dict__), "Check argparse"
        if self.method == "LWP":
            self.layer = 2
            self.z_size = 1000
            self.g_channel = 128
            self.d_channel = 64
            self.aug_dropout = 2
            self.aug_noise = 2
            self.aug_relu = False

            self.layer_idx = {
                'mnist-biased': {0: 0, 1: 3, 2: 6, 3: 9, 4: 12},
            }[self.data][self.layer]
            
            self.gen_output_size = {
                'mnist-biased': {0: 28, 1: 28, 2: 28, 3: 28, 4: 28},
            }[self.data][self.layer]

            self.gen_out_c_size = {
                'mnist-biased': {0: 3, 1: 16, 2: 32, 3: 64, 4: 128},
            }[self.data][self.layer]

        if self.data in ['mnist-biased']:
            self.num_class = 10
        self.num_task = 10

        self.hyper_param = {
            'data': '', 
            'method': '',
            'replay': 'Replay',
            'biased_r': 'Bias',
            'epoch': 'Ep',
            'batch_size': 'B',
            'seed': 'SEED',

        }

        self.hyper_param.update({
            
        })

        self._build()

    def _build(self):
        # Set exp name
        for k, v in self.hyper_param.items():
            self.exp_name += f"_{v}{self.__getattribute__(k)}"

        if self.exp_name[0] == '_': self.exp_name = self.exp_name[1:]

        print(self.exp_name)
        self._save()

    def _backend_setting(self):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu

        if self.seed is None:
            self.seed = random.randint(1, 10000)
        
        if torch.cuda.is_available() and not self.cuda:
            print('[WARNING] GPU is available, but not use it')

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.cuda:
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _save(self):
        log_dir = os.path.join(self.save_dir, self.exp_name)
        if os.path.exists(log_dir):
            if 'debug' in self.exp_name: 
                isdelete = "y"
            else:
                isdelete = input("delete exist exp dir (y/n): ")
            if isdelete == "y":
                shutil.rmtree(log_dir)
            elif isdelete == "n":
                raise FileExistsError
            else:
                raise FileExistsError

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        option_path = os.path.join(self.save_dir, self.exp_name, "options.json")

        with open(option_path, 'w') as fp:
            json.dump(self.__dict__, fp, indent=4, sort_keys=True)


def get_option() -> Config:
    option, unknown_args = parser.parse_known_args()
    if len(unknown_args) and unknown_args[0] == '-f' and 'jupyter' in unknown_args[1]:
        unknown_args = unknown_args[2:]
    assert len(unknown_args) == 0, f"Invalid Arguments: {str(unknown_args)}"
    return Config(option)
