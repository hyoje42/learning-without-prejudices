# -*- coding: utf-8 -*-
import os
import logging

def logger_setting(exp_name, save_dir):
    logger = logging.getLogger(exp_name)
    # formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    formatter = logging.Formatter('[%(asctime)s] : %(message)s')

    log_out = os.path.join(save_dir, exp_name, 'train.log')
    file_handler = logging.FileHandler(log_out)
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.setLevel(logging.INFO)

    return logger
