"""
python train.py --exp_name Exp --epoch 20 --biased_r 0.85 --method base 
python train.py --exp_name Exp --epoch 20 --biased_r 0.85 --method lwp  
python train.py --exp_name Exp --epoch 20 --biased_r 0.85 --method lwp  --replay
"""

import os, sys
import time
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset.loader import Attribute_Dataset, get_data_loader_from_npy, transform
from option import get_option
import trainer as Trainer

def main():
    start = time.time()
    option = get_option()
    option._backend_setting()

    # Load Dataset
    if option.data in ['mnist-biased']:
        print(f"[PREPARING LOADER] {option.data}")
        dic_tr_dl, dic_val_dl = get_data_loader_from_npy(option.data, batch_size=option.batch_size, 
                                                         transform=transform, biased_ratio=option.biased_r)
        dataset = Attribute_Dataset(f"dataset/{option.data.upper()}", split='test', transform=transform)
    
        unbiased_te_dl = DataLoader(dataset, batch_size=option.batch_size, shuffle=False)

    # Define Tensorboard Writer
    tb_writer = SummaryWriter(log_dir=os.path.join(option.save_dir, option.exp_name))
    # Define Trainer
    if option.method == 'LWP':
        trainer = Trainer.Trainer_LwP(option)
    else: # Baseline
        trainer = Trainer.Trainer(option)
    trainer.tb_writer = tb_writer

    # log command
    trainer.logger.info(' '.join(sys.argv))

    # weight initialization.
    trainer.logger.info("Use Initialization of weights.")
    trainer._init_weights()

    # Re Seed Setting for data loader issue
    option._backend_setting()

    # Continual Learning
    results_unbiased_test = []
    print(f"[START TRAINING] {option.data}")
    for i in range(1, option.num_task+1):
        acc_test = trainer.train_task(train_loader=dic_tr_dl[i], val_loader=unbiased_te_dl)
        results_unbiased_test.append(acc_test)
        if option.data in ['cifar10-c', 'mnist-biased']:
            acc_task_1_after = trainer._validate(data_loader=dic_val_dl[1], step=option.epoch, msg="[Validation][Task-1][After Task-" + str(i) + "]")
            if i > 1:
                acc_task_pre_after = trainer._validate(data_loader=dic_val_dl[i-1], step=option.epoch, msg="[Validation][Previous Step][After Task-" + str(i) + "]")
                tb_writer.add_scalar("valid/task_pre_acc_after", acc_task_pre_after, global_step=i)
            acc_task_i = trainer._validate(data_loader=dic_val_dl[i], step=option.epoch, msg="[Validation][Task-" + str(i) + "]")
            tb_writer.add_scalar("valid/task1_acc_after", acc_task_1_after, global_step=i)
            tb_writer.add_scalar("valid/task_acc", acc_task_i, global_step=i)
        tb_writer.add_scalar("test/acc", acc_test, global_step=i)
        tb_writer.flush()
        trainer._save_model(step=option.epoch, task=i)

    # validate all tasks on final model
    trainer.logger.info("Validate all Tasks on Final Model")
    results_biased_valid = []
    for i in range(1, option.num_task+1):
        acc_valid_last = trainer._validate(data_loader=dic_val_dl[i], step=option.epoch, msg="[Validation][Final Model][Task-" + str(i) + "]")
        results_biased_valid.append(acc_valid_last)
    trainer.logger.info(f"[Validation][Final Model] ACCURACY (Mean) : {np.mean(results_biased_valid):.4f}")
    trainer.logger.info(f"[Validation][Final Model] ACCURACY (Std ) : {np.std(results_biased_valid):.4f}")

    trainer.logger.info("UNBIASED TESTSET RESULTS")
    for idx, acc in enumerate(results_unbiased_test):
        trainer.logger.info(f"[Task-{idx+1}] ACCURACY : {acc:.4f}")
    trainer.logger.info(f"[TEST][Unbiased] ACCURACY (Mean) : {np.mean(results_unbiased_test):.4f}")
    trainer.logger.info(f"[TEST][Unbiased] ACCURACY (Std ) : {np.std(results_unbiased_test):.4f}")

    # Elapsed Time
    trainer.logger.info(f"Elapsed Time : {(time.time() - start)/3600:.1f} hour")

if __name__ == "__main__": main()
