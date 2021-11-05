from config import load_config
import torch
import numpy as np
from trainer import Trainer
import torch.optim as optim
from torch.utils.data import DataLoader
from dsets import *
from model.voxel2mesh import Voxel2Mesh as network

#import argparse
#
#parser = argparse.ArgumentParser()
#parser.add_argument(
#        '-proj',
#        type=str,
#        required=True,
#        help='projection along which to slice the volumes'
#)
#parser.add_argument(
#        '-batch_size',
#        type=int,
#        required=False,
#        default=4,
#        help='batch size'
#)
#parser.add_argument(
#        '-workers',
#        type=int,
#        required=False,
#        default=4,
#        help='number of subprocesses to spawn to do preprocessing'
#)
#parser.add_argument(
#        '-epochs',
#        type=int,
#        required=False,
#        default=50,
#        help='epochs'
#)
#parser.add_argument(
#        '-in_channels',
#        type=int,
#        required=False,
#        default=7,
#        help='num of input channels of the network'
#)
#parser.add_argument(
#        '-depth',
#        type=int,
#        required=False,
#        default=7,
#        help='depth of the network'
#)
#parser.add_argument(
#        '-wf',
#        type=int,
#        required=False,
#        default=3,
#        help='2**wf is the num of filters in the first conv'
#)
#parser.add_argument(
#        '-lr',
#        type=float,
#        required=False,
#        default=0.0001,
#        help='init learning rate'
#)
#parser.add_argument(
#        '--shuffle',
#        action='store_true',
#        default=False,
#        help=('shuffle the dataset every epoch')
#)
#
#args = parser.parse_args()
#
#
#restart_path = 0
#restart_path = 'runs/models/v2_noaug_sagittal_suffle_64batch/'
#
#shuffle_bool = args.shuffle
#num_workers = args.workers
#batch_size = args.batch_size
#epochs = args.epochs
## in_channels = args.in_channels
## depth = args.depth
## wf = args.wf
## lr = args.lr
shuffle_bool = True
num_workers = 2
batch_size = 1
epochs = 20

run_name = 'prova1'

print(args)




def init(cfg):

    save_path = cfg.save_path + cfg.save_dir_prefix + str(
        cfg.experiment_idx).zfill(3)

    mkdir(save_path)

    trial_id = (len([dir for dir in os.listdir(save_path) if 'trial' in dir]) +
                1) if cfg.trial_id is None else cfg.trial_id
    trial_save_path = save_path + '/trial_' + str(trial_id)

    if not os.path.isdir(trial_save_path):
        mkdir(trial_save_path)
        copytree(os.getcwd(),
                 trial_save_path + '/source_code',
                 ignore=ignore_patterns('*.git', '*.txt', '*.tif', '*.pkl',
                                        '*.off', '*.so', '*.json', '*.jsonl',
                                        '*.log', '*.patch', '*.yaml', 'wandb',
                                        'run-*'))

    seed = trial_id
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = True  # speeds up the computation

    return trial_save_path, trial_id


exp_id = 3
# Initialize
cfg = load_config(exp_id)
trial_path, trial_id = init(cfg)
print('Experiment ID: {}, Trial ID: {}'.format(cfg.experiment_idx,
                                                   trial_id))


print("Load pre-processed data")
trn_dset = dsets.SegmentationDataset(10, isValSet_bool=False)
val_dset = dsets.SegmentationDataset(10, isValSet_bool=True)

#print("Initialize evaluator")
#evaluator = Evaluator(classifier, optimizer, data, trial_path, cfg, data_obj)

print("Initialize trainer")
trainer = Trainer(classifier, trn_dset, val_dset, tb_prefix=run_name)

trainer.train(epochs,batch_size,num_workers=num_workers, shuffle_bool=shuffle_bool, lr=lr)

# To evaluate a pretrained model, uncomment line below and comment the line above
# evaluator.evaluate(epoch)
