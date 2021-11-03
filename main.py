from config import load_config
import torch
import numpy as np
from trainer import Trainer
import torch.optim as optim
from torch.utils.data import DataLoader
from model.voxel2mesh import Voxel2Mesh as network


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


print("Create network")
classifier = network(cfg)
classifier.cuda()

print("Initialize optimizer")
optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                              classifier.parameters()),
                       lr=cfg.learning_rate)

print("Load pre-processed data")
data_obj = cfg.data_obj
data = data_obj.quick_load_data(cfg, trial_id)

loader = DataLoader(data[DataModes.TRAINING],
                    batch_size=classifier.config.batch_size,
                    shuffle=True)

print("Trainset length: {}".format(loader.__len__()))

print("Initialize evaluator")
evaluator = Evaluator(classifier, optimizer, data, trial_path, cfg, data_obj)

print("Initialize trainer")
trainer = Trainer(classifier, loader, optimizer, cfg.numb_of_itrs,
                  cfg.eval_every, trial_path, evaluator)

trainer.train(start_iteration=epoch)
# To evaluate a pretrained model, uncomment line below and comment the line above
# evaluator.evaluate(epoch)
