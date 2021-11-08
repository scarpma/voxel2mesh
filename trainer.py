import argparse
import datetime
import hashlib
import os
import shutil
import socket
import sys

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.optim

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (chamfer_distance, mesh_edge_loss,
                            mesh_laplacian_smoothing, mesh_normal_consistency)

from util.util import enumerateWithEstimate
from dsets import SegmentationDataset, HU_TOP_LIM, HU_BOTTOM_LIM, THRESHOLD

METRIC_LOSS_IDX = 1
METRIC_CE_IDX   = 2 # cross entropy
METRIC_CF_IDX   = 3 # chamfer distance
METRIC_N_IDX    = 4 # normal consistency
METRIC_E_IDX    = 5 # edge
METRIC_LAP_IDX  = 6 # laplacian

METRICS_SIZE = 7

RUNS_DIR = '../voxel2meshRuns'

class Trainer:
    def __init__(self, seg_model, train_ds, val_ds, tb_prefix='', aug_dict={}):
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        self.segmentation_model = seg_model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.tb_prefix = tb_prefix
        #self.aug_dict = aug_dict

        self.totalTrainingSamples_count = 0
        self.trn_writer = None
        self.val_writer = None

        #self.aug_model = SegmentationAugmentation(**self.aug_dict)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        if self.use_cuda:
            print("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                self.segmentation_model = nn.DataParallel(self.segmentation_model)
                #self.aug_model = nn.DataParallel(self.aug_model)
            self.segmentation_model = self.segmentation_model.to(self.device)
            #self.aug_model = self.aug_model.to(self.device)



    def initOptimizer(self, lr):
        return Adam(self.segmentation_model.parameters(), lr=lr)



    def initTrainDl(self, batch_size, num_workers=0):
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        train_dl = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=self.use_cuda,
        )

        return train_dl



    def initValDl(self, batch_size, num_workers=0):
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()

        val_dl = DataLoader(
            self.val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=self.use_cuda,
        )

        return val_dl



    def initTensorboardWriters(self):
        if self.trn_writer is None:
            log_dir = os.path.join('runs', self.tb_prefix, self.time_str)

            self.trn_writer = SummaryWriter(
                log_dir=log_dir + '_trn_seg_')
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '_val_seg_')



    def train(self, epochs, batch_size, num_workers, shuffle_bool, lr=0.001):

        ## self.segmentation_model, self.aug_model = self.initModel()
        self.optimizer = self.initOptimizer(lr)

        #print("Starting {}, {}".format(type(self).__name__, self.cli_args))
        print("Starting {}".format(type(self).__name__))

        train_dl = self.initTrainDl(batch_size, num_workers)
        val_dl = self.initValDl(batch_size, num_workers)

        best_score = 0.0
        self.validation_cadence = 1

        for epoch_ndx in range(1, epochs + 1):

            ## TRAINING
            trnMetrics_t = self.doTraining(epoch_ndx, train_dl, shuffle_bool)
            self.logMetrics(epoch_ndx, 'trn', trnMetrics_t)

            ## VALIDATION
            if epoch_ndx == 1 or epoch_ndx % self.validation_cadence == 0:
                valMetrics_t = self.doValidation(epoch_ndx, val_dl)
                score = self.logMetrics(epoch_ndx, 'val', valMetrics_t)
                best_score = max(score, best_score)

                self.saveModel('seg', epoch_ndx, score == best_score, epoch_ndx == epochs)

                self.logImages(epoch_ndx, 'trn', train_dl)
                self.logImages(epoch_ndx, 'val', val_dl)

        self.trn_writer.close()
        self.val_writer.close()



    def doTraining(self, epoch_ndx, train_dl, shuffle_bool):
        trnMetrics_g = torch.zeros(METRICS_SIZE, len(train_dl.dataset), device=self.device)
        self.segmentation_model.train()
        if shuffle_bool : train_dl.dataset.shuffleSamples()

        batch_iter = enumerateWithEstimate(
            train_dl,
            "E{} Training".format(epoch_ndx),
            start_ndx=train_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.computeBatchLoss(batch_ndx, batch_tup, train_dl.batch_size, trnMetrics_g)
            loss_var.backward()

            self.optimizer.step()

        self.totalTrainingSamples_count += trnMetrics_g.size(1)

        return trnMetrics_g.to('cpu')



    def doValidation(self, epoch_ndx, val_dl):
        with torch.no_grad():
            valMetrics_g = torch.zeros(METRICS_SIZE, len(val_dl.dataset), device=self.device)
            self.segmentation_model.eval()

            #batch_iter = enumerateWithEstimate(
            #    val_dl,
            #    "E{} Validation ".format(epoch_ndx),
            #    start_ndx=val_dl.num_workers,
            #)
            for batch_ndx, batch_tup in enumerate(val_dl):
                self.computeBatchLoss(batch_ndx, batch_tup, val_dl.batch_size, valMetrics_g)

        return valMetrics_g.to('cpu')



    def computeBatchLoss(self, batch_ndx, data, batch_size, metrics_g,
                         classificationThreshold=0.5):

        data['x'] = data['x'].to(self.device, non_blocking=True)
        data['y_voxels'] = data['x'].to(self.device, non_blocking=True)
        data['surface_points'] = data['x'].to(self.device, non_blocking=True)

        ## if self.segmentation_model.training and self.aug_dict:
        ##     input_g, label_g = self.aug_model(input_g, label_g)

        pred = self.segmentation_model(data)
        # embed()

        CE_Loss = nn.CrossEntropyLoss()
        ce_loss = CE_Loss(pred[0][-1][3], data['y_voxels'])

        chamfer_loss = torch.tensor(0).float().cuda()
        edge_loss = torch.tensor(0).float().cuda()
        laplacian_loss = torch.tensor(0).float().cuda()
        normal_consistency_loss = torch.tensor(0).float().cuda()

        target = data['surface_points'][0].cuda()
        for k, (vertices, faces, _, _, _) in enumerate(pred[0][1:]):

            pred_mesh = Meshes(verts=list(vertices), faces=list(faces))
            pred_points = sample_points_from_meshes(pred_mesh, 3000)

            chamfer_loss += chamfer_distance(pred_points, target)[0]
            laplacian_loss += mesh_laplacian_smoothing(pred_mesh,
                                                       method="uniform")
            normal_consistency_loss += mesh_normal_consistency(pred_mesh)
            edge_loss += mesh_edge_loss(pred_mesh)

        loss = 1 * chamfer_loss + 1 * ce_loss + 0.1 * laplacian_loss + 1 * edge_loss + 0.1 * normal_consistency_loss

        metrics[METRIC_LOSS_IDX] = loss.detach(),
        metrics[METRIC_CF_IDX] = chamfer_loss.detach(),
        metrics[METRIC_CE_IDX] = ce_loss.detach(),
        metrics[METRIC_N_IDX] = normal_consistency_loss.detach(),
        metrics[METRIC_E_IDX] = edge_loss.detach(),
        metrics[METRIC_LAP_IDX] = laplacian_loss.detach(),

        return loss



    def diceLoss(self, prediction_g, label_g, epsilon=1):
        diceLabel_g = label_g.sum(dim=[1,2,3]) # cast bool to int
        dicePrediction_g = prediction_g.sum(dim=[1,2,3])
        diceCorrect_g = (prediction_g * label_g).sum(dim=[1,2,3])

        diceRatio_g = (2 * diceCorrect_g + epsilon) \
            / (dicePrediction_g + diceLabel_g + epsilon)

        return 1 - diceRatio_g



    def logImages(self, epoch_ndx, mode_str, dl):
        self.segmentation_model.eval()

        images = sorted(dl.dataset.patients_list)
        for patient_name in images:
            for slice_ndx in range(6):
                n_slices = dl.dataset.n_slices
                ct_ndx = slice_ndx * (n_slices - 1) // 5
                sample_tup = dl.dataset.getitem_TrainingSample(patient_name, ct_ndx)

                ct_t, label_t, patient_name, ct_ndx = sample_tup

                input_g = ct_t.to(self.device).unsqueeze(0)
                label_g = pos_g = label_t.to(self.device).unsqueeze(0)

                prediction_g = self.segmentation_model(input_g)[0]
                prediction_a = prediction_g.to('cpu').detach().numpy()[0] > THRESHOLD
                label_a = label_g.cpu().numpy()[0][0]

                ct_t[:-1,:,:] -= (HU_TOP_LIM + HU_BOTTOM_LIM) / 2
                ct_t[:-1,:,:] /= (HU_TOP_LIM - HU_BOTTOM_LIM)
                ct_t[:-1,:,:] += 0.5

                ctSlice_a = ct_t[dl.dataset.contextSlices_count].numpy()
                shape = list(ctSlice_a.shape[-2:])

                image_a = np.zeros(shape+[3], dtype=np.float32)
                image_a[:,:,:] = ctSlice_a.reshape(shape+[1])
                image_a[:,:,0] += prediction_a & (1 - label_a) # falsi positivi
                image_a[:,:,0] += (1 - prediction_a) & label_a # falsi negativi
                image_a[:,:,1] += ((1 - prediction_a) & label_a) * 0.5

                image_a[:,:,1] += prediction_a & label_a # veri positivi
                image_a *= 0.5
                image_a.clip(0, 1, image_a)

                writer = getattr(self, mode_str + '_writer')
                writer.add_image(
                    f'{mode_str}/{patient_name}_prediction_{slice_ndx}',
                    image_a,
                    self.totalTrainingSamples_count,
                    dataformats='HWC',
                )

                ## if epoch_ndx == 1:
                ##     image_a = np.zeros(shape+[3], dtype=np.float32)
                ##     image_a[:,:,:] = ctSlice_a.reshape(shape+[1])
                ##     # image_a[:,:,0] += (1 - label_a) & lung_a # Red
                ##     image_a[:,:,1] += label_a  # Green
                ##     # image_a[:,:,2] += neg_a  # Blue

                ##     image_a *= 0.5
                ##     image_a[image_a < 0] = 0
                ##     image_a[image_a > 1] = 1
                ##     writer.add_image(
                ##         '{}/{}_label_{}'.format(
                ##             mode_str,
                ##             patient_name,
                ##             slice_ndx,
                ##         ),
                ##         image_a,
                ##         self.totalTrainingSamples_count,
                ##         dataformats='HWC',
                ##     )
                # This flush prevents TB from getting confused about which
                # data item belongs where.
                writer.flush()



    def logMetrics(self, epoch_ndx, mode_str, metrics_t):

        metrics_a = metrics_t.detach().numpy()
        sum_a = metrics_a.sum(axis=1)
        assert np.isfinite(metrics_a).all()

        allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()

        metrics_dict['percent_all/tp'] = \
            sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fn'] = \
            sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
        metrics_dict['percent_all/fp'] = \
            sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100


        precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
        recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
            / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

        metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
            / ((precision + recall) or 1)

        print(("E{} {:8} "
                 + "{loss/all:.4f} loss, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score"
                  ).format(
            epoch_ndx,
            mode_str,
            **metrics_dict,
        ))
        print(("E{} {:8} "
                  + "{loss/all:.4f} loss, "
                  + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
        ).format(
            epoch_ndx,
            mode_str + '_all',
            **metrics_dict,
        ))

        self.initTensorboardWriters()
        writer = getattr(self, mode_str + '_writer')

        prefix_str = 'seg_'

        for key, value in metrics_dict.items():
            writer.add_scalar(prefix_str + key, value, self.totalTrainingSamples_count)

        writer.flush()

        score = metrics_dict['pr/recall']

        return score



    def saveModel(self, type_str, epoch_ndx, isBest=False, isLast=False):
        file_path = os.path.join(
            RUNS_DIR,
            'models',
            self.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.tb_prefix,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.segmentation_model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            #'sys_argv': sys.argv,
            'time': str(datetime.datetime.now()),
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        print("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                RUNS_DIR, 'models',
                self.tb_prefix,
                f'{type_str}_{self.time_str}._{self.tb_prefix}.best.state')
            shutil.copyfile(file_path, best_path)

            print("Saved model params to {}".format(best_path))

        if isLast:
            last_path = os.path.join(
                RUNS_DIR, 'models',
                self.tb_prefix,
                f'{type_str}_{self.time_str}._{self.tb_prefix}.last.state')
            shutil.copyfile(file_path, last_path)

            print("Saved model params to {}".format(last_path))

        with open(file_path, 'rb') as f:
            print("SHA1: " + hashlib.sha1(f.read()).hexdigest())



if __name__ == '__main__':

    SegmentationTrainingApp().main()


