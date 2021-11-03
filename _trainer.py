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

from util.util import enumerateWithEstimate
from dsets import SegmentationDataset, HU_TOP_LIM, HU_BOTTOM_LIM, THRESHOLD
from model import UNetWrapper, SegmentationAugmentation

import time
import wandb
from IPython import embed


class Trainer(object):
    def training_step(self, data, epoch):
        # Get the minibatch

        self.optimizer.zero_grad()
        loss, log = self.net.loss(data, epoch)
        loss.backward()
        self.optimizer.step()
        # embed()

        return log

    def __init__(self, net, trainloader, optimizer, numb_of_itrs, eval_every,
                 save_path, evaluator):

        self.net = net
        self.trainloader = trainloader
        self.optimizer = optimizer

        self.numb_of_itrs = numb_of_itrs
        self.eval_every = eval_every
        self.save_path = save_path

        self.evaluator = evaluator

    def train(self, start_iteration=1):

        print("Start training...")

        self.net = self.net.train()
        iteration = start_iteration

        print_every = 1
        for epoch in range(10000000):  # loop over the dataset multiple times

            for itr, data in enumerate(self.trainloader):

                # training step
                loss = self.training_step(data, start_iteration)

                if iteration % print_every == 0:
                    log_vals = {}
                    for key, value in loss.items():
                        log_vals[key] = value / print_every
                    log_vals['iteration'] = iteration

                iteration = iteration + 1

                if iteration % self.eval_every == self.eval_every - 1:  # print every K epochs
                    self.evaluator.evaluate(iteration)

                if iteration > self.numb_of_itrs:
                    break

        logger.info("... end training!")
