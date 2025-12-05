import os
from collections import deque
import sys
import numpy as np
import tqdm
import torch
from torch import nn, cat
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import models_lightning
import data_lightning
import config

data_dir = ""
conditions = ""
epoch = 50

model = models_lightning.ai23(config=config, device="gpu")
data = data_lightning.KARR_DataModule(data_dir=data_dir, conditions=condition, config=config)
trainer = pl.Trainer(accelerator="gpu", devices=1, min_epoch=1, max_epoch=epoch, precision=16)

trainer.fit(model, data)
trainer.validate(model, data)
trainer.test(model, data)
