# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os
import mat73
import numpy as np
import json
from tqdm import tqdm
import pytorch_lightning as pl

class PureMachineLearning(pl.LightningModule):
    def __init__(self):
        super(PureMachineLearning, self).__init__()
        self.number_producer_well = 9
        self.dense1 = nn.Linear(14, 128)  # 考虑时间和空间
        self.dense2 = nn.Linear(128, 128)
        self.bn1=nn.BatchNorm1d(128)
        self.bn2=nn.BatchNorm1d(128)
        self.dense_output = nn.Linear(128,  2 * self.number_producer_well)
    
    def forward(self, x):
        x = self.dense1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dense_output(x)
        return x
    def water_oil_production(self, inject_well_input, bottom_pressure, inject_input_time):
        bottom_pressure = bottom_pressure.to(self.device)
        inject_input_time = inject_input_time.to(self.device)
        inject_well_input = inject_well_input.to(self.device)
        input = torch.cat((inject_well_input, bottom_pressure, inject_input_time), dim=1).to(self.device)
        output = self.forward(input)
        q_o = output[:, :self.number_producer_well]
        q_w = output[:, self.number_producer_well:]
        return q_w, q_o

    def loss_fn(self, inject_well_input, bottom_pressure, inject_input_time, real_OPR, real_WPR):
        bottom_pressure = bottom_pressure.to(self.device)
        inject_input_time = inject_input_time.to(self.device)
        inject_well_input = inject_well_input.to(self.device)
        input = torch.cat((inject_well_input, bottom_pressure, inject_input_time), dim=1).to(self.device)
        output = self.forward(input)
        # 计算损失
        real_q = torch.cat((real_OPR, real_WPR), dim=1).to(self.device)
        loss = torch.mean((output - real_q)**2)
        return loss