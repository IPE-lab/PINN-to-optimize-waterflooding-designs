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


well_adjecent_index = {0: [0], 1: [0, 1], 2: [1], 3: [
    0, 2], 4: [0, 1, 2, 3], 5: [1, 3], 6: [2], 7: [2, 3], 8: [3]}

class PINNModel(pl.LightningModule):
    def __init__(self):
        super(PINNModel, self).__init__()
        self.number_producer_well = 9
        self.total_wells = 13

        self.dense1 = nn.Linear(14, 128)  # 考虑时间和空间
        self.dense2 = nn.Linear(128, 128)
        self.bn1=nn.BatchNorm1d(128)
        self.bn2=nn.BatchNorm1d(128)

        self.dense_output = nn.Linear(
            128, self.total_wells + self.number_producer_well)

        self.WIi = nn.Parameter(torch.tensor(
            [1.0]*len(well_adjecent_index), dtype=torch.float32), requires_grad=True)
        self.V0pi = nn.Parameter(torch.tensor(
            [0.2]*len(well_adjecent_index), dtype=torch.float32), requires_grad=True)

        self.Swc = nn.Parameter(torch.tensor(
            [-1.0], dtype=torch.float32), requires_grad=True)
        self.Sor = nn.Parameter(torch.tensor(
            [-1.0], dtype=torch.float32), requires_grad=True)

        self.Kro_ = nn.Parameter(torch.tensor(
            [0.00], dtype=torch.float32), requires_grad=True)
        self.Krw_ = nn.Parameter(torch.tensor(
            [0.00], dtype=torch.float32), requires_grad=True)

        self.mu_w = nn.Parameter(torch.tensor([0.31], dtype=torch.float32), requires_grad=False)
        self.mu_oil = nn.Parameter(torch.tensor([0.29], dtype=torch.float32), requires_grad=False)
        self.B_w = nn.Parameter(torch.tensor([1.0], dtype=torch.float32), requires_grad=False)
        self.B_oil = nn.Parameter(torch.tensor([1.30500], dtype=torch.float32), requires_grad=False)

        self.Cr = nn.Parameter(torch.tensor([8.1*10**-6], dtype=torch.float32), requires_grad=False)
        self.P0 = nn.Parameter(torch.tensor([5863.8], dtype=torch.float32), requires_grad=False)

        for producer_ind in well_adjecent_index.keys():
            for injector_ind in well_adjecent_index[producer_ind]:
                Tij = f'T_{str(producer_ind)}_{injector_ind}'
                if getattr(self, Tij, None) is None:
                    setattr(self, Tij, nn.Parameter(torch.tensor(
                        [0.0], dtype=torch.float32), requires_grad=True))

    def forward(self, x):
        x = torch.tanh(self.bn1(self.dense1(x)))
        # x = x + 0.1 * torch.randn_like(x)
        x = torch.tanh(self.bn2(self.dense2(x)))
        # x = x + 0.1 * torch.randn_like(x)
        output = self.dense_output(x)
        return output

    def water_oil_production(self, inject_well_input, bottom_pressure, inject_input_time):
        bottom_pressure = bottom_pressure.to(self.device)
        inject_input_time = inject_input_time.to(self.device)
        inject_well_input = inject_well_input.to(self.device)

        x = torch.cat((inject_well_input, bottom_pressure,
                      inject_input_time), dim=1).to(self.device)
        pressure_water_saturature_pred = self.forward(x)

        pressure = torch.exp(
            pressure_water_saturature_pred[:, :self.total_wells])
        water_saturate = torch.sigmoid(
            pressure_water_saturature_pred[:, self.total_wells:])

        S = (water_saturate - torch.sigmoid(self.Swc)) / (1 - torch.sigmoid(self.Swc) - torch.sigmoid(self.Sor))
        kro = (1 - S)**2
        krw = torch.exp(self.Krw_)/torch.exp(self.Kro_) * S**2

        q_w = torch.exp(self.WIi.reshape(1, -1)) * krw / (self.mu_w * self.B_w) * (pressure[:, :9] - bottom_pressure)
        q_o = torch.exp(self.WIi.reshape(1, -1)) * kro / (self.mu_oil * self.B_oil) * (pressure[:, :9] - bottom_pressure)
        return q_w, q_o


    def loss_fn(self, inject_well_input, bottom_pressure, inject_input_time, OPR, WPR):
        bottom_pressure = bottom_pressure.to(self.device)
        inject_input_time = inject_input_time.to(self.device)
        inject_well_input = inject_well_input.to(self.device)
        OPR = OPR.to(self.device)
        WPR = WPR.to(self.device)

        x = torch.cat((inject_well_input, bottom_pressure,
                      inject_input_time), dim=1).to(self.device)
        pressure_water_saturature_pred = self.forward(x)

        pressure = torch.exp(
            pressure_water_saturature_pred[:, :self.total_wells])
        water_saturate = torch.sigmoid(
            pressure_water_saturature_pred[:, self.total_wells:])

        S = (water_saturate - torch.sigmoid(self.Swc)) / (1 - torch.sigmoid(self.Swc) - torch.sigmoid(self.Sor))
        kro = (1 - S)**2
        krw = torch.exp(self.Krw_)/torch.exp(self.Kro_) * S**2

        q_w = torch.exp(self.WIi.reshape(1, -1)) * krw / (self.mu_w * self.B_w) * (pressure[:, :9] - bottom_pressure)
        q_o = torch.exp(self.WIi.reshape(1, -1)) * kro / (self.mu_oil * self.B_oil) * (pressure[:, :9] - bottom_pressure)

        
        fwpde = []
        fopde = []
        water_first_terms = []
        oil_third_terms = []
        for producer_ind in well_adjecent_index.keys():
                        
            injector_indices = well_adjecent_index[producer_ind]
            T_values = [torch.exp(getattr(self, f'T_{producer_ind}_{injector}')) for injector in injector_indices]
            Tij_stacked = torch.stack(T_values)

            Twij = krw[:, producer_ind].unsqueeze(-1).unsqueeze(-1) * Tij_stacked / (self.mu_w * self.B_w)
            Toij = kro[:, producer_ind].unsqueeze(-1).unsqueeze(-1) * Tij_stacked / (self.mu_oil * self.B_oil)

            injector_well_ind = torch.tensor(injector_indices) + self.number_producer_well
            pressure_diff = pressure[:, injector_well_ind] - pressure[:, producer_ind].unsqueeze(-1)

            w_frist_term = torch.sum(Twij.squeeze(-1) * pressure_diff, dim=1)
            oil_frist_term = torch.sum(Toij.squeeze(-1) * pressure_diff, dim=1)

            water_first_terms.append(w_frist_term)
            oil_third_terms.append(oil_frist_term)

        w_frist_term = torch.stack(water_first_terms, dim=1).squeeze()
        oil_frist_term = torch.stack(oil_third_terms, dim=1).squeeze()
        
        Vpi = (1 + self.Cr * (pressure[:, :self.number_producer_well] - self.P0))*torch.exp(self.V0pi).unsqueeze(0)
        water_third_term = Vpi * water_saturate / self.B_w
        oil_third_term = Vpi * (1 - water_saturate) / self.B_oil

        water_third_t_terms = []
        oil_third_t_terms = []
        for producer_ind in range(water_third_term.size(1)):
            water_third_t_terms.append(torch.autograd.grad(water_third_term[:, producer_ind], inject_input_time, grad_outputs=torch.ones_like(water_third_term[:, producer_ind]), retain_graph=True)[0])
            oil_third_t_terms.append(torch.autograd.grad(oil_third_term[:, producer_ind], inject_input_time, grad_outputs=torch.ones_like(oil_third_term[:, producer_ind]), retain_graph=True)[0])

        water_third_term_t = torch.stack(water_third_t_terms, dim=1).squeeze()
        oil_third_term_t = torch.stack(oil_third_t_terms, dim=1).squeeze()


        fwpde = w_frist_term - q_w - water_third_term_t
        fopde = oil_frist_term - q_o - oil_third_term_t

        f_loss = torch.mean(torch.square(fwpde)) + torch.mean(torch.square(fopde))

        pred_q = torch.cat((q_w, q_o), dim=1)
        real_q = torch.cat((WPR, OPR), dim=1)
        mse_loss = torch.mean(torch.square(pred_q - real_q))
        return f_loss, mse_loss
    
    @torch.no_grad()
    def model_analysis(self, inject_well_input, bottom_pressure, inject_input_time, OPR, WPR):
        bottom_pressure = bottom_pressure.to(self.device)
        inject_input_time = inject_input_time.to(self.device)
        inject_well_input = inject_well_input.to(self.device)
        OPR = OPR.to(self.device)
        WPR = WPR.to(self.device)

        x = torch.cat((inject_well_input, bottom_pressure,
                      inject_input_time), dim=1).to(self.device)
        pressure_water_saturature_pred = self.forward(x)

        pressure = torch.exp(
            pressure_water_saturature_pred[:, :self.total_wells])
        water_saturate = torch.sigmoid(
            pressure_water_saturature_pred[:, self.total_wells:])

        S = (water_saturate - torch.sigmoid(self.Swc)) / (1 - torch.sigmoid(self.Swc) - torch.sigmoid(self.Sor))
        kro = (1 - S)**2
        krw = torch.exp(self.Krw_)/torch.exp(self.Kro_) * S**2

        water_conn = {}
        oil_conn = {}
        for producer_ind in well_adjecent_index.keys():
            # first term culmulation
            for injector_ind in well_adjecent_index[producer_ind]:
                Tij = f'T_{str(producer_ind)}_{injector_ind}'
                Tij = torch.exp(getattr(self, Tij))

                Twij = krw[:, producer_ind] * Tij / (self.mu_w * self.B_w)
                Toij = kro[:, producer_ind] * Tij / (self.mu_oil * self.B_oil)
                
                key = f'{str(producer_ind)}_{injector_ind}'
                water_conn[key] = torch.sum(Twij*(pressure[:, injector_ind + 9] - pressure[:, producer_ind])).item()
                oil_conn[key] = torch.sum(Toij*(pressure[:, injector_ind + 9] - pressure[:, producer_ind])).item()
        return water_conn, oil_conn

    @torch.no_grad()
    def formation_pressure_relative_permeability(self, inject_well_input, bottom_pressure, inject_input_time, OPR, WPR):
        bottom_pressure = bottom_pressure.to(self.device)
        inject_input_time = inject_input_time.to(self.device)
        inject_well_input = inject_well_input.to(self.device)
        OPR = OPR.to(self.device)
        WPR = WPR.to(self.device)

        x = torch.cat((inject_well_input, bottom_pressure,
                      inject_input_time), dim=1).to(self.device)
        pressure_water_saturature_pred = self.forward(x)

        pressure = torch.exp(
            pressure_water_saturature_pred[:, :self.total_wells])
        water_saturate = torch.sigmoid(
            pressure_water_saturature_pred[:, self.total_wells:])

        S = (water_saturate - torch.sigmoid(self.Swc)) / (1 - torch.sigmoid(self.Swc) - torch.sigmoid(self.Sor))
        kro = (1 - S)**2
        krw = torch.exp(self.Krw_)/torch.exp(self.Kro_) * S**2

        return pressure, water_saturate, kro, krw

    def original_loss_fn(self, inject_well_input, bottom_pressure, inject_input_time, OPR, WPR):
        bottom_pressure = bottom_pressure.to(self.device)
        inject_input_time = inject_input_time.to(self.device)
        inject_well_input = inject_well_input.to(self.device)
        OPR = OPR.to(self.device)
        WPR = WPR.to(self.device)

        x = torch.cat((inject_well_input, bottom_pressure,
                      inject_input_time), dim=1).to(self.device)
        pressure_water_saturature_pred = self.forward(x)

        pressure = torch.exp(
            pressure_water_saturature_pred[:, :self.total_wells])
        water_saturate = torch.sigmoid(
            pressure_water_saturature_pred[:, self.total_wells:])

        S = (water_saturate - torch.sigmoid(self.Swc)) / (1 - torch.sigmoid(self.Swc) - torch.sigmoid(self.Sor))
        kro = (1 - S)**2
        krw = torch.exp(self.Krw_)/torch.exp(self.Kro_) * S**2

        q_w = torch.exp(self.WIi.reshape(1, -1)) * krw / (self.mu_w * self.B_w) * (pressure[:, :9] - bottom_pressure)
        q_o = torch.exp(self.WIi.reshape(1, -1)) * kro / (self.mu_oil * self.B_oil) * (pressure[:, :9] - bottom_pressure)

        fwpde = []
        fopde = []
        for producer_ind in well_adjecent_index.keys():
            w_frist_term = []
            o_first_term = []

            # first term culmulation
            for injector_ind in well_adjecent_index[producer_ind]:
                Tij = f'T_{str(producer_ind)}_{injector_ind}'
                Tij = torch.exp(getattr(self, Tij))

                Twij = krw[:, producer_ind] * Tij / (self.mu_w * self.B_w)
                Toij = kro[:, producer_ind] * Tij / (self.mu_oil * self.B_oil)

                w_frist_term.append(Twij*(pressure[:, injector_ind + 9] - pressure[:, producer_ind]))
                o_first_term.append(Toij*(pressure[:, injector_ind + 9] - pressure[:, producer_ind]))

            w_frist_term = torch.sum(torch.stack(w_frist_term, dim=1), dim=1)
            oil_frist_term = torch.sum(torch.stack(o_first_term, dim=1), dim=1)

            Vpi = torch.exp(self.V0pi[producer_ind])*(1 + self.Cr * (pressure[:, producer_ind] - self.P0))

            water_third_term = Vpi * water_saturate[:, producer_ind] / self.B_w
            oil_third_term = Vpi * (1 - water_saturate[:, producer_ind]) / self.B_oil

            water_third_term_t = torch.autograd.grad(water_third_term, inject_input_time, grad_outputs=torch.ones_like(water_third_term), create_graph=True)[0]
            oil_third_term_t = torch.autograd.grad(oil_third_term, inject_input_time, grad_outputs=torch.ones_like(oil_third_term), create_graph=True)[0]

            fwpde_tmp = w_frist_term - q_w[:, producer_ind] - water_third_term_t.squeeze()
            fopde_tmp = oil_frist_term - q_o[:, producer_ind] - oil_third_term_t.squeeze()

            fwpde.append(fwpde_tmp)
            fopde.append(fopde_tmp)

        f_loss = torch.mean(torch.square(torch.stack(fwpde, dim=1))) + torch.mean(torch.square(torch.stack(fopde, dim=1)))

        pred_q = torch.cat((q_w, q_o), dim=1)
        real_q = torch.cat((WPR, OPR), dim=1)
        mse_loss = torch.mean(torch.square(pred_q - real_q))
        return f_loss, mse_loss
