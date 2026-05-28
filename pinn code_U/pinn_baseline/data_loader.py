import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os
from config import L, W
device = torch.device("cuda:0")


def load_cfd_data(cfd_path):
    cfd_df = pd.read_csv(cfd_path)
    columns_to_convert = ['x', 'y', 'T']
    for col in columns_to_convert:
        cfd_df[col] = pd.to_numeric(cfd_df[col], errors='coerce')

    cfd_df = cfd_df.dropna(subset=columns_to_convert)
    x_fdm = cfd_df['x'].values
    y_fdm = cfd_df['y'].values
    T_fdm = cfd_df['T'].values


    return x_fdm, y_fdm, T_fdm


def load_train_data(data_path):
    train_df = pd.read_excel(data_path, sheet_name=None)
    encrypted_points = torch.tensor(train_df['encrypted_points'][['x', 'y']].values, dtype=torch.float32).to(device)
    inlet_points = torch.tensor(train_df['inlet_line_points'][['x', 'y']].values, dtype=torch.float32).to(device)
    outlet_points = torch.tensor(train_df['outlet_line_points'][['x', 'y']].values, dtype=torch.float32).to(device)
    bd1_points = torch.tensor(train_df['bd1'][['x', 'y']].values, dtype=torch.float32).to(device)
    bd2_points = torch.tensor(train_df['bd2'][['x', 'y']].values, dtype=torch.float32).to(device)


    original_points = torch.cat((encrypted_points, inlet_points, bd1_points, bd2_points, outlet_points), dim=0)
    min_vals = original_points.min(dim=0, keepdim=True)[0]
    max_vals = original_points.max(dim=0, keepdim=True)[0]
    mean_vals = (min_vals+max_vals)/2


    encrypted_points[:, 0] = (encrypted_points[:, 0] - mean_vals[:, 0]) / W
    encrypted_points[:, 1] = (encrypted_points[:, 1] - mean_vals[:, 1]) / L
    inlet_points[:, 0] = (inlet_points[:, 0] - mean_vals[:, 0]) / W
    inlet_points[:, 1] = (inlet_points[:, 1] - mean_vals[:, 1]) / L
    outlet_points[:, 0] = (outlet_points[:, 0] - mean_vals[:, 0]) / W
    outlet_points[:, 1] = (outlet_points[:, 1] - mean_vals[:, 1]) / L
    bd1_points[:, 0] = (bd1_points[:, 0] - mean_vals[:, 0]) / W
    bd1_points[:, 1] = (bd1_points[:, 1] - mean_vals[:, 1]) / L
    bd2_points[:, 0] = (bd2_points[:, 0] - mean_vals[:, 0]) / W
    bd2_points[:, 1] = (bd2_points[:, 1] - mean_vals[:, 1]) / L

    all_points = torch.cat((encrypted_points, inlet_points, bd1_points, bd2_points, outlet_points), dim=0)

    return all_points, encrypted_points, inlet_points, outlet_points, bd1_points, bd2_points, mean_vals, min_vals, max_vals

def save_history_data(l2_error_history, loss_df, pdeloss_history, bdloss_history, time_history, save_path):
    history_data = {
    "Epoch": range(len(l2_error_history)),
    "L2_Error_T": [err[0] for err in l2_error_history],
    "PDE_Loss": pdeloss_history,
    "Boundary_Loss": bdloss_history,
    "Total_Loss": loss_df.values.flatten(),
    "time_history": time_history,
    }

    history_df = pd.DataFrame(history_data)
    history_df.to_csv(save_path, index=False)

    return history_df
