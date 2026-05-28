import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os
from config import W, L
device = torch.device("cuda:0")


def load_cfd_data(cfd_path):
    cfd_df = pd.read_csv(cfd_path)
    columns_to_convert = ['x', 'y', 'd', 's', 'T']
    for col in columns_to_convert:
        cfd_df[col] = pd.to_numeric(cfd_df[col], errors='coerce')

    cfd_df = cfd_df.dropna(subset=columns_to_convert)
    x_fdm = cfd_df['x'].values
    y_fdm = cfd_df['y'].values
    d_fdm = cfd_df['d'].values
    s_fdm = cfd_df['s'].values
    T_fdm = cfd_df['T'].values

    return x_fdm, y_fdm, d_fdm, s_fdm, T_fdm


def load_train_data(data_path):
    train_df = pd.read_excel(data_path, sheet_name=None)
    encrypted_points = torch.tensor(train_df['encrypted_points'][['d', 's']].values, dtype=torch.float32).to(device)
    inlet_points = torch.tensor(train_df['inlet_line_points'][['d', 's']].values, dtype=torch.float32).to(device)
    outlet_points = torch.tensor(train_df['outlet_line_points'][['d', 's']].values, dtype=torch.float32).to(device)
    bd1_points = torch.tensor(train_df['bd1'][['d', 's']].values, dtype=torch.float32).to(device)
    bd2_points = torch.tensor(train_df['bd2'][['d', 's']].values, dtype=torch.float32).to(device)


    original_points = torch.cat((encrypted_points, inlet_points, bd1_points, bd2_points, outlet_points), dim=0)
    min_vals = original_points.min(dim=0, keepdim=True)[0]
    max_vals = original_points.max(dim=0, keepdim=True)[0]
    mean_vals = (min_vals+max_vals)/2


    encrypted_points[:, 0] = (encrypted_points[:, 0] - mean_vals[:, 0])
    encrypted_points[:, 1] = (encrypted_points[:, 1] - mean_vals[:, 1])
    inlet_points[:, 0] = (inlet_points[:, 0] - mean_vals[:, 0])
    inlet_points[:, 1] = (inlet_points[:, 1] - mean_vals[:, 1])
    outlet_points[:, 0] = (outlet_points[:, 0] - mean_vals[:, 0])
    outlet_points[:, 1] = (outlet_points[:, 1] - mean_vals[:, 1])
    bd1_points[:, 0] = (bd1_points[:, 0] - mean_vals[:, 0])
    bd1_points[:, 1] = (bd1_points[:, 1] - mean_vals[:, 1])
    bd2_points[:, 0] = (bd2_points[:, 0] - mean_vals[:, 0])
    bd2_points[:, 1] = (bd2_points[:, 1] - mean_vals[:, 1])

    all_points = torch.cat((encrypted_points, inlet_points, bd1_points, bd2_points, outlet_points), dim=0)
    return all_points, encrypted_points, inlet_points, outlet_points, bd1_points, bd2_points, mean_vals, min_vals, max_vals

def load_origin_data(data_path):
    train_df_xy = pd.read_excel(data_path, sheet_name=None)
    encrypted_xy = torch.tensor(train_df_xy['encrypted_points'][['x', 'y']].values, dtype=torch.float32).to(device)
    inlet_xy = torch.tensor(train_df_xy['inlet_line_points'][['x', 'y']].values, dtype=torch.float32).to(device)
    outlet_xy = torch.tensor(train_df_xy['outlet_line_points'][['x', 'y']].values, dtype=torch.float32).to(device)
    bd1_xy = torch.tensor(train_df_xy['bd1'][['x', 'y']].values, dtype=torch.float32).to(device)
    bd2_xy = torch.tensor(train_df_xy['bd2'][['x', 'y']].values, dtype=torch.float32).to(device)


    original_points = torch.cat((encrypted_xy, inlet_xy, bd1_xy, bd2_xy, outlet_xy), dim=0)
    min_vals_xy = original_points.min(dim=0, keepdim=True)[0]
    max_vals_xy = original_points.max(dim=0, keepdim=True)[0]
    mean_vals_xy = (min_vals_xy+max_vals_xy)/2


    encrypted_xy[:, 0] = (encrypted_xy[:, 0] - mean_vals_xy[:, 0]) / W
    encrypted_xy[:, 1] = (encrypted_xy[:, 1] - mean_vals_xy[:, 1]) / L
    inlet_xy[:, 0] = (inlet_xy[:, 0] - mean_vals_xy[:, 0]) / W
    inlet_xy[:, 1] = (inlet_xy[:, 1] - mean_vals_xy[:, 1]) / L
    outlet_xy[:, 0] = (outlet_xy[:, 0] - mean_vals_xy[:, 0]) / W
    outlet_xy[:, 1] = (outlet_xy[:, 1] - mean_vals_xy[:, 1]) / L
    bd1_xy[:, 0] = (bd1_xy[:, 0] - mean_vals_xy[:, 0]) / W
    bd1_xy[:, 1] = (bd1_xy[:, 1] - mean_vals_xy[:, 1]) / L
    bd2_xy[:, 0] = (bd2_xy[:, 0] - mean_vals_xy[:, 0]) / W
    bd2_xy[:, 1] = (bd2_xy[:, 1] - mean_vals_xy[:, 1]) / L

    all_xy = torch.cat((encrypted_xy, inlet_xy, bd1_xy, bd2_xy, outlet_xy), dim=0)
    return all_xy, encrypted_xy, inlet_xy, outlet_xy, bd1_xy, bd2_xy, mean_vals_xy, min_vals_xy, max_vals_xy, train_df_xy

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
