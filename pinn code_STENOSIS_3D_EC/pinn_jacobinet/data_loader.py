import pandas as pd
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import os
from pathlib import Path
from config import L
device = torch.device("cuda:0")


def read_groundtruth_csv(csv_path, **read_csv_kwargs):
    csv_path = Path(csv_path)
    if csv_path.is_file():
        return pd.read_csv(csv_path, **read_csv_kwargs)

    parts_dir = csv_path.with_name(f"{csv_path.name}.parts")
    if not parts_dir.is_dir():
        raise FileNotFoundError(f"Missing CSV file or split directory: {csv_path}")

    part_files = sorted(parts_dir.glob("*.csv"))
    if not part_files:
        raise FileNotFoundError(f"No CSV chunks found in: {parts_dir}")

    return pd.concat(
        (pd.read_csv(part_file, **read_csv_kwargs) for part_file in part_files),
        ignore_index=True,
    )


def load_cfd_data(cfd_path):
    cfd_df = read_groundtruth_csv(cfd_path, skiprows=5)
    columns_to_convert = ['X [ m ]', ' Y [ m ]', ' Z [ m ]', ' Velocity [ m s^-1 ]', ' Velocity u [ m s^-1 ]', ' Velocity v [ m s^-1 ]', ' Velocity w [ m s^-1 ]', ' Pressure [ Pa ]']
    for col in columns_to_convert:
        cfd_df[col] = pd.to_numeric(cfd_df[col], errors='coerce')

    cfd_df = cfd_df.dropna(subset=columns_to_convert)
    x_cfd = cfd_df['X [ m ]'].values
    y_cfd = cfd_df[' Y [ m ]'].values
    z_cfd = cfd_df[' Z [ m ]'].values
    velocity_cfd = cfd_df[' Velocity [ m s^-1 ]'].values
    u_cfd = cfd_df[' Velocity u [ m s^-1 ]'].values
    v_cfd = cfd_df[' Velocity v [ m s^-1 ]'].values
    w_cfd = cfd_df[' Velocity w [ m s^-1 ]'].values
    p_cfd = cfd_df[' Pressure [ Pa ]'].values

    return x_cfd, y_cfd, z_cfd, velocity_cfd, u_cfd, v_cfd, w_cfd, p_cfd


def load_train_data(data_path):
    train_df = pd.read_excel(data_path, sheet_name=None)

    encrypted_points = torch.tensor(train_df['internal'][['x_n', 'y_n', 'z_n']].values, dtype=torch.float32).to(device)
    inlet_points = torch.tensor(train_df['inlet'][['x_n', 'y_n', 'z_n']].values, dtype=torch.float32).to(device)
    outlet_points = torch.tensor(train_df['outlet'][['x_n', 'y_n', 'z_n']].values, dtype=torch.float32).to(device)
    bd_points = torch.tensor(train_df['bd'][['x_n', 'y_n', 'z_n']].values, dtype=torch.float32).to(device)


    original_points = torch.cat((encrypted_points, inlet_points, bd_points, outlet_points), dim=0)
    min_vals = original_points.min(dim=0, keepdim=True)[0]
    max_vals = original_points.max(dim=0, keepdim=True)[0]
    mean_vals = (min_vals+max_vals)/2


    all_points = torch.cat((encrypted_points, inlet_points, bd_points, outlet_points), dim=0)

    return all_points, encrypted_points, inlet_points, outlet_points, bd_points, mean_vals, min_vals, max_vals

def load_origin_data(data_path):
    train_df = pd.read_excel(data_path, sheet_name=None)

    encrypted_points = torch.tensor(train_df['internal'][['x', 'y', 'z']].values, dtype=torch.float32).to(device)
    inlet_points = torch.tensor(train_df['inlet'][['x', 'y', 'z']].values, dtype=torch.float32).to(device)
    outlet_points = torch.tensor(train_df['outlet'][['x', 'y', 'z']].values, dtype=torch.float32).to(device)
    bd_points = torch.tensor(train_df['bd'][['x', 'y', 'z']].values, dtype=torch.float32).to(device)


    original_points = torch.cat((encrypted_points, inlet_points, bd_points, outlet_points), dim=0)
    min_vals = original_points.min(dim=0, keepdim=True)[0]
    max_vals = original_points.max(dim=0, keepdim=True)[0]
    mean_vals = (min_vals+max_vals)/2


    encrypted_points[:, 0] = (encrypted_points[:, 0] - mean_vals[:, 0]) / L
    encrypted_points[:, 1] = (encrypted_points[:, 1] - mean_vals[:, 1]) / L
    encrypted_points[:, 2] = (encrypted_points[:, 2] - mean_vals[:, 2]) / L

    inlet_points[:, 0] = (inlet_points[:, 0] - mean_vals[:, 0]) / L
    inlet_points[:, 1] = (inlet_points[:, 1] - mean_vals[:, 1]) / L
    inlet_points[:, 2] = (inlet_points[:, 2] - mean_vals[:, 2]) / L

    outlet_points[:, 0] = (outlet_points[:, 0] - mean_vals[:, 0]) / L
    outlet_points[:, 1] = (outlet_points[:, 1] - mean_vals[:, 1]) / L
    outlet_points[:, 2] = (outlet_points[:, 2] - mean_vals[:, 2]) / L

    bd_points[:, 0] = (bd_points[:, 0] - mean_vals[:, 0]) / L
    bd_points[:, 1] = (bd_points[:, 1] - mean_vals[:, 1]) / L
    bd_points[:, 2] = (bd_points[:, 2] - mean_vals[:, 2]) / L

    all_points = torch.cat((encrypted_points, inlet_points, bd_points, outlet_points), dim=0)

    return all_points, encrypted_points, inlet_points, outlet_points, bd_points, mean_vals, min_vals, max_vals

def save_history_data(l2_error_history, loss_df, pdeloss_history, bdloss_history, time_history, save_path):
    history_data = {
    "Epoch": range(len(l2_error_history)),
    "L2_Error_U": [err[0] for err in l2_error_history],
    "L2_Error_V": [err[1] for err in l2_error_history],
    "L2_Error_W": [err[2] for err in l2_error_history],
    "L2_Error_P": [err[3] for err in l2_error_history],
    "PDE_Loss": pdeloss_history,
    "Boundary_Loss": bdloss_history,
    "Total_Loss": loss_df.values.flatten(),
    "time_history": time_history,
    }

    history_df = pd.DataFrame(history_data)
    history_df.to_csv(save_path, index=False)

    return history_df
