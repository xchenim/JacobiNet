import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.interpolate import griddata
import os
from config import L, W, T_non, fttype, ftsize, data_path, cfd_path, train_path, net3_path, v_max, vmax_error, plt_x, plt_y
from data_loader import load_cfd_data, load_train_data, save_history_data, load_origin_data
from models_helmholtz import HybridNet
import matplotlib.ticker as mticker

device = torch.device("cuda:0")
plt.rcParams['font.family'] = fttype
plt.rcParams['font.size'] = ftsize
def denormalize_points(points, mean_vals, scale_factor):
    denormalized_points = points * scale_factor + mean_vals
    return denormalized_points


def plot_T_gt_xy(filename):
    data = pd.read_csv(filename)
    x, y, T = data['x'].values, data['y'].values, data['T'].values

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(x, y, c=T, cmap='coolwarm', s=1, vmin=-v_max, vmax=v_max)
    plt.colorbar(scatter, label="Temperature")
    plt.title("T")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def plot_T_pred_xy(net3, points_list_xy, points_list_ds, min_vals, max_vals):
    net3.eval()
    with torch.no_grad():
        all_points = torch.cat(points_list_xy, dim=0)
        points_list_ds = torch.cat(points_list_ds, dim=0)
        outputs, _ = net3(points_list_ds)
        T = outputs[:, 0].cpu().numpy()

        x = denormalize_points(all_points[:, 0], mean_vals_xy[:, 0], W).cpu().numpy()
        y = denormalize_points(all_points[:, 1], mean_vals_xy[:, 1], L).cpu().numpy()


    fig, ax = plt.subplots(1, 1, figsize=(plt_x, plt_y))
    sc0 = ax.scatter(x*100, y*100, c=T * T_non, cmap='coolwarm', s=2, vmin=-v_max, vmax=v_max)
    formatter = mticker.FormatStrFormatter('%.1f')
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)

    plt.axis('scaled')

    plt.xlim(-0.55, 0.55)
    plt.ylim(-0.55, 0.55)
    tick_positions = [-0.5, 0.0, 0.5]
    tick_labels = ['-0.5', '0.0', '0.5']

    plt.xticks(tick_positions, tick_labels)
    plt.yticks(tick_positions, tick_labels)
    plt.show()

from matplotlib.colors import LogNorm
def plot_error(x_fdm, y_fdm, T_fdm, net2, points_list):
    net2.eval()

    with torch.no_grad():
        all_points = torch.cat(points_list, dim=0)
        outputs, _ = net2(all_points)
        T_pred = outputs[:, 0].cpu().numpy() * T_non

        x_pred = denormalize_points(all_points[:, 0], mean_vals_xy[:, 0], W).cpu().numpy()
        y_pred = denormalize_points(all_points[:, 1], mean_vals_xy[:, 1], L).cpu().numpy()

        T_fdm_interp = griddata(np.column_stack((x_fdm, y_fdm)), T_fdm, np.column_stack((x_pred, y_pred)), method='linear')

        T_pred[np.isinf(T_pred)] = 0
        T_fdm_interp[np.isinf(T_fdm_interp)] = 0
        T_pred[np.isnan(T_pred)] = 0
        T_fdm_interp[np.isnan(T_fdm_interp)] = 0

        T_error = np.abs(T_pred - T_fdm_interp)
        T_l1_error = np.mean(np.abs(T_pred - T_fdm_interp))
        T_l1_relative_error = np.mean(np.abs(T_pred - T_fdm_interp)) / np.mean(np.abs(T_fdm_interp))
        print(f"L1_T: ", f"abs: {round(T_l1_error, 3)}", f" rel: {round(T_l1_relative_error, 3)}")
        T_l2_error = np.sqrt(np.mean((T_pred - T_fdm_interp) ** 2))
        T_l2_relative_error = np.sqrt((np.mean((T_pred - T_fdm_interp) ** 2))) / np.sqrt(np.mean(T_fdm_interp ** 2))

        print(f"L2_T: ", f"abs: {round(T_l2_error, 3)}", f" rel: {round(T_l2_relative_error, 3)}")


        fig, ax = plt.subplots(1, 1, figsize=(plt_x, plt_y))

        sc0 = ax.scatter(x_pred*100, y_pred*100, c=abs(T_error), cmap='coolwarm', s=5, vmin=0, vmax=vmax_error)
        formatter = mticker.FormatStrFormatter('%.1f')
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().yaxis.set_major_formatter(formatter)


        plt.axis('scaled')

        plt.xlim(-0.55, 0.55)
        plt.ylim(-0.55, 0.55)
        tick_positions = [-0.5, 0.0, 0.5]
        tick_labels = ['-0.5', '0.0', '0.5']

        plt.xticks(tick_positions, tick_labels)
        plt.yticks(tick_positions, tick_labels)
        plt.show()

if __name__ == "__main__":

    os.makedirs(train_path, exist_ok=True)


    all_xy, encrypted_xy, inlet_xy, outlet_xy, bd1_xy, bd2_xy, mean_vals_xy, min_vals_xy, max_vals_xy, train_df_xy = load_origin_data(data_path)
    x_fdm, y_fdm, d_fdm, s_fdm, T_fdm = load_cfd_data(cfd_path)

    points_list_xy = [encrypted_xy, inlet_xy, outlet_xy, bd1_xy, bd2_xy]


    net3 = HybridNet().to(device)
    net3.load_state_dict(torch.load(net3_path, weights_only=True))

    plot_T_gt_xy(cfd_path)
    plot_T_pred_xy(net3, points_list_xy, points_list_xy, min_vals_xy, max_vals_xy)


    plot_error(x_fdm, y_fdm, T_fdm, net3, points_list_xy)
