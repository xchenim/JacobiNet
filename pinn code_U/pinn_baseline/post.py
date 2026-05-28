import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.interpolate import griddata
import os
device = torch.device("cuda:0")
from config import L, W, T_non, fttype, ftsize, data_path, cfd_path, net2_path, v_max, vmax_error, plt_x, plt_y, post_path
from data_loader import load_cfd_data, load_train_data
from models_laplace import Net_laplace
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = fttype
plt.rcParams['font.size'] = ftsize

def denormalize_points(points, mean_vals, scale_factor):
    denormalized_points = points * scale_factor + mean_vals
    return denormalized_points

def plot_T_gt(filename):
    data = pd.read_csv(filename)
    x, y, T = data['x'].values, data['y'].values, data['T'].values

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(x, y, c=T, cmap='coolwarm', s=1, vmin = - v_max, vmax = v_max)
    plt.colorbar(scatter, label="Temperature")
    plt.title("T")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_T_pred(net2, points_list, min_vals, max_vals):
    net2.eval()
    with torch.no_grad():
        all_points = torch.cat(points_list, dim=0)
        outputs = net2(all_points)
        T = outputs[:, 0].cpu().numpy()

        x = denormalize_points(all_points[:, 0], mean_vals[:, 0], W).cpu().numpy()
        y = denormalize_points(all_points[:, 1], mean_vals[:, 1], L).cpu().numpy()

    fig, ax = plt.subplots(figsize=(plt_x, plt_y))
    sc = ax.scatter(x, y, c=T * T_non, cmap='coolwarm', s=2, vmin = - v_max, vmax = v_max)
    plt.colorbar(sc, ax=ax, orientation='vertical', label='Velocity')
    plt.title('T pred')
    plt.axis("equal")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

from matplotlib.colors import LogNorm
def plot_error(x_fdm, y_fdm, T_fdm, net2, points_list):
    net2.eval()

    with torch.no_grad():
        all_points = torch.cat(points_list, dim=0)
        outputs = net2(all_points)
        T_pred = outputs[:, 0].cpu().numpy() * T_non

        x_pred = denormalize_points(all_points[:, 0], mean_vals[:, 0], W).cpu().numpy()
        y_pred = denormalize_points(all_points[:, 1], mean_vals[:, 1], L).cpu().numpy()

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
        sc0 = ax.scatter(x_pred*100, y_pred*100, c=T_error, cmap='coolwarm', s=5, norm=LogNorm(vmin=1e-3, vmax=10*vmax_error))

        formatter = mticker.FormatStrFormatter('%.1f')
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.gca().yaxis.set_major_formatter(formatter)

        plt.axis('scaled')

        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        tick_positions = [0.0, 0.5, 1.0]
        tick_labels = ['0.0', '0.5', '1.0']

        plt.xticks(tick_positions, tick_labels)
        plt.yticks(tick_positions, tick_labels)


        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    all_points, encrypted_points, inlet_points, outlet_points, bd1_points, bd2_points, mean_vals, min_vals, max_vals = load_train_data(post_path)
    x_fdm, y_fdm, T_fdm = load_cfd_data(cfd_path)
    points_list = [encrypted_points, inlet_points, outlet_points, bd1_points, bd2_points]


    net2 = Net_laplace().to(device)
    net2.load_state_dict(torch.load(net2_path, weights_only=True))


    plot_T_gt(cfd_path)
    plot_T_pred(net2, points_list, min_vals, max_vals)
    plot_error(x_fdm, y_fdm, T_fdm, net2, points_list)
