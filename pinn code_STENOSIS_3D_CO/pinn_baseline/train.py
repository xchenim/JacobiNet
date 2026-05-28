import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.autograd as autograd
import time
import matplotlib.pyplot as plt
import os
import random
from scipy.spatial import ConvexHull
from eval import plot_points_and_vectors, plot_loss_curve, plot_l2_error_curve, compute_l2_error
from models_NS import Net_PINN, inlet_outlet_loss, boundary_loss, pde_loss
from data_loader import load_cfd_data, load_train_data, save_history_data
from utils import get_indices
from config import lambda_u, lambda_v, lambda_w, lambda_c, lambda_inlet, lambda_bd, lambda_outlet, batch_size, net2_u_path


def train(net2, optimizer, scheduler, epochs, encrypted_points, inlet_points, outlet_points, bd_points, train_path, cfd_path, mean_vals, his_freq, inlet_velocity, batch_size):
    best_loss = float('inf')
    sigma_history = []
    loss_history = []
    pdeloss_history = []
    bdloss_history = []
    time_history = []
    l2_error_history = []
    cumulative_time = 0
    start_time = time.time()
    lambda_history = []


    x_cfd, y_cfd, z_cfd, velocity_cfd, u_cfd, v_cfd, w_cfd, p_cfd = load_cfd_data(cfd_path)


    lam_u, lam_v, lam_w, lam_c, lam_inlet, lam_bd, lam_outlet = lambda_u, lambda_v, lambda_w, lambda_c, lambda_inlet, lambda_bd, lambda_outlet

    for epoch in range(epochs):
        all_points = torch.cat((encrypted_points, inlet_points, bd_points, outlet_points), dim=0)
        num_batches = len(all_points) // batch_size if batch_size > 0 else 1

        if batch_size == 0:
            batch_size = len(all_points)

        indices = torch.randperm(len(all_points))

        for i in range(num_batches):
            optimizer.zero_grad()
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            batch_points = all_points[batch_indices]

            u_loss, v_loss, w_loss, c_loss = pde_loss(net2, batch_points)
            loss_pde = u_loss + v_loss + w_loss + c_loss
            inlet_loss, outlet_loss = inlet_outlet_loss(net2, inlet_points, outlet_points, inlet_velocity)
            bd_loss = boundary_loss(net2, bd_points)

        if epoch % 100 == 0 and i == 0:
            lambda_history.append([lam_u, lam_v, lam_w, lam_c, lam_inlet, lam_bd, lam_outlet])


        total_loss = (lam_u * u_loss + lam_v * v_loss + lam_w * w_loss + lam_c * c_loss + lam_inlet * inlet_loss + lam_bd * bd_loss + lam_outlet * outlet_loss)

        total_loss.backward()
        optimizer.step()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(total_loss)
        else:
            scheduler.step()


        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            torch.save(net2.state_dict(), net2_u_path)


        if epoch % his_freq == 0:
            pdeloss_history.append(loss_pde.item())
            bdloss_history.append((inlet_loss + bd_loss + outlet_loss).item())
            loss_history.append(total_loss.item())
            cumulative_time = time.time() - start_time
            time_history.append(cumulative_time)


            u_rl2_error, v_rl2_error, w_rl2_error, p_rl2_error = compute_l2_error(net2, x_cfd, y_cfd, z_cfd, u_cfd, v_cfd, w_cfd, p_cfd, mean_vals)
            l2_error_history.append([u_rl2_error, v_rl2_error, w_rl2_error, p_rl2_error])

        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch}, Loss: {total_loss.item():.2e}, "
                f"L2_U: {u_rl2_error:.2e}, L2_V: {v_rl2_error:.2e}, L2_W: {w_rl2_error:.2e}, L2_P: {p_rl2_error:.2e}, "
                f"PDE: {loss_pde.item():.2e}, u: {u_loss.item():.2e}, v: {v_loss.item():.2e}, w: {w_loss.item():.2e}, c: {c_loss.item():.2e}, "
                f"Inlet: {inlet_loss.item():.2e}, Boundary: {bd_loss.item():.2e}, Outlet: {outlet_loss.item():.2e}, "
                f"λu: {lam_u:.2e}, λv: {lam_v:.2e}, λw: {lam_w:.2e}, λc: {lam_c:.2e}, λin: {lam_inlet:.2e}, λout: {lam_outlet:.2e}, λbd: {lam_bd:.2e}, "
                f"time: {round(cumulative_time,1)}"
            )


    return pd.DataFrame(loss_history), pdeloss_history, bdloss_history, time_history, sigma_history, l2_error_history, lambda_history
