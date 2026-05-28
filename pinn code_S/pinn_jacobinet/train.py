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
from eval import plot_points_and_vectors, plot_loss_curve , plot_l2_error_curve, compute_l2_error
from models_poisson import Net_poisson, inlet_outlet_loss_poisson, boundary_loss_poisson, pde_loss_poisson
from data_loader import load_cfd_data, load_train_data, save_history_data
from utils import get_indices, gaussian_source
from config import lambda_T, lambda_inlet, lambda_bd, lambda_outlet, batch_size, net3_path

device = torch.device("cuda:0")

def train(net3, optimizer, scheduler, epochs, train_path, cfd_path, his_freq, batch_size, xy_points, encrypted_xy, inlet_xy, outlet_xy, bd1_xy, bd2_xy, jacobinet_path, mean_vals_xy):
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


    x_fdm, y_fdm, _, _, T_fdm = load_cfd_data(cfd_path)


    lam_T, lam_inlet, lam_bd, lam_outlet = lambda_T, lambda_inlet, lambda_bd, lambda_outlet
    for epoch in range(epochs):
        all_points = torch.cat((encrypted_xy, inlet_xy, outlet_xy, bd1_xy, bd2_xy), dim=0)

        num_batches = len(all_points) // batch_size if batch_size > 0 else 1

        if batch_size == 0:
            batch_size = len(all_points)

        indices = torch.randperm(len(all_points))


        for i in range(num_batches):
            optimizer.zero_grad()
            batch_indices = indices[i * batch_size:(i + 1) * batch_size]
            batch_points_xy = all_points[batch_indices]

            T_loss = pde_loss_poisson(net3, batch_points_xy)
            loss_pde = T_loss

            inlet_loss, outlet_loss = inlet_outlet_loss_poisson(net3, inlet_xy, outlet_xy)
            bd1_loss, bd2_loss = boundary_loss_poisson(net3, bd1_xy, bd2_xy)

            bd_loss = bd1_loss + bd2_loss

            if epoch % 100 == 0 and i == 0:
                lambda_history.append([lam_T, lam_inlet, lam_bd, lam_outlet])

            total_loss = lam_T * loss_pde
            total_loss.backward()
            optimizer.step()

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(total_loss)
        else:
            scheduler.step()

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(net3.state_dict(), net3_path)

        if epoch % his_freq == 0:
            pdeloss_history.append(loss_pde.item())
            bdloss_history.append((inlet_loss + bd_loss + outlet_loss).item())
            loss_history.append(total_loss.item())
            cumulative_time = time.time() - start_time
            time_history.append(cumulative_time)
            T_rl2_error = compute_l2_error(net3, x_fdm, y_fdm, T_fdm, mean_vals_xy)
            l2_error_history.append([T_rl2_error])

        if epoch % 100  == 0 or epoch+1 == epochs:
            print(f"Epoch: {epoch}, Loss: {total_loss.item():.2e}, L2_T: {T_rl2_error:.2e}, PDE Loss: {loss_pde.item():.2e}, T Loss: {T_loss.item():.2e}, Inlet Loss: {inlet_loss.item():.2e}, Boundary Loss: {bd_loss.item():.2e}, Outlet Loss: {outlet_loss.item():.2e}, lda_T: {lam_T:.2e}, lda_inlet: {lam_inlet:.2e}, lda_outlet: {lam_outlet:.2e}, lda_bd: {lam_bd:.2e}, time_history: {round(cumulative_time,1)}")

    return pd.DataFrame(loss_history), pdeloss_history, bdloss_history, time_history, sigma_history, l2_error_history, lambda_history
