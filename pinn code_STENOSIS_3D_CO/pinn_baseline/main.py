import torch
import os
from eval import plot_points_and_vectors, plot_loss_curve , plot_l2_error_curve, plot_lambda_curve, plot_sigma_curve
from models_NS import Net_PINN
from data_loader import load_cfd_data, load_train_data, save_history_data
from train import train
from utils import set_seed, parabolic_velocity
from config import  seed, epochs, T_max, learning_rate, batch_size, his_freq, eta_min, train_path, cfd_path, data_path, save_path, inlet_vel, U_non, R

device = torch.device("cuda:0")

if __name__ == "__main__":
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_seed(seed)
    net2 = Net_PINN().to(device)
    optimizer = torch.optim.Adam(net2.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=eta_min)

    x_cfd, y_cfd, z_cfd, velocity_cfd, u_cfd, v_cfd, w_cfd, p_cfd = load_cfd_data(cfd_path)
    all_points, encrypted_points, inlet_points, outlet_points, bd_points, mean_vals, min_vals, max_vals = load_train_data(data_path)

    print(mean_vals)

    inlet_velocity = inlet_vel / U_non * parabolic_velocity(inlet_points, outlet_points)

    plot_points_and_vectors(inlet_points, outlet_points, bd_points, encrypted_points)

    loss_df, pdeloss_history, bdloss_history, time_history, sigma_history, l2_error_history, lambda_history = train(net2, optimizer, scheduler, epochs, encrypted_points, inlet_points, outlet_points, bd_points, train_path, cfd_path, mean_vals, his_freq, inlet_velocity, batch_size)

    save_history_data(l2_error_history, loss_df, pdeloss_history, bdloss_history, time_history, save_path)
    plot_sigma_curve(sigma_history)
    plot_l2_error_curve(l2_error_history)
    plot_lambda_curve(lambda_history)
    plot_loss_curve(loss_df.values.flatten(), pdeloss_history, bdloss_history)
