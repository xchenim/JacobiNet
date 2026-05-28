import torch
import os
from eval import plot_points_and_vectors, plot_loss_curve , plot_l2_error_curve, plot_lambda_curve, plot_sigma_curve
from models_NS import HybridNet
from data_loader import load_cfd_data, load_train_data, load_origin_data, save_history_data
from train import train
from utils import set_seed, parabolic_velocity
from config import seed, epochs, T_max, learning_rate, batch_size, his_freq, eta_min, train_path, jacobinet_path, cfd_path, data_path, save_path, inlet_vel, U_non

device = torch.device("cuda:0")

if __name__ == "__main__":
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_seed(seed)
    net3 = HybridNet(geo_ckpt_path=jacobinet_path, device=device).to(device)
    optimizer = torch.optim.Adam(net3.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=eta_min)

    x_cfd, y_cfd, z_cfd, velocity_cfd, u_cfd, v_cfd, w_cfd, p_cfd = load_cfd_data(cfd_path)


    all_xyz, encrypted_xyz, inlet_xyz, outlet_xyz, bd_xyz, mean_vals_xyz, min_vals_xyz, max_vals_xyz = load_origin_data(data_path)
    print(mean_vals_xyz)
    inlet_velocity = inlet_vel / U_non * parabolic_velocity(inlet_xyz, outlet_xyz)


    plot_points_and_vectors(inlet_xyz, outlet_xyz, bd_xyz, encrypted_xyz)


    train_results = train(net3, optimizer, scheduler, epochs, train_path, cfd_path, his_freq, batch_size, all_xyz, encrypted_xyz, inlet_xyz, outlet_xyz, bd_xyz, jacobinet_path, inlet_velocity, mean_vals_xyz)

    loss_df, pdeloss_history, bdloss_history, time_history, sigma_history, l2_error_history, lambda_history = train_results

    save_history_data(l2_error_history, loss_df, pdeloss_history, bdloss_history, time_history, save_path)
    plot_sigma_curve(sigma_history)
    plot_l2_error_curve(l2_error_history)
    plot_lambda_curve(lambda_history)
    plot_loss_curve(loss_df.values.flatten(), pdeloss_history, bdloss_history)
