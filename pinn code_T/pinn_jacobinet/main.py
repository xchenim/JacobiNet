import torch
import os
from eval import plot_points_and_vectors, plot_loss_curve , plot_l2_error_curve, plot_lambda_curve, plot_sigma_curve
from models_helmholtz import Net_helmholtz, HybridNet
from data_loader import load_cfd_data, load_train_data, load_origin_data, save_history_data
from train import train
from utils import set_seed
from config import  seed, epochs, T_max, learning_rate, batch_size, his_freq, eta_min, train_path, jacobinet_path, cfd_path, data_path, save_path


device = torch.device("cuda:0")

if __name__ == "__main__":
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    set_seed(seed)

    net3 = HybridNet(geo_ckpt_path=jacobinet_path, device=device).to(device)
    optimizer = torch.optim.Adam(net3.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=eta_min)

    x_fdm, y_fdm, d_fdm, s_fdm, T_fdm = load_cfd_data(cfd_path)


    all_xy, encrypted_xy, inlet_xy, outlet_xy, bd1_xy, bd2_xy, mean_vals_xy, min_vals_xy, max_vals_xy, train_df_xy = load_origin_data(data_path)


    train_results = train(net3, optimizer, scheduler, epochs, train_path, cfd_path, his_freq, batch_size, all_xy, encrypted_xy, inlet_xy, outlet_xy, bd1_xy, bd2_xy, jacobinet_path, mean_vals_xy)

    loss_df, pdeloss_history, bdloss_history, time_history, sigma_history, l2_error_history, lambda_history = train_results
    save_history_data(l2_error_history, loss_df, pdeloss_history, bdloss_history, time_history, save_path)
