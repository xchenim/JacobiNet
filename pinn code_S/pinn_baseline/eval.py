import matplotlib.pyplot as plt
import numpy as np
import torch
from config import L, W, T_non

device = torch.device("cuda:0")
def compute_l2_error(net2, x_fdm, y_fdm, T_fdm, mean_vals):
    net2.eval()


    all_points = torch.tensor(np.vstack((x_fdm, y_fdm)).T, dtype=torch.float32).to(device)
    all_points[:, 0] = (all_points[:, 0] - mean_vals[:, 0]) / W
    all_points[:, 1] = (all_points[:, 1] - mean_vals[:, 1]) / L

    with torch.no_grad():

        outputs = net2(all_points)
        u_pred = outputs[:, 0].cpu().numpy().flatten() * T_non

        T_rl2_error = np.sqrt(np.mean((u_pred - T_fdm) ** 2))/np.sqrt(np.mean(T_fdm ** 2))

    return T_rl2_error

def plot_points_and_vectors(inlet_points, outlet_points, bd1_points, bd2_points, encrypted_points):
    plt.figure(figsize=(8, 12))
    plt.scatter(encrypted_points[:, 0].cpu().detach().numpy(), encrypted_points[:, 1].cpu().detach().numpy(), color='#92B9FF', s=2)
    plt.scatter(inlet_points[:, 0].cpu().detach().numpy(), inlet_points[:, 1].cpu().detach().numpy(), color='#7E7EBB', s=2)
    plt.scatter(outlet_points[:, 0].cpu().detach().numpy(), outlet_points[:, 1].cpu().detach().numpy(), color='#E8974E', s=2)
    plt.scatter(bd1_points[:, 0].cpu().detach().numpy(), bd1_points[:, 1].cpu().detach().numpy(), color='#A1D06E', s=2)
    plt.scatter(bd2_points[:, 0].cpu().detach().numpy(), bd2_points[:, 1].cpu().detach().numpy(), color='#A1D06E', s=2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points and Velocity Vectors')
    plt.legend()
    plt.show()

def plot_loss_curve(loss_history, pdeloss_history, bdloss_history):
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    plt.plot(loss_history, label='Total Loss', color='#3480b8')
    plt.plot(pdeloss_history, label='PDE Loss', color='#9bbf8a')
    plt.plot(bdloss_history, label='BC Loss', color='#f79059')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_l2_error_curve(l2_error_history):
    l2_errors = np.array(l2_error_history)
    epochs = np.arange(len(l2_errors))
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    plt.plot(epochs, l2_errors[:, 0], label='relative L2 error T', color='#3480b8')

    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('L2 Error', fontsize=16)
    plt.title('L2 Error Curve', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.ylim(0.0015, 1.5)
    plt.tight_layout()
    plt.show()

def plot_lambda_curve(lambda_history):
    lambda_list = np.array(lambda_history)
    epochs = np.arange(len(lambda_list))
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 14})
    plt.plot(epochs, lambda_list[:, 0], label='lambda_T', color='#6cc2c7')
    plt.plot(epochs, lambda_list[:, 1], label='lambda_in', color='#f5ebb9')
    plt.plot(epochs, lambda_list[:, 2], label='lambda_out', color='#f5d6b1')
    plt.plot(epochs, lambda_list[:, 3], label='lambda_bd', color='#f7bdab')


    plt.yscale('log')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('lambda', fontsize=16)
    plt.title('lambda Curve', fontsize=18)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_sigma_curve(sigma_history):
    epochs = np.arange(len(sigma_history))
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 14})

    plt.plot(epochs, sigma_history, label='sigma', color='#7AA6DC', linewidth=2)

    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Sigma', fontsize=16)
    plt.title('Learnable Sigma Curve', fontsize=18)
    plt.ylim(0.5, 11)
    plt.grid(True)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()
