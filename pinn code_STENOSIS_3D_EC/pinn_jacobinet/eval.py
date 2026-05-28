import matplotlib.pyplot as plt
import numpy as np
import torch
from config import L, U_non, P_non, inlet_vel
from utils import parabolic_velocity

device = torch.device("cuda:0")
def compute_l2_error(net3, x_cfd, y_cfd, z_cfd, u_cfd, v_cfd, w_cfd, p_cfd, mean_vals):
    net3.eval()


    all_points = torch.tensor(
        np.vstack((x_cfd, y_cfd, z_cfd)).T, dtype=torch.float32, device=mean_vals.device
    )
    all_points[:, 0] = (all_points[:, 0] - mean_vals[:, 0]) / L
    all_points[:, 1] = (all_points[:, 1] - mean_vals[:, 1]) / L
    all_points[:, 2] = (all_points[:, 2] - mean_vals[:, 2]) / L

    with torch.no_grad():
        outputs, _ = net3(all_points)
        u_pred = (outputs[:, 0].cpu().numpy().flatten()) * U_non
        v_pred = (outputs[:, 1].cpu().numpy().flatten()) * U_non
        w_pred = (outputs[:, 2].cpu().numpy().flatten()) * U_non
        p_pred = (outputs[:, 3].cpu().numpy().flatten()) * P_non

    def rl2(a, b):
        denom = np.sqrt(np.mean(b**2)) + 1e-12
        return np.sqrt(np.mean((a - b)**2)) / denom

    u_rl2_error = rl2(u_pred, u_cfd)
    v_rl2_error = rl2(v_pred, v_cfd)
    w_rl2_error = rl2(w_pred, w_cfd)
    p_rl2_error = rl2(p_pred, p_cfd)

    return u_rl2_error, v_rl2_error, w_rl2_error, p_rl2_error


def plot_points_and_vectors(
    inlet_points, outlet_points, bd_points, encrypted_points,
    umax=0.7,
    max_arrows=1000
):
    def to_np(t): return t.detach().cpu().numpy()


    vel_t = parabolic_velocity(inlet_points, outlet_points) * float(umax)
    V     = to_np(vel_t)

    P_in  = to_np(inlet_points)
    P_out = to_np(outlet_points)
    P_bd  = to_np(bd_points)
    P_en  = to_np(encrypted_points)


    Ni = P_in.shape[0]
    if Ni > max_arrows:
        idx = np.linspace(0, Ni - 1, max_arrows, dtype=int)
        Pq, Vq = P_in[idx], V[idx]
    else:
        Pq, Vq = P_in, V


    if np.allclose(Vq, 0):
        print("[warn] inlet velocity is all zero. Check that R uses the same units as inlet_points; use R_phys/L for nondimensional coordinates.")


    span = max(
        float(np.ptp(P_in[:, 0])) if P_in.size else 1.0,
        float(np.ptp(P_in[:, 1])) if P_in.size else 1.0,
        float(np.ptp(P_in[:, 2])) if P_in.size else 1.0,
    )
    arrow_len = 0.15 * span


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(P_en[:, 0], P_en[:, 1], P_en[:, 2], s=2, c='#92B9FF', label='interior')
    ax.scatter(P_in[:, 0], P_in[:, 1], P_in[:, 2], s=8, c='#7E7EBB', label='inlet')
    ax.scatter(P_out[:, 0], P_out[:, 1], P_out[:, 2], s=8, c='#E8974E', label='outlet')
    ax.scatter(P_bd[:, 0], P_bd[:, 1], P_bd[:, 2], s=2, c='#A1D06E', label='bd')

    ax.quiver(Pq[:, 0], Pq[:, 1], Pq[:, 2],
              Vq[:, 0], Vq[:, 1], Vq[:, 2],
              length=arrow_len, normalize=False, color='green')

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('Points and Parabolic Inlet Velocity (3D)')
    ax.legend()
    plt.tight_layout()
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
    plt.plot(epochs, l2_errors[:, 0], label='relative L2 error u', color='#3480b8')
    plt.plot(epochs, l2_errors[:, 1], label='relative L2 error v', color='#9bbf8a')
    plt.plot(epochs, l2_errors[:, 2], label='relative L2 error w', color="#9900ff")
    plt.plot(epochs, l2_errors[:, 3], label='relative L2 error p', color='#f79059')

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
    plt.plot(epochs, lambda_list[:, 0], label='lambda_u', color='#6cc2c7')
    plt.plot(epochs, lambda_list[:, 1], label='lambda_v', color='#f5ebb9')
    plt.plot(epochs, lambda_list[:, 2], label='lambda_w', color="#2B9E52")
    plt.plot(epochs, lambda_list[:, 3], label='lambda_c', color='#f5d6b1')
    plt.plot(epochs, lambda_list[:, 4], label='lambda_inlet', color='#f7bdab')
    plt.plot(epochs, lambda_list[:, 5], label='lambda_bd', color='#e0a2ae')
    plt.plot(epochs, lambda_list[:, 6], label='lambda_outlet', color='#b792b2')


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
