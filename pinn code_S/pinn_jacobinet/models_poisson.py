import torch
import torch.nn as nn
from models_JacobiNet import jacobinet
device = torch.device("cuda:0")


class HybridNet(nn.Module):
    def __init__(self, geo_ckpt_path=None, device="cuda:0"):
        super(HybridNet, self).__init__()
        self.jacobinet = jacobinet().to(device)
        self.poisson_net = Net_poisson().to(device)

        if geo_ckpt_path is not None:
            self.load_and_freeze_jacobinet(geo_ckpt_path, device)

    def forward(self, xy):
        ds = self.jacobinet(xy)
        T = self.poisson_net(ds)
        d = ds[:, 0:1]
        s = ds[:, 1:2]
        T = (1 - d**2) * (1 - s**2) * T

        return T, ds

    def load_and_freeze_jacobinet(self, geo_ckpt_path, device="cuda:0"):
        checkpoint = torch.load(geo_ckpt_path, map_location=device, weights_only=True)
        self.jacobinet.load_state_dict(checkpoint["state_dict"])
        self.jacobinet.eval()
        for param in self.jacobinet.parameters():
            param.requires_grad = False


class Net_poisson(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


def pde_loss_poisson(net3, interior_points):
    X = interior_points.clone().detach().requires_grad_(True)
    T, _ = net3(X)
    x = X[:, 0:1]
    y = X[:, 1:2]
    grads = torch.autograd.grad(T, X, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_x = grads[:, 0:1]
    T_y = grads[:, 1:2]
    T_xx = torch.autograd.grad(T_x, X, grad_outputs=torch.ones_like(T_x), create_graph=True)[0][:, 0:1]
    T_yy = torch.autograd.grad(T_y, X, grad_outputs=torch.ones_like(T_y), create_graph=True)[0][:, 1:2]
    laplace_T = T_xx + T_yy

    Q_val = -2 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

    residual = laplace_T - Q_val
    pde_loss = (residual ** 2).mean()

    return pde_loss

def inlet_outlet_loss_poisson(net3, inlet_xy, outlet_xy):
    inlet_xy = inlet_xy.clone().requires_grad_(True)
    outlet_xy = outlet_xy.clone().requires_grad_(True)

    T_pred_in, _ = net3(inlet_xy)
    T_pred_out, _ = net3(outlet_xy)

    return ((T_pred_in) ** 2).mean(), ((T_pred_out) ** 2).mean()


def boundary_loss_poisson(net2, bd1_points, bd2_points):
    T_pred_bd1, _ = net2(bd1_points)
    T_pred_bd2, _ = net2(bd2_points)
    return ((T_pred_bd1) ** 2).mean(), ((T_pred_bd2) ** 2).mean()
