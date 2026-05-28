import torch
import torch.nn as nn
device = torch.device("cuda:0")

class Net_helmholtz(nn.Module):
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

def pde_loss_helmholtz(net, interior_points, Q_func=None):
    X = interior_points.clone().detach().requires_grad_(True)
    T = net(X)
    x = X[:, 0:1]
    y = X[:, 1:2]

    grads = torch.autograd.grad(T, X, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_x = grads[:, 0:1]
    T_y = grads[:, 1:2]
    T_xx = torch.autograd.grad(T_x, X, grad_outputs=torch.ones_like(T_x), create_graph=True)[0][:, 0:1]
    T_yy = torch.autograd.grad(T_y, X, grad_outputs=torch.ones_like(T_y), create_graph=True)[0][:, 1:2]
    laplace_T = T_xx + T_yy
    k_wave = 1.0
    a1 = 2
    a2 = 6
    Q_val = (k_wave**2 - (a1*torch.pi)**2 - (a2*torch.pi)**2) * torch.sin(a1*torch.pi * x) * torch.sin(a2*torch.pi * y)


    residual = laplace_T - Q_val + T
    pde_loss = (residual ** 2).mean()

    return pde_loss


def inlet_outlet_loss_helmholtz(net2, inlet_xy, outlet_xy):
    inlet_xy = inlet_xy.clone().requires_grad_(True)
    outlet_xy = outlet_xy.clone().requires_grad_(True)

    T_pred_in = net2(inlet_xy)
    T_pred_out = net2(outlet_xy)

    inlet_loss = ((T_pred_in) ** 2).mean()
    outlet_loss = ((T_pred_out) ** 2).mean()

    return inlet_loss, outlet_loss

def boundary_loss_helmholtz(net2, bd1_points, bd2_points, T_bd1, T_bd2):
    T_pred_bd1 = net2(bd1_points)
    T_pred_bd2 = net2(bd2_points)
    return ((T_pred_bd1) ** 2).mean(), ((T_pred_bd2) ** 2).mean()
