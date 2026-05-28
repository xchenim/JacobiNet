import torch
import torch.nn as nn
device = torch.device("cuda:0")

class Net_laplace(nn.Module):
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

def pde_loss_laplace(net, interior_points, Q_func=None):
    X = interior_points.clone().detach().requires_grad_(True)
    T = net(X)

    grads = torch.autograd.grad(T, X, grad_outputs=torch.ones_like(T), create_graph=True)[0]
    T_x = grads[:, 0:1]
    T_y = grads[:, 1:2]
    T_xx = torch.autograd.grad(T_x, X, grad_outputs=torch.ones_like(T_x), create_graph=True)[0][:, 0:1]
    T_yy = torch.autograd.grad(T_y, X, grad_outputs=torch.ones_like(T_y), create_graph=True)[0][:, 1:2]
    laplace_T = T_xx + T_yy

    pde_loss = (laplace_T ** 2).mean()

    return pde_loss


def inlet_outlet_loss_laplace(net2, inlet_xy, outlet_xy):
    inlet_xy = inlet_xy.clone().detach().requires_grad_(True)
    outlet_xy = outlet_xy.clone().detach().requires_grad_(True)

    T_pred_in = net2(inlet_xy)
    T_pred_out = net2(outlet_xy)


    grad_T_in = torch.autograd.grad(
        outputs=T_pred_in,
        inputs=inlet_xy,
        grad_outputs=torch.ones_like(T_pred_in),
        create_graph=True
    )[0]

    grad_T_out = torch.autograd.grad(
        outputs=T_pred_out,
        inputs=outlet_xy,
        grad_outputs=torch.ones_like(T_pred_out),
        create_graph=True
    )[0]


    n_in = torch.tensor([[1.0, 0.0]], device=inlet_xy.device).repeat(len(inlet_xy), 1)
    n_out = torch.tensor([[-1.0, 0.0]], device=outlet_xy.device).repeat(len(outlet_xy), 1)


    neumann_in = (grad_T_in * n_in).sum(dim=1, keepdim=True)
    neumann_out = (grad_T_out * n_out).sum(dim=1, keepdim=True)

    inlet_loss = (neumann_in ** 2).mean()
    outlet_loss = (neumann_out ** 2).mean()

    return inlet_loss, outlet_loss

def boundary_loss_laplace(net2, bd1_points, bd2_points, T_bd1, T_bd2):
    T_pred_bd1 = net2(bd1_points)
    T_pred_bd2 = net2(bd2_points)
    return ((T_pred_bd1 - T_bd1) ** 2).mean(), ((T_pred_bd2 - T_bd2) ** 2).mean()
