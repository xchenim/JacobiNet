import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from config import L, U_non, rho, nu, inlet_vel, Re
from models_JacobiNet import jacobinet
import numpy as np
device = torch.device("cuda:0")

class HybridNet(nn.Module):
    def __init__(self, geo_ckpt_path=None, device="cuda:0"):
        super(HybridNet, self).__init__()
        self.jacobinet = jacobinet().to(device)
        self.phys_net  = MLP().to(device)
        self.device    = device


        self.inlet_scale = float(inlet_vel) / float(U_non)

        if geo_ckpt_path is not None:
            self.load_and_freeze_geonet(geo_ckpt_path, device)

    def forward(self, xyz):
        B = xyz.shape[0]

        rds = self.jacobinet(xyz)
        r, d, s = rds[:, :1], rds[:, 1:2], rds[:, 2:3]
        x, y, z = xyz[:, :1], xyz[:, 1:2], xyz[:, 2:3]

        T = self.phys_net(rds)
        N_u, N_v, N_w, N_p = T[:, 0:1], T[:, 1:2], T[:, 2:3], T[:, 3:4]


        phi_wall = 1.0 - (r**2 + d**2)
        phi_in   = (1.0 - s) * 0.5


        u = phi_wall * (1.0 - phi_in) * N_u
        v = phi_wall * (1.0 - phi_in) * N_v
        w = phi_wall * (1.0 - phi_in) * N_w + phi_wall * phi_in * self.inlet_scale


        p = phi_in * N_p

        T = torch.cat([u, v, w, p], dim=1)
        return T, rds[:, 1:3]

    def load_and_freeze_geonet(self, geo_ckpt_path, device="cuda:0"):
        checkpoint = torch.load(geo_ckpt_path, map_location=device, weights_only=True)
        self.jacobinet.load_state_dict(checkpoint)
        self.jacobinet.eval()
        for param in self.jacobinet.parameters():
            param.requires_grad = False


class RandomFourierFeature(nn.Module):
    def __init__(self, in_dim, out_dim, sigma):
        super(RandomFourierFeature, self).__init__()
        assert out_dim % 2 == 0, "out_dim must be even"
        self.out_dim = out_dim

        B = torch.randn(out_dim // 2, in_dim) * (1.0 / sigma)
        self.register_buffer("B", B)

    def forward(self, x):

        x = x.to(self.B.device).float()
        x_proj = 2 * torch.pi * (x @ self.B.T)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class MLP(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=4, rff_dim=64, sigma=5.0):
        super(MLP, self).__init__()
        self.rff = RandomFourierFeature(in_dim=in_dim, out_dim=rff_dim, sigma=sigma)
        self.net = nn.Sequential(
            nn.Linear(rff_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        x_rff = self.rff(x)
        return self.net(x_rff)


def pde_loss(net, interior_points, i):
    X = interior_points.clone().detach().requires_grad_(True)
    outputs, _ = net(X)
    u = outputs[:, 0:1]
    v = outputs[:, 1:2]
    w = outputs[:, 2:3]
    p = outputs[:, 3:4]


    u_x = autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 0:1]
    u_y = autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 1:2]
    u_z = autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0][:, 2:3]

    v_x = autograd.grad(v, X, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0][:, 0:1]
    v_y = autograd.grad(v, X, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0][:, 1:2]
    v_z = autograd.grad(v, X, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0][:, 2:3]

    w_x = autograd.grad(w, X, grad_outputs=torch.ones_like(w), create_graph=True, retain_graph=True)[0][:, 0:1]
    w_y = autograd.grad(w, X, grad_outputs=torch.ones_like(w), create_graph=True, retain_graph=True)[0][:, 1:2]
    w_z = autograd.grad(w, X, grad_outputs=torch.ones_like(w), create_graph=True, retain_graph=True)[0][:, 2:3]

    p_x = autograd.grad(p, X, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0][:, 0:1]
    p_y = autograd.grad(p, X, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0][:, 1:2]
    p_z = autograd.grad(p, X, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0][:, 2:3]


    u_xx = autograd.grad(u_x, X, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    u_yy = autograd.grad(u_y, X, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0][:, 1:2]
    u_zz = autograd.grad(u_z, X, grad_outputs=torch.ones_like(u_z), create_graph=True, retain_graph=True)[0][:, 2:3]

    v_xx = autograd.grad(v_x, X, grad_outputs=torch.ones_like(v_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    v_yy = autograd.grad(v_y, X, grad_outputs=torch.ones_like(v_y), create_graph=True, retain_graph=True)[0][:, 1:2]
    v_zz = autograd.grad(v_z, X, grad_outputs=torch.ones_like(v_z), create_graph=True, retain_graph=True)[0][:, 2:3]

    w_xx = autograd.grad(w_x, X, grad_outputs=torch.ones_like(w_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    w_yy = autograd.grad(w_y, X, grad_outputs=torch.ones_like(w_y), create_graph=True, retain_graph=True)[0][:, 1:2]
    w_zz = autograd.grad(w_z, X, grad_outputs=torch.ones_like(w_z), create_graph=True, retain_graph=True)[0][:, 2:3]


    ns_u = u * u_x + v * u_y + w * u_z + p_x - (1.0 / Re) * (u_xx + u_yy + u_zz)
    ns_v = u * v_x + v * v_y + w * v_z + p_y - (1.0 / Re) * (v_xx + v_yy + v_zz)
    ns_w = u * w_x + v * w_y + w * w_z + p_z - (1.0 / Re) * (w_xx + w_yy + w_zz)


    continuity = u_x + v_y + w_z


    pde_loss_u = (ns_u ** 2).mean()
    pde_loss_v = (ns_v ** 2).mean()
    pde_loss_w = (ns_w ** 2).mean()


    continuity_loss = (continuity ** 2).mean()
    return pde_loss_u, pde_loss_v, pde_loss_w, continuity_loss

def inlet_outlet_loss(net, inlet_points, outlet_points, inlet_velocity):
    outputs_inlet, _ = net(inlet_points)
    u_inlet = outputs_inlet[:, 0:1]
    v_inlet = outputs_inlet[:, 1:2]
    w_inlet = outputs_inlet[:, 2:3]

    outputs_outlet, _ = net(outlet_points)
    p_outlet = outputs_outlet[:, 3:4]

    inlet_vel_pred = torch.cat((u_inlet, v_inlet, w_inlet), dim=1)
    inlet_loss = ((inlet_vel_pred - inlet_velocity) ** 2).mean()


    outlet_loss = (p_outlet ** 2).mean()
    return inlet_loss, outlet_loss


def boundary_loss(net, bd_points):
    outputs_bd, _ = net(bd_points)

    u_v_w_bd = outputs_bd[:, 0:3]

    bd1_loss = (u_v_w_bd ** 2).mean()
    return bd1_loss
