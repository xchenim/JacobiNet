import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from scipy.interpolate import griddata
import os
from pathlib import Path
device = torch.device("cuda:0")
from config import L, U_non, P_non, data_path, cfd_path, post_path, net3_path, v_max, vmax_error, p_max, pmax_error, plt_x, plt_y
from data_loader import load_cfd_data, load_train_data, read_groundtruth_csv, save_history_data, load_origin_data
from models_NS import HybridNet


def denormalize_points(points, mean_vals, scale_factor):
    denormalized_points = points * scale_factor + mean_vals
    return denormalized_points


def plot_cfd_velocity_pressure(x_cfd, y_cfd, z_cfd, u_cfd, v_cfd, w_cfd, p_cfd,x0=0.0, tol=None, v_max=v_max, p_max=None, s=2):

    x_min, x_max = np.nanmin(x_cfd), np.nanmax(x_cfd)
    if tol is None:
        rng = (x_max - x_min)
        tol = 0 if rng == 0 else thickness * rng

    mask = np.isclose(x_cfd, x0, atol=tol)
    if not np.any(mask):

        idx = np.argmin(np.abs(x_cfd - x0))
        x_star = x_cfd[idx]
        mask = np.isclose(x_cfd, x_star, atol=tol)

    Y, Z = y_cfd[mask], z_cfd[mask]
    U, V, W, P = u_cfd[mask], v_cfd[mask], w_cfd[mask], p_cfd[mask]

    vel_mag = np.sqrt(U**2 + V**2 + W**2)
    if v_max is None:
        v_max = np.nanmax(vel_mag) if np.isfinite(vel_mag).any() else 1.0
    if p_max is None:
        p_max = np.nanmax(P) if np.isfinite(P).any() else 1.0


    fig, ax = plt.subplots(figsize=(plt_x, plt_y))
    sc = ax.scatter(Y, Z, c=vel_mag, cmap='coolwarm', s=s, vmin=0, vmax=v_max)
    plt.colorbar(sc, ax=ax, orientation='vertical', label='Velocity Magnitude')
    ax.set_xlabel('Y'); ax.set_ylabel('Z')
    ax.set_title('CFD Velocity Magnitude on YZ @ x≈0')
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots(figsize=(plt_x, plt_y))
    sc = ax.scatter(Y, Z, c=P, cmap='coolwarm', s=s, vmin=0, vmax=p_max)
    plt.colorbar(sc, ax=ax, orientation='vertical', label='Pressure')
    ax.set_xlabel('Y'); ax.set_ylabel('Z')
    ax.set_title('CFD Pressure on YZ @ x≈0')
    ax.set_aspect('equal', 'box')
    plt.tight_layout()
    plt.show()


def plot_velocity_magnitude(net3, points_list, min_vals, max_vals, x0=0.0, tol=None, vec_scale=2000, s=1):
    net3.eval()
    with torch.no_grad():
        all_points = torch.cat(points_list, dim=0)
        outputs,_ = net3(all_points)
        u = outputs[:, 0].cpu().numpy()
        v = outputs[:, 1].cpu().numpy()
        w = outputs[:, 2].cpu().numpy()


        mean_vals = (min_vals + max_vals) / 2.0
        x = denormalize_points(all_points[:, 0], mean_vals[:, 0], L).cpu().numpy()
        y = denormalize_points(all_points[:, 1], mean_vals[:, 1], L).cpu().numpy()
        z = denormalize_points(all_points[:, 2], mean_vals[:, 2], L).cpu().numpy()


    x_min, x_max = np.nanmin(x), np.nanmax(x)
    if tol is None:
        rng = (x_max - x_min)
        tol = 1e-8 if rng == 0 else thickness * rng

    mask = np.isclose(x, x0, atol=tol)
    if not np.any(mask):
        idx = np.argmin(np.abs(x - x0))
        x_star = x[idx]
        mask = np.isclose(x, x_star, atol=tol)

    Y, Z = y[mask], z[mask]
    V, W = v[mask], w[mask]
    mag = np.sqrt(u[mask]**2 + V**2 + W**2)


    fig, ax = plt.subplots(figsize=(plt_x, plt_y))
    ax.quiver(Y, Z, V, W, angles='xy', scale_units='xy', scale=vec_scale, width=0.001)
    ax.set_title('Velocity Vectors on YZ @ x≈0')
    ax.set_xlabel('y'); ax.set_ylabel('z')
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    plt.tight_layout()
    plt.show()


def plot_velocity_pressure(net3, points_list, min_vals, max_vals, x0=0.0, tol=None):
    net3.eval()
    with torch.no_grad():

        all_points = torch.cat(points_list, dim=0)
        outputs,_ = net3(all_points)
        u = outputs[:, 0].cpu().numpy()
        v = outputs[:, 1].cpu().numpy()
        w = outputs[:, 2].cpu().numpy()
        p = outputs[:, 3].cpu().numpy()


        mean_vals = (min_vals + max_vals) / 2.0
        x = denormalize_points(all_points[:, 0], mean_vals[:, 0], L).cpu().numpy()
        y = denormalize_points(all_points[:, 1], mean_vals[:, 1], L).cpu().numpy()
        z = denormalize_points(all_points[:, 2], mean_vals[:, 2], L).cpu().numpy()


        x_min, x_max = np.nanmin(x), np.nanmax(x)
        if tol is None:
            rng = (x_max - x_min)
            tol = 1e-8 if rng == 0 else thickness * rng

        mask = np.isclose(x, x0, atol=tol)
        if not np.any(mask):
            idx = np.argmin(np.abs(x - x0))
            x_star = x[idx]
            mask = np.isclose(x, x_star, atol=tol)

        X, Y, Z = x[mask], y[mask], z[mask]
        U, V, W, P = u[mask], v[mask], w[mask], p[mask]

        mag = np.sqrt(U**2 + V**2 + W**2)

        print(f"Max velocity magnitude: {mag.max() * U_non:.6f}")
        print(f"Mean velocity magnitude: {mag.mean() * U_non:.6f}")
        print(f"Max |u|: {np.abs(U).max() * U_non:.6f}, Mean |u|: {np.abs(U).mean() * U_non:.6f}")
        print(f"Max |v|: {np.abs(V).max() * U_non:.6f}, Mean |v|: {np.abs(V).mean() * U_non:.6f}")
        print(f"Max |w|: {np.abs(W).max() * U_non:.6f}, Mean |w|: {np.abs(W).mean() * U_non:.6f}")


    fig, ax = plt.subplots(figsize=(plt_x, plt_y))
    sc = ax.scatter(Y, Z, c=mag * U_non, cmap='coolwarm', s=1, vmin=0, vmax=1)
    ax.set_aspect('equal', 'box'); plt.tight_layout(); plt.show()


    fig, ax = plt.subplots(figsize=(plt_x, plt_y))
    sc = ax.scatter(Y, Z, c=np.abs(P) * P_non, cmap='coolwarm', s=1, vmin=-100, vmax=700)
    ax.set_aspect('equal', 'box'); plt.tight_layout(); plt.show()


def plot_error(
    x_cfd, y_cfd, z_cfd, u_cfd, v_cfd, w_cfd, p_cfd,
    net3, points_list, min_vals, max_vals,
    x0=0.0, tol=None, method='linear'
):
    net3.eval()
    with torch.no_grad():

        all_points = torch.cat(points_list, dim=0).to(next(net3.parameters()).device)
        outputs,_ = net3(all_points)
        u_pred = outputs[:, 0].cpu().numpy() * U_non
        v_pred = outputs[:, 1].cpu().numpy() * U_non
        w_pred = outputs[:, 2].cpu().numpy() * U_non
        p_pred = outputs[:, 3].cpu().numpy() * P_non
        vel_pred = (u_pred**2 + v_pred**2 + w_pred**2) ** 0.5


        mean_vals = (min_vals + max_vals) / 2.0
        x_pred = denormalize_points(all_points[:, 0], mean_vals[:, 0], L).cpu().numpy()
        y_pred = denormalize_points(all_points[:, 1], mean_vals[:, 1], L).cpu().numpy()
        z_pred = denormalize_points(all_points[:, 2], mean_vals[:, 2], L).cpu().numpy()

        x_min, x_max = np.nanmin(x_pred), np.nanmax(x_pred)
        if tol is None:
            rng = (x_max - x_min)
            tol = 1e-8 if rng == 0 else thickness * rng

        m = np.isclose(x_pred, x0, atol=tol)
        if not np.any(m):
            i0 = np.argmin(np.abs(x_pred - x0))
            x_star = x_pred[i0]
            m = np.isclose(x_pred, x_star, atol=tol)

        yq, zq = y_pred[m], z_pred[m]
        up, vp, wp, pp, vmagp = u_pred[m], v_pred[m], w_pred[m], p_pred[m], vel_pred[m]


    mc = np.isclose(x_cfd, x0, atol=tol)
    if not np.any(mc):
        j0 = np.argmin(np.abs(x_cfd - x0))
        x_star_c = x_cfd[j0]
        mc = np.isclose(x_cfd, x_star_c, atol=tol)
    yc, zc = y_cfd[mc], z_cfd[mc]
    uc, vc, wc, pc = u_cfd[mc], v_cfd[mc], w_cfd[mc], p_cfd[mc]
    vmagc = (uc**2 + vc**2 + wc**2) ** 0.5

    def interp(qy, qz, vals):
        pts = np.c_[yc, zc]; qry = np.c_[qy, qz]
        out = griddata(pts, vals, qry, method=method)
        if np.isnan(out).any():
            out_nn = griddata(pts, vals, qry, method='nearest')
            out = np.where(np.isnan(out), out_nn, out)
        return out

    u_gt = interp(yq, zq, uc)
    v_gt = interp(yq, zq, vc)
    w_gt = interp(yq, zq, wc)
    p_gt = interp(yq, zq, pc)
    vmag_gt = interp(yq, zq, vmagc)


    err_u   = np.abs(up - u_gt)
    err_v   = np.abs(vp - v_gt)
    err_w   = np.abs(wp - w_gt)
    err_p   = np.abs(pp - p_gt)
    err_vel = np.abs(vmagp - vmag_gt)


    def safe_mean(a):
        a = np.where(np.isfinite(a), a, 0.0); return np.mean(a)
    def safe_rms(a):
        a = np.where(np.isfinite(a), a, 0.0); return np.sqrt(np.mean(a*a) + 1e-18)


    L1_u_abs, L1_v_abs, L1_w_abs, L1_p_abs, L1_vel_abs = map(safe_mean, [err_u, err_v, err_w, err_p, err_vel])
    L2_u_abs, L2_v_abs, L2_w_abs, L2_p_abs, L2_vel_abs = map(safe_rms,  [up-u_gt, vp-v_gt, wp-w_gt, pp-p_gt, vmagp-vmag_gt])


    L1_u_rel = L1_u_abs   / (safe_mean(np.abs(u_gt))    + 1e-18)
    L1_v_rel = L1_v_abs   / (safe_mean(np.abs(v_gt))    + 1e-18)
    L1_w_rel = L1_w_abs   / (safe_mean(np.abs(w_gt))    + 1e-18)
    L1_p_rel = L1_p_abs   / (safe_mean(np.abs(p_gt))    + 1e-18)
    L1_vel_rel = L1_vel_abs / (safe_mean(np.abs(vmag_gt)) + 1e-18)

    L2_u_rel = L2_u_abs   / (safe_rms(u_gt)    + 1e-18)
    L2_v_rel = L2_v_abs   / (safe_rms(v_gt)    + 1e-18)
    L2_w_rel = L2_w_abs   / (safe_rms(w_gt)    + 1e-18)
    L2_p_rel = L2_p_abs   / (safe_rms(p_gt)    + 1e-18)
    L2_vel_rel = L2_vel_abs / (safe_rms(vmag_gt) + 1e-18)


    L1_abs_total = np.sqrt(L1_u_abs**2 + L1_v_abs**2 + L1_w_abs**2 + L1_p_abs**2)
    L1_rel_total = np.sqrt(L1_u_rel**2 + L1_v_rel**2 + L1_w_rel**2 + L1_p_rel**2)
    L2_abs_total = np.sqrt(L2_u_abs**2 + L2_v_abs**2 + L2_w_abs**2 + L2_p_abs**2)
    L2_rel_total = np.sqrt(L2_u_rel**2 + L2_v_rel**2 + L2_w_rel**2 + L2_p_rel**2)


    print("=== SLICE ===")
    print(f"L1_vel  abs: {L1_vel_abs:.3e}  rel: {L1_vel_rel:.3e}")
    print(f"L1_u    abs: {L1_u_abs:.3e}    rel: {L1_u_rel:.3e}")
    print(f"L1_v    abs: {L1_v_abs:.3e}    rel: {L1_v_rel:.3e}")
    print(f"L1_w    abs: {L1_w_abs:.3e}    rel: {L1_w_rel:.3e}")
    print(f"L1_p    abs: {L1_p_abs:.3e}    rel: {L1_p_rel:.3e}")
    print(f"L1_total: {L1_abs_total:.3e}   L1_rel_total: {L1_rel_total:.3e}")

    print(f"L2_vel  abs: {L2_vel_abs:.3e}  rel: {L2_vel_rel:.3e}")
    print(f"L2_u    abs: {L2_u_abs:.3e}    rel: {L2_u_rel:.3e}")
    print(f"L2_v    abs: {L2_v_abs:.3e}    rel: {L2_v_rel:.3e}")
    print(f"L2_w    abs: {L2_w_abs:.3e}    rel: {L2_w_rel:.3e}")
    print(f"L2_p    abs: {L2_p_abs:.3e}    rel: {L2_p_rel:.3e}")
    print(f"L2_total: {L2_abs_total:.3e}   L2_rel_total: {L2_rel_total:.3e}")


    fig, axs = plt.subplots(1, 5, figsize=(plt_x*4, plt_y*2))
    sc0 = axs[0].scatter(yq, zq, c=err_vel, cmap='coolwarm', s=1, vmin=0, vmax=vmax_error)
    axs[0].set_title('Error |U|'); axs[0].set_aspect('equal', 'box'); axs[0].axis('off')

    sc1 = axs[1].scatter(yq, zq, c=err_u, cmap='coolwarm', s=1, vmin=0, vmax=vmax_error)
    axs[1].set_title('Error u'); axs[1].set_aspect('equal', 'box'); axs[1].axis('off')

    sc2 = axs[2].scatter(yq, zq, c=err_v, cmap='coolwarm', s=1, vmin=0, vmax=vmax_error)
    axs[2].set_title('Error v'); axs[2].set_aspect('equal', 'box'); axs[2].axis('off')

    sc3 = axs[3].scatter(yq, zq, c=err_w, cmap='coolwarm', s=1, vmin=0, vmax=vmax_error)
    axs[3].set_title('Error w'); axs[3].set_aspect('equal', 'box'); axs[3].axis('off')


    sc4 = axs[4].scatter(yq, zq, c=err_p, cmap='coolwarm', s=1, vmin=0, vmax=pmax_error)
    axs[4].set_title('Error p'); axs[4].set_aspect('equal', 'box'); axs[4].axis('off')

    plt.tight_layout(); plt.show()

def plot_error_Z(
    x_cfd, y_cfd, z_cfd, u_cfd, v_cfd, w_cfd, p_cfd,
    net3, points_list, min_vals, max_vals,
    z0=0.0, tol=None, method='linear'
):
    net3.eval()
    with torch.no_grad():

        all_points = torch.cat(points_list, dim=0).to(next(net3.parameters()).device)
        outputs, _ = net3(all_points)
        u_pred = outputs[:, 0].cpu().numpy() * U_non
        v_pred = outputs[:, 1].cpu().numpy() * U_non
        w_pred = outputs[:, 2].cpu().numpy() * U_non
        p_pred = outputs[:, 3].cpu().numpy() * P_non
        vel_pred = (u_pred**2 + v_pred**2 + w_pred**2) ** 0.5


        mean_vals = (min_vals + max_vals) / 2.0
        x_pred = denormalize_points(all_points[:, 0], mean_vals[:, 0], L).cpu().numpy()
        y_pred = denormalize_points(all_points[:, 1], mean_vals[:, 1], L).cpu().numpy()
        z_pred = denormalize_points(all_points[:, 2], mean_vals[:, 2], L).cpu().numpy()

        z_min, z_max = np.nanmin(z_pred), np.nanmax(z_pred)
        if tol is None:
            rng = (z_max - z_min)
            tol = 1e-8 if rng == 0 else thickness * rng

        m = np.isclose(z_pred, z0, atol=tol)
        if not np.any(m):
            i0 = np.argmin(np.abs(z_pred - z0))
            z_star = z_pred[i0]
            m = np.isclose(z_pred, z_star, atol=tol)

        xq, yq = x_pred[m], y_pred[m]
        up, vp, wp, pp, vmagp = u_pred[m], v_pred[m], w_pred[m], p_pred[m], vel_pred[m]


    mc = np.isclose(z_cfd, z0, atol=tol)
    if not np.any(mc):
        j0 = np.argmin(np.abs(z_cfd - z0))
        z_star_c = z_cfd[j0]
        mc = np.isclose(z_cfd, z_star_c, atol=tol)
    xc, yc = x_cfd[mc], y_cfd[mc]
    uc, vc, wc, pc = u_cfd[mc], v_cfd[mc], w_cfd[mc], p_cfd[mc]
    vmagc = (uc**2 + vc**2 + wc**2) ** 0.5

    def interp(qx, qy, vals):
        pts = np.c_[xc, yc]; qry = np.c_[qx, qy]
        out = griddata(pts, vals, qry, method=method)
        if np.isnan(out).any():
            out_nn = griddata(pts, vals, qry, method='nearest')
            out = np.where(np.isnan(out), out_nn, out)
        return out

    u_gt = interp(xq, yq, uc)
    v_gt = interp(xq, yq, vc)
    w_gt = interp(xq, yq, wc)
    p_gt = interp(xq, yq, pc)
    vmag_gt = interp(xq, yq, vmagc)


    du   = up - u_gt
    dv   = vp - v_gt
    dw   = wp - w_gt
    dp   = pp - p_gt
    dvel = vmagp - vmag_gt


    def l2_abs(a, b): return np.sqrt(np.mean((a-b)**2))
    def l2_rel(a, b): return l2_abs(a, b) / (np.sqrt(np.mean(b**2)) + 1e-18)

    L2_u_abs, L2_v_abs, L2_w_abs, L2_p_abs, L2_vel_abs =\
        [l2_abs(pred, gt) for pred, gt in [(up,u_gt),(vp,v_gt),(wp,w_gt),(pp,p_gt),(vmagp,vmag_gt)]]
    L2_u_rel, L2_v_rel, L2_w_rel, L2_p_rel, L2_vel_rel =\
        [l2_rel(pred, gt) for pred, gt in [(up,u_gt),(vp,v_gt),(wp,w_gt),(pp,p_gt),(vmagp,vmag_gt)]]


    L2_abs_total = np.sqrt(L2_u_abs**2 + L2_v_abs**2 + L2_w_abs**2 + L2_p_abs**2)
    L2_rel_total = np.sqrt(L2_u_rel**2 + L2_v_rel**2 + L2_w_rel**2 + L2_p_rel**2)

    print(f"=== SLICE z≈{z0:g} ===")
    print(f"L2_vel abs: {L2_vel_abs:.3e}  rel: {L2_vel_rel:.3e}")
    print(f"L2_u   abs: {L2_u_abs:.3e}   rel: {L2_u_rel:.3e}")
    print(f"L2_v   abs: {L2_v_abs:.3e}   rel: {L2_v_rel:.3e}")
    print(f"L2_w   abs: {L2_w_abs:.3e}   rel: {L2_w_rel:.3e}")
    print(f"L2_p   abs: {L2_p_abs:.3e}   rel: {L2_p_rel:.3e}")
    print(f"L2_total abs: {L2_abs_total:.3e}   rel: {L2_rel_total:.3e}")


    fig, axs = plt.subplots(1, 5, figsize=(plt_x*4, plt_y*2))
    axs[0].scatter(xq, yq, c=np.abs(dvel), cmap='coolwarm', s=1, vmin=0, vmax=vmax_error)
    axs[0].set_title('Error |U|'); axs[0].set_aspect('equal', 'box'); axs[0].axis('off')

    axs[1].scatter(xq, yq, c=np.abs(du), cmap='coolwarm', s=1, vmin=0, vmax=vmax_error)
    axs[1].set_title('Error u'); axs[1].set_aspect('equal', 'box'); axs[1].axis('off')

    axs[2].scatter(xq, yq, c=np.abs(dv), cmap='coolwarm', s=1, vmin=0, vmax=vmax_error)
    axs[2].set_title('Error v'); axs[2].set_aspect('equal', 'box'); axs[2].axis('off')

    axs[3].scatter(xq, yq, c=np.abs(dw), cmap='coolwarm', s=1, vmin=0, vmax=vmax_error)
    axs[3].set_title('Error w'); axs[3].set_aspect('equal', 'box'); axs[3].axis('off')

    axs[4].scatter(xq, yq, c=np.abs(dp), cmap='coolwarm', s=1, vmin=0, vmax=pmax_error)
    axs[4].set_title('Error p'); axs[4].set_aspect('equal', 'box'); axs[4].axis('off')

    plt.tight_layout(); plt.show()


import numpy as np
import torch
from config import L, U_non, P_non

def print_l2_error_3d_full(
    net3,
    x_cfd, y_cfd, z_cfd,
    u_cfd, v_cfd, w_cfd, p_cfd,
    mean_vals,
    verbose=True
):

    device = next(net3.parameters()).device
    net3.eval()


    x = np.asarray(x_cfd); y = np.asarray(y_cfd); z = np.asarray(z_cfd)
    u = np.asarray(u_cfd); v = np.asarray(v_cfd); w = np.asarray(w_cfd); p = np.asarray(p_cfd)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) &\
        np.isfinite(u) & np.isfinite(v) & np.isfinite(w) & np.isfinite(p)
    x, y, z, u, v, w, p = x[m], y[m], z[m], u[m], v[m], w[m], p[m]


    pts = torch.tensor(np.vstack((x, y, z)).T, dtype=torch.float32, device=device)

    mv = mean_vals.to(device)
    pts[:, 0] = (pts[:, 0] - mv[:, 0]) / L
    pts[:, 1] = (pts[:, 1] - mv[:, 1]) / L
    pts[:, 2] = (pts[:, 2] - mv[:, 2]) / L

    with torch.no_grad():
        out, _ = net3(pts)
        up = (out[:, 0].detach().cpu().numpy()) * U_non
        vp = (out[:, 1].detach().cpu().numpy()) * U_non
        wp = (out[:, 2].detach().cpu().numpy()) * U_non
        pp = (out[:, 3].detach().cpu().numpy()) * P_non


    vmag_gt = np.sqrt(u**2 + v**2 + w**2)
    vmag_p  = np.sqrt(up**2 + vp**2 + wp**2)


    def l2_abs(a, b):
        return np.sqrt(np.mean((a - b)**2))
    def l2_rel(a, b):
        denom = np.sqrt(np.mean(b**2)) + 1e-18
        return l2_abs(a, b) / denom


    metrics = {
        "L2_abs_u":   l2_abs(up, u),
        "L2_abs_v":   l2_abs(vp, v),
        "L2_abs_w":   l2_abs(wp, w),
        "L2_abs_p":   l2_abs(pp, p),
        "L2_abs_vel": l2_abs(vmag_p, vmag_gt),

        "RL2_u":   l2_rel(up, u),
        "RL2_v":   l2_rel(vp, v),
        "RL2_w":   l2_rel(wp, w),
        "RL2_p":   l2_rel(pp, p),
        "RL2_vel": l2_rel(vmag_p, vmag_gt),
    }


    if verbose:
        print("=== 3D TOTAL ===")
        print(f"L2_vel  abs: {metrics['L2_abs_vel']:.3e}  rel: {metrics['RL2_vel']:.3e}")
        print(f"L2_u    abs: {metrics['L2_abs_u']:.3e}    rel: {metrics['RL2_u']:.3e}")
        print(f"L2_v    abs: {metrics['L2_abs_v']:.3e}    rel: {metrics['RL2_v']:.3e}")
        print(f"L2_w    abs: {metrics['L2_abs_w']:.3e}    rel: {metrics['RL2_w']:.3e}")
        print(f"L2_p    abs: {metrics['L2_abs_p']:.3e}    rel: {metrics['RL2_p']:.3e}")


        L2_abs_total = sum([metrics[k] for k in ['L2_abs_u','L2_abs_v','L2_abs_w','L2_abs_p']])
        L2_rel_total = sum([metrics[k] for k in ['RL2_u','RL2_v','RL2_w','RL2_p']])
        print(f"L2_total abs: {L2_abs_total:.3e}   rel: {L2_rel_total:.3e}")

    return metrics


import numpy as np

def export_to_vtk(filename, x, y, z, u, v, w, p):
    N = len(x)
    with open(filename, "w") as f:

        f.write("# vtk DataFile Version 3.0\n")
        f.write("CFD/NN results\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {N} float\n")


        for xi, yi, zi in zip(x, y, z):
            f.write(f"{xi} {yi} {zi}\n")


        f.write(f"\nPOINT_DATA {N}\n")


        f.write("VECTORS velocity float\n")
        for ui, vi, wi in zip(u, v, w):
            f.write(f"{ui} {vi} {wi}\n")


        f.write("SCALARS pressure float 1\n")
        f.write("LOOKUP_TABLE default\n")
        for pi in p:
            f.write(f"{pi}\n")


import pandas as pd
import torch

def load_post_data(data_path):

    data_dir = Path(data_path).resolve().parent
    internal_df = read_groundtruth_csv(data_dir / "internal.csv")
    inlet_df    = read_groundtruth_csv(data_dir / "inlet.csv")
    outlet_df   = read_groundtruth_csv(data_dir / "outlet.csv")
    bd_df       = read_groundtruth_csv(data_dir / "bd.csv")


    encrypted_points = torch.tensor(internal_df[['x', 'y', 'z']].values, dtype=torch.float32).to(device)
    inlet_points     = torch.tensor(inlet_df[['x', 'y', 'z']].values, dtype=torch.float32).to(device)
    outlet_points    = torch.tensor(outlet_df[['x', 'y', 'z']].values, dtype=torch.float32).to(device)
    bd_points        = torch.tensor(bd_df[['x', 'y', 'z']].values, dtype=torch.float32).to(device)


    original_points = torch.cat((encrypted_points, inlet_points, bd_points, outlet_points), dim=0)
    min_vals = original_points.min(dim=0, keepdim=True)[0]
    max_vals = original_points.max(dim=0, keepdim=True)[0]
    mean_vals = (min_vals + max_vals) / 2


    def normalize(points):
        points[:, 0] = (points[:, 0] - mean_vals[:, 0]) / L
        points[:, 1] = (points[:, 1] - mean_vals[:, 1]) / L
        points[:, 2] = (points[:, 2] - mean_vals[:, 2]) / L
        return points

    encrypted_points = normalize(encrypted_points)
    inlet_points     = normalize(inlet_points)
    outlet_points    = normalize(outlet_points)
    bd_points        = normalize(bd_points)

    all_points = torch.cat((encrypted_points, inlet_points, bd_points, outlet_points), dim=0)

    return all_points, encrypted_points, inlet_points, outlet_points, bd_points, mean_vals, min_vals, max_vals


if __name__ == "__main__":
    thickness = 0.005
    all_points, encrypted_points, inlet_points, outlet_points, bd_points, mean_vals, min_vals, max_vals = load_post_data(post_path)
    x_cfd, y_cfd, z_cfd, velocity_cfd, u_cfd, v_cfd, w_cfd, p_cfd = load_cfd_data(cfd_path)
    points_list = [encrypted_points, inlet_points, outlet_points, bd_points]


    net3 = HybridNet().to(device)
    state_dict = torch.load(net3_path, map_location=device, weights_only=True)
    net3.load_state_dict(state_dict)


    plot_error_Z(x_cfd, y_cfd, z_cfd, u_cfd, v_cfd, w_cfd, p_cfd, net3, points_list, min_vals, max_vals)


    metrics = print_l2_error_3d_full(net3, x_cfd, y_cfd, z_cfd, u_cfd, v_cfd, w_cfd, p_cfd, mean_vals, verbose=True)
