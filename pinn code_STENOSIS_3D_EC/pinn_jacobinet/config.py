from pathlib import Path

R = 0.00469/2
L = 0.00469/2

inlet_vel = 0.50

U_non = inlet_vel/2
rho = 1060
nu = 0.0035/rho
P_non = rho*U_non**2
Re = U_non * L / nu

seed = 99


epochs = 50000


T_max = epochs
learning_rate = 1e-3
eta_min = 1e-5

lambda_u, lambda_v, lambda_w, lambda_c, lambda_inlet, lambda_bd, lambda_outlet = 1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7
opti_mode = 'Adam'


his_freq = 10
batch_size = 0


work_dir = r'STENOSIS_3D_EC'
METHOD_DIR = Path(__file__).resolve().parent
CASE_DIR = METHOD_DIR.parent
DATA_DIR = CASE_DIR / f"groundtruth_{work_dir}"
PARAM_DIR = CASE_DIR / "parameter"
HISTORY_DIR = CASE_DIR / "history"

train_path = str(PARAM_DIR)
cfd_path = str(DATA_DIR / "export_stenosis.csv")
data_path = str(DATA_DIR / "stenosis_n.xlsx")
post_path = str(DATA_DIR / "stenosis_post.xlsx")
save_path = str(HISTORY_DIR / "training_history_jacobinet.csv")
jacobinet_path = str(PARAM_DIR / "jacobinet.pth")
geo_net_path = jacobinet_path
net3_path = str(PARAM_DIR / "pinn_jacobinet.pth")

fttype, ftsize, ftsize_s = 'Arial', 22, 10
plt_x, plt_y = 6, 4.5


v_max = 1
p_max = 700


vmax_error = v_max/5
pmax_error = 160
