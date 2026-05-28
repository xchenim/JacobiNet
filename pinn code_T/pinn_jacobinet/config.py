from pathlib import Path

W = 0.01/2
L = 0.01/2
L = W
k = L/W
T_non = 1
seed = 99

epochs = 5000
T_max = epochs
learning_rate = 0.001
eta_min = 1e-3

lambda_T, lambda_inlet, lambda_bd, lambda_outlet = 1, 1, 1, 1

opti_mode = 'Adam'

his_freq = 10
batch_size = 0


T_bd1, T_bd2 = 1, -1


work_dir = r'T'
METHOD_DIR = Path(__file__).resolve().parent
CASE_DIR = METHOD_DIR.parent
DATA_DIR = CASE_DIR / f"groundtruth_{work_dir}"
PARAM_DIR = CASE_DIR / "parameter"
HISTORY_DIR = CASE_DIR / "history"

train_path = str(PARAM_DIR)
cfd_path = str(DATA_DIR / "export.csv")
data_path = str(DATA_DIR / "xyds.xlsx")
post_path = str(DATA_DIR / "xyds.xlsx")
save_path = str(HISTORY_DIR / "training_history_jacobinet.csv")
jacobinet_path = str(PARAM_DIR / "jacobinet.pth")
geo_net_path = jacobinet_path
net3_path = str(PARAM_DIR / "pinn_jacobinet.pth")

fttype, ftsize, ftsize_s = 'Arial', 22, 10
plt_x, plt_y = 6, 4.5
v_max = 1
vmax_error = v_max/10
