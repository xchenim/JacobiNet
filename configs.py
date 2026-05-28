from __future__ import annotations

from dataclasses import dataclass


REYNOLDS_VALUES = ["10", "100", "300", "500", "1000"]


PINN_CASES = {
    "S": {
        "code_dir": "pinn code_S",
        "data_dir": "groundtruth_S",
        "methods": {
            "baseline": {
                "method_dir": "pinn_baseline",
                "train_script": "main.py",
                "eval_script": "post.py",
            },
            "jacobinet": {
                "method_dir": "pinn_jacobinet",
                "train_script": "main.py",
                "eval_script": "post.py",
            },
        },
        "required_files": [
            "groundtruth_S/export.csv",
            "groundtruth_S/xyds.xlsx",
            "parameter/pinn_baseline.pth",
            "parameter/jacobinet.pth",
            "parameter/pinn_jacobinet.pth",
        ],
    },
    "STENOSIS_3D_CO": {
        "code_dir": "pinn code_STENOSIS_3D_CO",
        "data_dir": "groundtruth_STENOSIS_3D_CO",
        "methods": {
            "baseline": {
                "method_dir": "pinn_baseline",
                "train_script": "main.py",
                "eval_script": "post.py",
            },
            "jacobinet": {
                "method_dir": "pinn_jacobinet",
                "train_script": "main.py",
                "eval_script": "post.py",
            },
        },
        "required_files": [
            "groundtruth_STENOSIS_3D_CO/export_stenosis.csv",
            "groundtruth_STENOSIS_3D_CO/stenosis.xlsx",
            "groundtruth_STENOSIS_3D_CO/stenosis_n.xlsx",
            "groundtruth_STENOSIS_3D_CO/stenosis_post.xlsx",
            "groundtruth_STENOSIS_3D_CO/internal.csv",
            "groundtruth_STENOSIS_3D_CO/inlet.csv",
            "groundtruth_STENOSIS_3D_CO/outlet.csv",
            "groundtruth_STENOSIS_3D_CO/bd.csv",
            "parameter/pinn_baseline.pth",
            "parameter/jacobinet.pth",
            "parameter/pinn_jacobinet.pth",
        ],
    },
    "STENOSIS_3D_EC": {
        "code_dir": "pinn code_STENOSIS_3D_EC",
        "data_dir": "groundtruth_STENOSIS_3D_EC",
        "methods": {
            "baseline": {
                "method_dir": "pinn_baseline",
                "train_script": "main.py",
                "eval_script": "post.py",
            },
            "jacobinet": {
                "method_dir": "pinn_jacobinet",
                "train_script": "main.py",
                "eval_script": "post.py",
            },
        },
        "required_files": [
            "groundtruth_STENOSIS_3D_EC/export_stenosis.csv",
            "groundtruth_STENOSIS_3D_EC/stenosis.xlsx",
            "groundtruth_STENOSIS_3D_EC/stenosis_n.xlsx",
            "groundtruth_STENOSIS_3D_EC/stenosis_post.xlsx",
            "groundtruth_STENOSIS_3D_EC/internal.csv",
            "groundtruth_STENOSIS_3D_EC/inlet.csv",
            "groundtruth_STENOSIS_3D_EC/outlet.csv",
            "groundtruth_STENOSIS_3D_EC/bd.csv",
            "parameter/pinn_baseline.pth",
            "parameter/jacobinet.pth",
            "parameter/pinn_jacobinet.pth",
        ],
    },
    "T": {
        "code_dir": "pinn code_T",
        "data_dir": "groundtruth_T",
        "methods": {
            "baseline": {
                "method_dir": "pinn_baseline",
                "train_script": "main.py",
                "eval_script": "post.py",
            },
            "jacobinet": {
                "method_dir": "pinn_jacobinet",
                "train_script": "main.py",
                "eval_script": "post.py",
            },
        },
        "required_files": [
            "groundtruth_T/export.csv",
            "groundtruth_T/xyds.xlsx",
            "parameter/pinn_baseline.pth",
            "parameter/jacobinet.pth",
            "parameter/pinn_jacobinet.pth",
        ],
    },
    "U": {
        "code_dir": "pinn code_U",
        "data_dir": "groundtruth_U",
        "methods": {
            "baseline": {
                "method_dir": "pinn_baseline",
                "train_script": "main.py",
                "eval_script": "post.py",
            },
            "jacobinet": {
                "method_dir": "pinn_jacobinet",
                "train_script": "main.py",
                "eval_script": "post.py",
            },
        },
        "required_files": [
            "groundtruth_U/export.csv",
            "groundtruth_U/xyds.xlsx",
            "parameter/pinn_baseline.pth",
            "parameter/jacobinet.pth",
            "parameter/pinn_jacobinet.pth",
        ],
    },
}

PINN_CASE_SELECTIONS = {
    "U": ["U"],
    "S": ["S"],
    "T": ["T"],
    "STENOSIS_CO": ["STENOSIS_3D_CO"],
    "STENOSIS_EC": ["STENOSIS_3D_EC"],
    "STENOSIS_3D_CO": ["STENOSIS_3D_CO"],
    "STENOSIS_3D_EC": ["STENOSIS_3D_EC"],
}
PINN_CASE_ORDER = [
    "U",
    "S",
    "T",
    "STENOSIS_3D_CO",
    "STENOSIS_3D_EC",
]
PINN_CASE_SELECTION_ORDER = [
    "U",
    "S",
    "T",
    "STENOSIS_CO",
    "STENOSIS_EC",
]
PINN_METHOD_ORDER = ["baseline", "jacobinet"]


@dataclass(frozen=True)
class JacobiNetCaseSpec:
    name: str
    code_dir: str
    data_file: str
    dim: int
    input_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    internal_sheet: str
    boundary_sheets: tuple[str, ...]
    coordinate_scale: tuple[float, ...]
    epochs: int
    seed: int
    boundary_weight: float
    checkpoint_format: str
    early_stop_rmse: float | None = None


JACOBINET_MODEL_HIDDEN_WIDTH = 128
JACOBINET_MODEL_HIDDEN_LAYERS = 2

JACOBINET_DEFAULT_OUTPUT_ROOT = "jacobinet_runs"
JACOBINET_DEFAULT_OUTPUT_NAME = "jacobinet.pth"
JACOBINET_DEFAULT_RUN_ID_FORMAT = "%Y%m%d_%H%M%S"
JACOBINET_DEFAULT_CASE_SELECTION = "ALL"
JACOBINET_DEFAULT_DEVICE: str | None = None
JACOBINET_DEFAULT_EPOCHS_OVERRIDE: int | None = None
JACOBINET_DEFAULT_LEARNING_RATE = 1e-3
JACOBINET_DEFAULT_ETA_MIN = 1e-5
JACOBINET_DEFAULT_PRINT_EVERY = 500
JACOBINET_DEFAULT_DET_SAMPLE_SIZE = 20000
JACOBINET_DEFAULT_TRAIN_FRACTION = 0.70
JACOBINET_DEFAULT_VALID_FRACTION = 0.15
JACOBINET_DEFAULT_TEST_FRACTION = 0.15
JACOBINET_DEFAULT_SEED_OVERRIDE: int | None = None

JACOBINET_CASES: dict[str, JacobiNetCaseSpec] = {
    "U": JacobiNetCaseSpec(
        name="U",
        code_dir="pinn code_U",
        data_file="groundtruth_U/xyds.xlsx",
        dim=2,
        input_columns=("x", "y"),
        target_columns=("d", "s"),
        internal_sheet="encrypted_points",
        boundary_sheets=("inlet_line_points", "outlet_line_points", "bd1", "bd2"),
        coordinate_scale=(0.01 / 2, 0.01 / 2),
        epochs=50000,
        seed=10,
        boundary_weight=10.0,
        checkpoint_format="dict",
        early_stop_rmse=1e-3,
    ),
    "S": JacobiNetCaseSpec(
        name="S",
        code_dir="pinn code_S",
        data_file="groundtruth_S/xyds.xlsx",
        dim=2,
        input_columns=("x", "y"),
        target_columns=("d", "s"),
        internal_sheet="encrypted_points",
        boundary_sheets=("inlet_line_points", "outlet_line_points", "bd1", "bd2"),
        coordinate_scale=(0.01 / 4, 0.01 / 4),
        epochs=50000,
        seed=99,
        boundary_weight=10.0,
        checkpoint_format="dict",
        early_stop_rmse=1e-3,
    ),
    "T": JacobiNetCaseSpec(
        name="T",
        code_dir="pinn code_T",
        data_file="groundtruth_T/xyds.xlsx",
        dim=2,
        input_columns=("x", "y"),
        target_columns=("d", "s"),
        internal_sheet="encrypted_points",
        boundary_sheets=("inlet_line_points", "outlet_line_points", "bd1", "bd2"),
        coordinate_scale=(0.01 / 2, 0.01 / 2),
        epochs=50000,
        seed=99,
        boundary_weight=10.0,
        checkpoint_format="dict",
        early_stop_rmse=1e-3,
    ),
    "STENOSIS_3D_CO": JacobiNetCaseSpec(
        name="STENOSIS_3D_CO",
        code_dir="pinn code_STENOSIS_3D_CO",
        data_file="groundtruth_STENOSIS_3D_CO/stenosis_n.xlsx",
        dim=3,
        input_columns=("x", "y", "z"),
        target_columns=("x_n", "y_n", "z_n"),
        internal_sheet="internal",
        boundary_sheets=("inlet", "outlet", "bd"),
        coordinate_scale=(0.00469 / 2, 0.00469 / 2, 0.00469 / 2),
        epochs=100000,
        seed=99,
        boundary_weight=10.0,
        checkpoint_format="state_dict",
    ),
    "STENOSIS_3D_EC": JacobiNetCaseSpec(
        name="STENOSIS_3D_EC",
        code_dir="pinn code_STENOSIS_3D_EC",
        data_file="groundtruth_STENOSIS_3D_EC/stenosis_n.xlsx",
        dim=3,
        input_columns=("x", "y", "z"),
        target_columns=("x_n", "y_n", "z_n"),
        internal_sheet="internal",
        boundary_sheets=("inlet", "outlet", "bd"),
        coordinate_scale=(0.00469 / 2, 0.00469 / 2, 0.00469 / 2),
        epochs=100000,
        seed=99,
        boundary_weight=10.0,
        checkpoint_format="state_dict",
    ),
}

JACOBINET_CASE_ALIASES = {
    "ALL": tuple(JACOBINET_CASES),
    "UST": ("U", "S", "T"),
    "STENOSIS_CO": ("STENOSIS_3D_CO",),
    "STENOSIS_EC": ("STENOSIS_3D_EC",),
}
