from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from configs import (
    JACOBINET_CASES,
    JACOBINET_CASE_ALIASES,
    JACOBINET_DEFAULT_CASE_SELECTION,
    JACOBINET_DEFAULT_DEVICE,
    JACOBINET_DEFAULT_DET_SAMPLE_SIZE,
    JACOBINET_DEFAULT_EPOCHS_OVERRIDE,
    JACOBINET_DEFAULT_ETA_MIN,
    JACOBINET_DEFAULT_LEARNING_RATE,
    JACOBINET_DEFAULT_OUTPUT_NAME,
    JACOBINET_DEFAULT_OUTPUT_ROOT,
    JACOBINET_DEFAULT_PRINT_EVERY,
    JACOBINET_DEFAULT_RUN_ID_FORMAT,
    JACOBINET_DEFAULT_SEED_OVERRIDE,
    JACOBINET_DEFAULT_TEST_FRACTION,
    JACOBINET_DEFAULT_TRAIN_FRACTION,
    JACOBINET_DEFAULT_VALID_FRACTION,
    JACOBINET_MODEL_HIDDEN_LAYERS,
    JACOBINET_MODEL_HIDDEN_WIDTH,
    JacobiNetCaseSpec,
)

ROOT = Path(__file__).resolve().parent

# VSCode F5 defaults:
#   F5_CASES accepts "ALL", "UST", "U", "S", "T", "STENOSIS_CO", "STENOSIS_EC" or multiple values such as ("U", "S", "T").
F5_CASES = ("U", "S", "T")
# F5_CASES = ("STENOSIS_CO", "STENOSIS_EC")
F5_DEVICE = JACOBINET_DEFAULT_DEVICE
F5_RUN_ID = None
F5_OUTPUT_ROOT = JACOBINET_DEFAULT_OUTPUT_ROOT
F5_OUTPUT_NAME = JACOBINET_DEFAULT_OUTPUT_NAME
F5_EPOCHS = JACOBINET_DEFAULT_EPOCHS_OVERRIDE
F5_LEARNING_RATE = JACOBINET_DEFAULT_LEARNING_RATE
F5_ETA_MIN = JACOBINET_DEFAULT_ETA_MIN
F5_PRINT_EVERY = JACOBINET_DEFAULT_PRINT_EVERY
F5_DET_SAMPLE_SIZE = JACOBINET_DEFAULT_DET_SAMPLE_SIZE
F5_TRAIN_FRACTION = JACOBINET_DEFAULT_TRAIN_FRACTION
F5_VALID_FRACTION = JACOBINET_DEFAULT_VALID_FRACTION
F5_TEST_FRACTION = JACOBINET_DEFAULT_TEST_FRACTION
F5_SEED = JACOBINET_DEFAULT_SEED_OVERRIDE

# Backward-compatible names for small comparison/debug snippets.
CASES = JACOBINET_CASES
CASE_ALIASES = JACOBINET_CASE_ALIASES
CaseSpec = JacobiNetCaseSpec


class JacobiNet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = dim
        for _ in range(JACOBINET_MODEL_HIDDEN_LAYERS):
            layers.extend([nn.Linear(in_features, JACOBINET_MODEL_HIDDEN_WIDTH), nn.Tanh()])
            in_features = JACOBINET_MODEL_HIDDEN_WIDTH
        layers.append(nn.Linear(in_features, dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)


@dataclass
class SplitData:
    x: torch.Tensor
    y: torch.Tensor


@dataclass
class GeometryData:
    train_internal: SplitData
    valid_internal: SplitData
    test_internal: SplitData
    train_boundary: SplitData
    valid_boundary: SplitData
    test_boundary: SplitData
    input_mean: torch.Tensor
    target_mean: torch.Tensor
    input_min: torch.Tensor
    input_max: torch.Tensor
    target_min: torch.Tensor
    target_max: torch.Tensor
    internal_count: int
    boundary_count: int


def selection_to_arg(value: str | Sequence[str]) -> str:
    if isinstance(value, str):
        return value
    return ",".join(value)


def split_selection(value: str | Sequence[str], *, uppercase: bool) -> list[str]:
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = []
        for item in value:
            raw_items.extend(str(item).split(","))
    items = [item.strip() for item in raw_items if item.strip()]
    if uppercase:
        return [item.upper() for item in items]
    return items


def parse_cases(value: str | Sequence[str]) -> list[str]:
    selected: list[str] = []
    for item in split_selection(value, uppercase=True):
        if item in CASE_ALIASES:
            selected.extend(CASE_ALIASES[item])
        elif item in CASES:
            selected.append(item)
        else:
            valid = ", ".join(["ALL", "UST", *CASES])
            raise SystemExit(f"Unknown case {item!r}. Valid cases: {valid}")

    result = []
    seen = set()
    for item in selected:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def set_seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_points(sheets: dict[str, pd.DataFrame], sheet_names: Iterable[str], columns: tuple[str, ...]) -> torch.Tensor:
    frames = []
    for sheet_name in sheet_names:
        if sheet_name not in sheets:
            raise KeyError(f"Missing sheet {sheet_name!r}")
        frame = sheets[sheet_name].loc[:, list(columns)].apply(pd.to_numeric, errors="coerce").dropna()
        frames.append(frame)
    values = pd.concat(frames, ignore_index=True).to_numpy(dtype=np.float32)
    return torch.from_numpy(values)


def normalize_inputs(raw_inputs: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (raw_inputs - mean) / scale


def normalize_targets(raw_targets: torch.Tensor, mean: torch.Tensor, subtract_mean: bool) -> torch.Tensor:
    if subtract_mean:
        return raw_targets - mean
    return raw_targets


def split_tensor_pair(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    generator: torch.Generator,
    fractions: tuple[float, float, float],
) -> tuple[SplitData, SplitData, SplitData]:
    if len(inputs) != len(targets):
        raise ValueError("Input and target tensors must have the same length.")
    if len(inputs) < 3:
        raise ValueError("At least three points are required for train/valid/test splitting.")

    train_fraction, valid_fraction, test_fraction = fractions
    if abs((train_fraction + valid_fraction + test_fraction) - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1.0.")

    count = len(inputs)
    order = torch.randperm(count, generator=generator)
    train_count = max(1, int(count * train_fraction))
    valid_count = max(1, int(count * valid_fraction))
    if train_count + valid_count >= count:
        valid_count = max(1, count - train_count - 1)
    test_count = count - train_count - valid_count
    if test_count < 1:
        test_count = 1
        train_count = count - valid_count - test_count

    train_idx = order[:train_count]
    valid_idx = order[train_count : train_count + valid_count]
    test_idx = order[train_count + valid_count :]
    return (
        SplitData(inputs[train_idx], targets[train_idx]),
        SplitData(inputs[valid_idx], targets[valid_idx]),
        SplitData(inputs[test_idx], targets[test_idx]),
    )


def load_geometry_data(
    spec: CaseSpec,
    device: torch.device,
    split: tuple[float, float, float],
) -> GeometryData:
    data_path = ROOT / spec.code_dir / spec.data_file
    sheets = pd.read_excel(data_path, sheet_name=None)
    boundary_sheets = tuple(spec.boundary_sheets)

    raw_internal_x = read_points(sheets, (spec.internal_sheet,), spec.input_columns)
    raw_boundary_x = read_points(sheets, boundary_sheets, spec.input_columns)
    raw_internal_y = read_points(sheets, (spec.internal_sheet,), spec.target_columns)
    raw_boundary_y = read_points(sheets, boundary_sheets, spec.target_columns)

    raw_all_x = torch.cat([raw_internal_x, raw_boundary_x], dim=0)
    raw_all_y = torch.cat([raw_internal_y, raw_boundary_y], dim=0)
    input_min = raw_all_x.min(dim=0, keepdim=True).values
    input_max = raw_all_x.max(dim=0, keepdim=True).values
    input_mean = (input_min + input_max) / 2
    target_min = raw_all_y.min(dim=0, keepdim=True).values
    target_max = raw_all_y.max(dim=0, keepdim=True).values
    target_mean = (target_min + target_max) / 2

    scale = torch.tensor(spec.coordinate_scale, dtype=torch.float32).view(1, spec.dim)
    subtract_target_mean = spec.dim == 2
    internal_x = normalize_inputs(raw_internal_x, input_mean, scale)
    boundary_x = normalize_inputs(raw_boundary_x, input_mean, scale)
    internal_y = normalize_targets(raw_internal_y, target_mean, subtract_target_mean)
    boundary_y = normalize_targets(raw_boundary_y, target_mean, subtract_target_mean)

    generator = torch.Generator().manual_seed(spec.seed)
    train_i, valid_i, test_i = split_tensor_pair(internal_x, internal_y, generator, split)
    train_b, valid_b, test_b = split_tensor_pair(boundary_x, boundary_y, generator, split)

    def move(data: SplitData) -> SplitData:
        return SplitData(data.x.to(device), data.y.to(device))

    return GeometryData(
        train_internal=move(train_i),
        valid_internal=move(valid_i),
        test_internal=move(test_i),
        train_boundary=move(train_b),
        valid_boundary=move(valid_b),
        test_boundary=move(test_b),
        input_mean=input_mean,
        target_mean=target_mean,
        input_min=input_min,
        input_max=input_max,
        target_min=target_min,
        target_max=target_max,
        internal_count=len(raw_internal_x),
        boundary_count=len(raw_boundary_x),
    )


def jacobian_outputs(inputs: torch.Tensor, model: nn.Module, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    x = inputs.detach().clone().requires_grad_(True)
    outputs = model(x)
    rows = []
    for col in range(dim):
        grad = torch.autograd.grad(
            outputs[:, col : col + 1],
            x,
            torch.ones_like(outputs[:, col : col + 1]),
            create_graph=False,
            retain_graph=True,
        )[0]
        rows.append(grad)
    return outputs, torch.cat(rows, dim=1)


def det_jacobian(jacobian_flat: torch.Tensor, dim: int) -> torch.Tensor:
    if dim == 2:
        d_x, d_y, s_x, s_y = jacobian_flat[:, 0], jacobian_flat[:, 1], jacobian_flat[:, 2], jacobian_flat[:, 3]
        return d_x * s_y - d_y * s_x

    r_x, r_y, r_z = jacobian_flat[:, 0], jacobian_flat[:, 1], jacobian_flat[:, 2]
    d_x, d_y, d_z = jacobian_flat[:, 3], jacobian_flat[:, 4], jacobian_flat[:, 5]
    s_x, s_y, s_z = jacobian_flat[:, 6], jacobian_flat[:, 7], jacobian_flat[:, 8]
    return r_x * (d_y * s_z - d_z * s_y) - r_y * (d_x * s_z - d_z * s_x) + r_z * (d_x * s_y - d_y * s_x)


def rmse(model: nn.Module, data: SplitData, mse: nn.Module) -> float:
    if len(data.x) == 0:
        return float("nan")
    with torch.no_grad():
        return float(mse(model(data.x), data.y).sqrt().item())


def evaluate(model: nn.Module, data: GeometryData, mse: nn.Module, boundary_weight: float, prefix: str) -> dict[str, float]:
    internal = getattr(data, f"{prefix}_internal")
    boundary = getattr(data, f"{prefix}_boundary")
    internal_rmse = rmse(model, internal, mse)
    boundary_rmse = rmse(model, boundary, mse)
    total = internal_rmse**2 + boundary_weight * boundary_rmse**2
    return {
        f"{prefix}_internal_rmse": internal_rmse,
        f"{prefix}_boundary_rmse": boundary_rmse,
        f"{prefix}_weighted_mse": total,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    spec: CaseSpec,
    data: GeometryData,
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec.checkpoint_format == "dict":
        torch.save(
            {
                "state_dict": model.state_dict(),
                "mean_ds": data.target_mean.cpu(),
                "mean_xy": data.input_mean.cpu(),
            },
            path,
        )
    else:
        torch.save(model.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def sample_for_jacobian(points: torch.Tensor, max_points: int) -> torch.Tensor:
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = torch.randperm(len(points), device=points.device)[:max_points]
    return points[indices]


def train_case(
    spec: CaseSpec,
    args: argparse.Namespace,
    run_root: Path,
    device: torch.device,
) -> dict[str, object]:
    set_seed(args.seed if args.seed is not None else spec.seed, device)
    split = (args.train_fraction, args.valid_fraction, args.test_fraction)
    data = load_geometry_data(spec, device, split)

    epochs = args.epochs if args.epochs is not None else spec.epochs
    output_path = run_root / spec.name / args.output_name
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {output_path}")

    model = JacobiNet(spec.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.eta_min)
    mse = nn.MSELoss()

    best_valid = float("inf")
    best_epoch = 0
    start_time = time.time()

    print(f"[{spec.name}] data={ROOT / spec.code_dir / spec.data_file}")
    print(f"[{spec.name}] split=train/valid/test {split[0]:.2f}/{split[1]:.2f}/{split[2]:.2f}")
    print(f"[{spec.name}] points internal={data.internal_count}, boundary={data.boundary_count}")
    print(f"[{spec.name}] epochs={epochs}, boundary_weight={spec.boundary_weight}, output={output_path}")

    for epoch in range(1, epochs + 1):
        model.train()
        pred_internal = model(data.train_internal.x)
        pred_boundary = model(data.train_boundary.x)
        internal_loss = mse(pred_internal, data.train_internal.y)
        boundary_loss = mse(pred_boundary, data.train_boundary.y)
        loss = internal_loss + spec.boundary_weight * boundary_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        valid_metrics = evaluate(model, data, mse, spec.boundary_weight, "valid")
        valid_total = valid_metrics["valid_weighted_mse"]
        if valid_total < best_valid:
            best_valid = valid_total
            best_epoch = epoch
            metadata = {
                "case": spec.name,
                "epoch": epoch,
                "epochs_requested": epochs,
                "checkpoint_format": spec.checkpoint_format,
                "split": {
                    "train": args.train_fraction,
                    "valid": args.valid_fraction,
                    "test": args.test_fraction,
                },
                "valid_weighted_mse": valid_total,
                "valid_internal_rmse": valid_metrics["valid_internal_rmse"],
                "valid_boundary_rmse": valid_metrics["valid_boundary_rmse"],
            }
            save_checkpoint(output_path, model, spec, data, metadata)

        should_print = epoch == 1 or epoch % args.print_every == 0 or epoch == epochs
        if should_print:
            train_internal_rmse = float(internal_loss.sqrt().item())
            train_boundary_rmse = float(boundary_loss.sqrt().item())
            det_points = sample_for_jacobian(data.train_internal.x, args.det_sample_size)
            _, jacobian = jacobian_outputs(det_points, model, spec.dim)
            det_values = det_jacobian(jacobian, spec.dim)
            det_penalty = torch.relu(-det_values).mean().item()
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"[{spec.name}] Ep {epoch:6d} "
                f"| lr={lr_now:.1e} "
                f"| train_internal_RMSE={train_internal_rmse:.3e} "
                f"| train_boundary_RMSE={train_boundary_rmse:.3e} "
                f"| valid_internal_RMSE={valid_metrics['valid_internal_rmse']:.3e} "
                f"| valid_boundary_RMSE={valid_metrics['valid_boundary_rmse']:.3e} "
                f"| det_penalty={det_penalty:.2e} "
                f"| min(detJ)={det_values.min().item():.2e} "
                f"| mean(detJ)={det_values.mean().item():.2e}"
            )

        if spec.early_stop_rmse is not None and not args.no_early_stop:
            if float(internal_loss.sqrt().item()) < spec.early_stop_rmse:
                print(f"[{spec.name}] Early stop at epoch {epoch}: train internal RMSE < {spec.early_stop_rmse:.1e}")
                break

    checkpoint = torch.load(output_path, map_location=device, weights_only=True)
    if spec.checkpoint_format == "dict":
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    train_metrics = evaluate(model, data, mse, spec.boundary_weight, "train")
    valid_metrics = evaluate(model, data, mse, spec.boundary_weight, "valid")
    test_metrics = evaluate(model, data, mse, spec.boundary_weight, "test")
    elapsed = time.time() - start_time
    result: dict[str, object] = {
        "case": spec.name,
        "checkpoint": output_path.relative_to(ROOT).as_posix(),
        "best_epoch": best_epoch,
        "epochs_requested": epochs,
        "elapsed_sec": round(elapsed, 3),
        "internal_points": data.internal_count,
        "boundary_points": data.boundary_count,
        **train_metrics,
        **valid_metrics,
        **test_metrics,
    }
    output_path.with_suffix(".json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[{spec.name}] done | best_epoch={best_epoch} "
        f"| test_internal_RMSE={result['test_internal_rmse']:.3e} "
        f"| test_boundary_RMSE={result['test_boundary_rmse']:.3e}"
    )
    return result


def write_report(run_root: Path, results: list[dict[str, object]]) -> None:
    if not results:
        return
    report_csv = run_root / "jacobinet_training_report.csv"
    fieldnames = list(results[0].keys())
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    (run_root / "jacobinet_training_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Report written to {report_csv.relative_to(ROOT)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train standalone JacobiNet coordinate-transform checkpoints.")
    default_device = F5_DEVICE or ("cuda:0" if torch.cuda.is_available() else "cpu")
    default_run_id = F5_RUN_ID or datetime.now().strftime(JACOBINET_DEFAULT_RUN_ID_FORMAT)
    parser.add_argument("--cases", default=selection_to_arg(F5_CASES), help="ALL, UST, U, S, T, STENOSIS_3D_CO, STENOSIS_3D_EC")
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--run-id", default=default_run_id)
    parser.add_argument("--output-root", default=F5_OUTPUT_ROOT)
    parser.add_argument("--output-name", default=F5_OUTPUT_NAME)
    parser.add_argument("--epochs", type=int, default=F5_EPOCHS, help="Override all case-specific epoch counts.")
    parser.add_argument("--learning-rate", type=float, default=F5_LEARNING_RATE)
    parser.add_argument("--eta-min", type=float, default=F5_ETA_MIN)
    parser.add_argument("--print-every", type=int, default=F5_PRINT_EVERY)
    parser.add_argument("--det-sample-size", type=int, default=F5_DET_SAMPLE_SIZE)
    parser.add_argument("--train-fraction", type=float, default=F5_TRAIN_FRACTION)
    parser.add_argument("--valid-fraction", type=float, default=F5_VALID_FRACTION)
    parser.add_argument("--test-fraction", type=float, default=F5_TEST_FRACTION)
    parser.add_argument("--seed", type=int, default=F5_SEED, help="Override each case's default seed.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting checkpoints in the selected run folder.")
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    selected_cases = parse_cases(args.cases)
    split_total = args.train_fraction + args.valid_fraction + args.test_fraction
    if abs(split_total - 1.0) > 1e-8:
        raise SystemExit("train/valid/test fractions must sum to 1.0")

    run_root = ROOT / args.output_root / args.run_id
    device = torch.device(args.device)

    print(f"Selected cases: {', '.join(selected_cases)}")
    print(f"Device: {device}")
    print(f"Run root: {run_root}")
    for case_name in selected_cases:
        spec = CASES[case_name]
        epochs = args.epochs if args.epochs is not None else spec.epochs
        output_path = run_root / spec.name / args.output_name
        print(
            f"[{spec.name}] planned epochs={epochs}, data={spec.code_dir}/{spec.data_file}, "
            f"checkpoint={output_path.relative_to(ROOT).as_posix()}"
        )

    if args.dry_run:
        return 0

    run_root.mkdir(parents=True, exist_ok=True)
    results = [train_case(CASES[case_name], args, run_root, device) for case_name in selected_cases]
    write_report(run_root, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
