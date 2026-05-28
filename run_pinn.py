from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import threading
from collections.abc import Sequence

from configs import (
    PINN_CASES,
    PINN_CASE_ORDER,
    PINN_CASE_SELECTIONS,
    PINN_METHOD_ORDER,
    REYNOLDS_VALUES,
)

ROOT = Path(__file__).resolve().parent
DEBUGGER_ENV_PREFIXES = ("DEBUGPY_", "PYDEVD_")

# VSCode F5 defaults:
#   F5_CASES accepts "ALL", "U", "S", "T", "STENOSIS_CO",
#   "STENOSIS_EC", or multiple values such as ("U", "S", "T").
#   F5_METHODS accepts "all", "baseline", "jacobinet", or multiple values.
#   F5_MODE accepts "train" or "eval".
#   F5_RE is kept for Reynolds-number cases; active release cases ignore it.
F5_CASES = ("U", "S", "T")

# F5_CASES = ("ALL",)
# F5_CASES = ("STENOSIS_CO", "STENOSIS_EC")
F5_METHODS = ("all",)
# F5_METHODS = ("baseline",)
# F5_METHODS = ("jacobinet",)
# F5_MODE = "eval"
F5_MODE = "train"
F5_RE = "ALL"
# F5_RE = ("10", "100", "300", "500", "1000")
# F5_PYTHON = r"path\to\python.exe"
F5_PYTHON = None


def disable_debugger_subprocess_injection() -> None:
    if sys.gettrace() is None:
        return

    try:
        import debugpy  # type: ignore[import-not-found]
    except Exception:
        return

    try:
        debugpy.configure(subProcess=False)
    except Exception:
        return


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
    return [item.lower() for item in items]


def dedupe(items: list[str]) -> list[str]:
    result = []
    seen = set()
    for item in items:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def parse_method_selection(value: str | Sequence[str]) -> list[str]:
    selected = split_selection(value, uppercase=False)
    if selected == ["all"]:
        return PINN_METHOD_ORDER
    unknown = [item for item in selected if item not in PINN_METHOD_ORDER]
    if unknown:
        raise SystemExit(f"Unknown method: {', '.join(unknown)}")
    return selected


def parse_case_selection(value: str | Sequence[str]) -> list[str]:
    selected = split_selection(value, uppercase=True)
    if selected == ["ALL"]:
        return PINN_CASE_ORDER
    unknown = [item for item in selected if item not in PINN_CASE_SELECTIONS]
    if unknown:
        valid = ", ".join(["ALL", *PINN_CASE_SELECTIONS])
        raise SystemExit(f"Unknown case: {', '.join(unknown)}. Valid cases: {valid}")

    cases = []
    for item in selected:
        cases.extend(PINN_CASE_SELECTIONS[item])
    return dedupe(cases)


def normalize_re_label(value: str) -> str:
    return str(int(float(value)))


def parse_re_selection(value: str | Sequence[str]) -> list[str]:
    selected = split_selection(value, uppercase=True)
    if selected == ["ALL"]:
        return REYNOLDS_VALUES

    normalized = [normalize_re_label(item) for item in selected]
    unknown = [item for item in normalized if item not in REYNOLDS_VALUES]
    if unknown:
        valid = ", ".join(["ALL", *REYNOLDS_VALUES])
        raise SystemExit(f"Unknown Re: {', '.join(unknown)}. Valid Re values: {valid}")
    return dedupe(normalized)


def build_runs(mode: str, cases: list[str], methods: list[str], re_labels: list[str]) -> list[dict]:
    runs = []
    script_key = "train_script" if mode == "train" else "eval_script"
    for case_name in cases:
        case = PINN_CASES[case_name]
        code_dir = ROOT / case["code_dir"]
        case_re_values = case.get("re_values")
        case_re_labels = re_labels if case_re_values else [None]
        if case_re_values:
            unsupported = [re_label for re_label in case_re_labels if re_label not in case_re_values]
            if unsupported:
                valid = ", ".join(case_re_values)
                raise SystemExit(f"{case_name} does not support Re {', '.join(unsupported)}. Valid Re values: {valid}")
        for re_label in case_re_labels:
            for method_name in methods:
                method = case["methods"][method_name]
                method_dir = code_dir / method["method_dir"]
                script = method_dir / method[script_key]
                runs.append(
                    {
                        "case": case_name,
                        "method": method_name,
                        "re": re_label,
                        "cwd": method_dir,
                        "script": script,
                    }
                )
    return runs


def validate_runs(runs: list[dict]) -> None:
    missing = []
    for run in runs:
        if not run["cwd"].is_dir():
            missing.append(f"missing method directory: {run['cwd']}")
        if not run["script"].is_file():
            missing.append(f"missing script: {run['script']}")
    if missing:
        raise SystemExit("\n".join(missing))


def tee_stream(pipe, console_stream, log_file) -> None:
    try:
        for line in iter(pipe.readline, ""):
            console_stream.write(line)
            console_stream.flush()
            log_file.write(line)
            log_file.flush()
    finally:
        pipe.close()


def run_with_live_output(cmd: list[str], cwd: Path, env: dict[str, str], stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            errors="replace",
        )

        threads = [
            threading.Thread(
                target=tee_stream,
                args=(process.stdout, sys.stdout, stdout_file),
                daemon=True,
            ),
            threading.Thread(
                target=tee_stream,
                args=(process.stderr, sys.stderr, stderr_file),
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()

        returncode = process.wait()
        for thread in threads:
            thread.join()
        return returncode


def main() -> int:
    disable_debugger_subprocess_injection()

    parser = argparse.ArgumentParser(description="Run JacobiNet cases from one entrypoint.")
    parser.add_argument("--mode", choices=["train", "eval"], default=F5_MODE)
    parser.add_argument(
        "--cases",
        default=selection_to_arg(F5_CASES),
        help=(
            "ALL, U, S, T, STENOSIS_CO, STENOSIS_EC, or comma-separated selections"
        ),
    )
    parser.add_argument(
        "--methods",
        default=selection_to_arg(F5_METHODS),
        help="all, baseline, jacobinet, or comma-separated",
    )
    parser.add_argument(
        "--re",
        default=selection_to_arg(F5_RE),
        help="ALL, 10, 100, 300, 500, 1000, or comma-separated values for Reynolds-number cases",
    )
    parser.add_argument("--python", default=F5_PYTHON or sys.executable, help="Python executable to use.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without running them.")
    args = parser.parse_args()

    cases = parse_case_selection(args.cases)
    methods = parse_method_selection(args.methods)
    re_labels = parse_re_selection(args.re)
    runs = build_runs(args.mode, cases, methods, re_labels)
    validate_runs(runs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = ROOT / "runs" / timestamp
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(DEBUGGER_ENV_PREFIXES):
            env.pop(key, None)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    warning_filters = [
        env.get("PYTHONWARNINGS"),
        "ignore:FigureCanvasAgg is non-interactive:UserWarning",
    ]
    env["PYTHONWARNINGS"] = ",".join(filter(None, warning_filters))

    summaries = []
    for run in runs:
        cmd = [args.python, str(run["script"].name)]
        re_label = f"_Re{run['re']}" if run["re"] else ""
        label = f"{run['case']}{re_label}_{run['method']}_{args.mode}"
        print(f"[{label}] cwd={run['cwd']}")
        print(f"[{label}] command={' '.join(cmd)}")
        if run["re"]:
            print(f"[{label}] JACOBINET_RE={run['re']}")

        if args.dry_run:
            continue

        run_root.mkdir(parents=True, exist_ok=True)
        stdout_path = run_root / f"{label}.out.log"
        stderr_path = run_root / f"{label}.err.log"
        run_env = env.copy()
        if run["re"]:
            run_env["JACOBINET_RE"] = run["re"]
        else:
            run_env.pop("JACOBINET_RE", None)
        returncode = run_with_live_output(cmd, run["cwd"], run_env, stdout_path, stderr_path)

        summary = {
            "case": run["case"],
            "method": run["method"],
            "re": run["re"],
            "mode": args.mode,
            "returncode": returncode,
            "stdout": stdout_path.relative_to(ROOT).as_posix(),
            "stderr": stderr_path.relative_to(ROOT).as_posix(),
        }
        summaries.append(summary)
        print(f"[{label}] returncode={returncode}")

        if returncode != 0:
            break

    if args.dry_run:
        return 0

    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    failed = [item for item in summaries if item["returncode"] != 0]
    if failed:
        print(f"One or more runs failed. See {summary_path.relative_to(ROOT)}")
        return 1
    print(f"All runs completed. See {summary_path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
