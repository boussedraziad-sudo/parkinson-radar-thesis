"""Patch notebooks (h5py→scipy.io.loadmat) and execute them in-place.

Run from the repo root with the venv's python:
    .venv/bin/python scripts/_patch_and_run_notebooks.py 01 02 03 ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[1]
NB_DIR = REPO / "notebooks"

# Replacements applied to every code cell's source. Each entry is (old, new).
# Keep them surgical — single-line substitutions that don't fight indentation.
PATCHES: list[tuple[str, str]] = [
    ("import h5py\n", "import scipy.io as sio\nfrom tqdm.auto import tqdm\n"),
    ("import h5py", "import scipy.io as sio\nfrom tqdm.auto import tqdm"),
    (
        'with h5py.File(TRIAL, "r") as f:\n    for k in sorted(f.keys()):\n        try:\n            print(f"  {k:22s}  shape={tuple(f[k].shape)}  dtype={f[k].dtype}")\n        except AttributeError:\n            print(f"  {k:22s}  (group)")',
        'info = sio.whosmat(str(TRIAL))\nfor name, shape, dtype in sorted(info):\n    print(f"  {name:22s}  shape={tuple(shape)}  dtype={dtype}")',
    ),
    # Notebook 02 cell 11 — read t_axis to compute duration
    (
        '    try:\n        with h5py.File(r["path"], "r") as f:\n            t = np.array(f["t_axis_target"]).squeeze()\n            durations.append(float(t.max() - t.min()))\n    except Exception:\n        durations.append(np.nan)',
        '    try:\n        _raw = sio.loadmat(r["path"], variable_names=["t_axis_target"], squeeze_me=True)\n        t = np.asarray(_raw["t_axis_target"]).ravel().astype(float)\n        durations.append(float(t.max() - t.min()))\n    except Exception:\n        durations.append(np.nan)',
    ),
    # Notebook 02 cell 14 — count Doppler bins (use whosmat, no data read)
    (
        '    try:\n        with h5py.File(path, "r") as f:\n            dop_n.append(int(np.array(f["doppler_axis"]).size))\n    except Exception:\n        dop_n.append(-1)',
        '    try:\n        _info = {n: s for n, s, _ in sio.whosmat(path)}\n        dop_n.append(int(np.prod(_info["doppler_axis"])) if "doppler_axis" in _info else -1)\n    except Exception:\n        dop_n.append(-1)',
    ),
]


def patch_cell(src: str) -> str:
    for old, new in PATCHES:
        if old in src:
            src = src.replace(old, new)
    # Idempotent: make sure tqdm is importable from any cell that uses it.
    if "tqdm(" in src and "from tqdm" not in src and "import tqdm" not in src:
        src = "from tqdm.auto import tqdm\n" + src
    return src


def patch_notebook(path: Path) -> nbformat.NotebookNode:
    nb = nbformat.read(path, as_version=4)
    # Set kernel
    nb.metadata.setdefault("kernelspec", {})
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3.10 (thesis)",
        "language": "python",
        "name": "thesis-parkinson-radar",
    }
    nb.metadata["language_info"] = {"name": "python"}
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.source = patch_cell(cell.source)
    return nb


def run_notebook(path: Path, timeout: int = 1800) -> None:
    print(f"\n=== Executing {path.name} ===")
    nb = patch_notebook(path)
    client = NotebookClient(nb, timeout=timeout, kernel_name="thesis-parkinson-radar",
                            resources={"metadata": {"path": str(path.parent)}})
    try:
        client.execute()
        status = "OK"
    except Exception as exc:
        status = f"FAIL: {type(exc).__name__}: {exc}"
        # Continue: still write whatever ran so we can inspect partial outputs.
    nbformat.write(nb, path)
    print(f"--- {path.name}: {status}")


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: _patch_and_run_notebooks.py <prefix> [<prefix> ...]", file=sys.stderr)
        return 2
    notebooks = []
    for prefix in argv:
        matches = sorted(NB_DIR.glob(f"{prefix}*.ipynb"))
        if not matches:
            print(f"warning: no notebook matched {prefix!r}", file=sys.stderr)
        notebooks.extend(matches)
    for nb_path in notebooks:
        run_notebook(nb_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
