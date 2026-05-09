from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def softmax_with_temperature(input, t=1, axis=-1):
    ex = torch.exp(input / t)
    return ex / torch.sum(ex, axis=axis, keepdim=True)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def get_noises(timesteps=500):
    betas = linear_beta_schedule(timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    return sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod


def _pick_key(mat, candidates):
    lower_to_key = {k.lower(): k for k in mat.keys() if not k.startswith("__")}
    for name in candidates:
        if name.lower() in lower_to_key:
            return lower_to_key[name.lower()]
    return None


def _resolve_mat_path(dataset, data_dir):
    data_dir = Path(data_dir).expanduser()
    name = Path(dataset).stem
    aliases = {
        "books": "book",
        "book": "book",
        "disney": "Disney",
        "enron": "Enron",
        "reddit": "Reddit",
        "yelpchi": "YelpChi",
        "yelpchi-all": "YelpChi-all",
        "amazon-all": "Amazon-all",
    }

    candidates = []
    if str(dataset).endswith(".mat"):
        candidates.append(data_dir / dataset)
    candidates.append(data_dir / f"{dataset}.mat")
    candidates.append(data_dir / f"{name}.mat")
    if name.lower() in aliases:
        candidates.append(data_dir / f"{aliases[name.lower()]}.mat")

    for path in candidates:
        if path.exists():
            return path

    # Case-insensitive fallback for Windows/Linux name differences.
    for path in data_dir.glob("*.mat"):
        if path.stem.lower() == name.lower() or path.name.lower() == f"{dataset}.mat".lower():
            return path

    raise FileNotFoundError(f"Cannot find dataset '{dataset}' under {data_dir}")


def load_mat_data(dataset, data_dir="~/datasets/GAD/mat"):
    """Load Awesome-Deep-Graph-Anomaly-Detection .mat files as PyG Data.

    Supports common keys used by GAD .mat datasets:
    - adjacency: Network / A / adj / adjacency
    - features: Attributes / attrb / X / x / features
    - labels: Label / gnd / y / labels
    """
    path = _resolve_mat_path(dataset, data_dir)
    mat = sio.loadmat(path, squeeze_me=True)

    adj_key = _pick_key(mat, ["Network", "A", "adj", "adjacency"])
    feat_key = _pick_key(mat, ["Attributes", "attrb", "features", "X", "x"])
    label_key = _pick_key(mat, ["Label", "gnd", "label", "labels", "y"])

    if adj_key is None or feat_key is None or label_key is None:
        available = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(
            f"Unsupported .mat keys in {path}. Available keys: {available}. "
            "Please map adjacency/features/labels in load_mat_data()."
        )

    adj = mat[adj_key]
    x = mat[feat_key]
    y = mat[label_key]

    if sp.issparse(x):
        x = x.toarray()
    x = torch.tensor(np.asarray(x), dtype=torch.float32)

    if sp.issparse(adj):
        adj = adj.tocoo()
        row = torch.from_numpy(adj.row).long()
        col = torch.from_numpy(adj.col).long()
    else:
        row_np, col_np = np.nonzero(np.asarray(adj))
        row = torch.from_numpy(row_np).long()
        col = torch.from_numpy(col_np).long()
    edge_index = torch.stack([row, col], dim=0)

    y = torch.tensor(np.asarray(y).reshape(-1), dtype=torch.long)
    y = (y > 0).long()

    return Data(x=x, edge_index=edge_index, y=y, dataset_path=str(path))
