import copy
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, precision_recall_curve
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj
from tqdm.auto import tqdm

from auto_encoder import GraphAE
from diffusion_model import MLPDiffusion, Model, sample_dm_free
from pygod.metric.metric import eval_average_precision, eval_recall_at_k, eval_roc_auc
from utils import extract, get_noises, load_mat_data, softmax_with_temperature


sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = get_noises(timesteps=500)


class DiffGAD(BaseTransform):
    def __init__(
        self,
        name="",
        hid_dim=None,
        diff_dim=None,
        ae_epochs=300,
        diff_epochs=800,
        patience=100,
        lr=0.005,
        wd=0.0,
        weight=0.0,
        sample_steps=50,
        ae_dropout=0.3,
        ae_lr=0.05,
        ae_alpha=0.8,
        proto_alpha=None,
        data_dir="~/datasets/GAD/mat",
        save_dir="checkpoints",
        save_ckpt=False,
        num_trials=20,
        device=0,
    ):
        self.name = name
        self.hid_dim = hid_dim
        self.diff_dim = diff_dim
        self.ae_epochs = ae_epochs
        self.diff_epochs = diff_epochs
        self.patience = patience
        self.lr = lr
        self.wd = wd
        self.sample_steps = sample_steps
        self.weight = weight
        self.proto = None
        self.dm = None
        self.proto_alpha = proto_alpha
        self.ae = None
        self.ae_dropout = ae_dropout
        self.ae_lr = ae_lr
        self.ae_alpha = ae_alpha
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.timesteps = 500
        self.data_dir = data_dir
        self.save_dir_root = Path(save_dir)
        self.save_ckpt = save_ckpt
        self.num_trials = num_trials
        self.device = torch.device("cpu" if device < 0 or not torch.cuda.is_available() else "cuda")

    def forward(self, dset):
        self.dataset = dset
        data = load_mat_data(self.dataset, self.data_dir)

        if self.hid_dim is None:
            self.hid_dim = 2 ** int(math.log2(data.x.size(1)) - 1)
        if self.diff_dim is None:
            self.diff_dim = 2 * self.hid_dim

        self.ae = GraphAE(
            in_dim=data.num_node_features,
            hid_dim=self.hid_dim,
            dropout=self.ae_dropout,
        ).to(self.device)

        ae_state = self.train_ae(data)
        self.ae.load_state_dict(ae_state)

        dm_auc, dm_ap, dm_rec, dm_auprc = [], [], [], []
        final_state = None

        trial_bar = tqdm(range(self.num_trials), desc="Trials", dynamic_ncols=True)
        for _ in trial_bar:
            denoise_fn = MLPDiffusion(self.hid_dim, self.diff_dim).to(self.device)
            self.dm = Model(denoise_fn=denoise_fn, hid_dim=self.hid_dim).to(self.device)
            dm_state, self.proto = self.train_dm(data)
            self.dm.load_state_dict(dm_state)

            denoise_proto = MLPDiffusion(self.hid_dim, self.diff_dim).to(self.device)
            self.dm_proto = Model(denoise_fn=denoise_proto, hid_dim=self.hid_dim).to(self.device)
            proto_dm_state = self.train_dm_proto(data)
            self.dm_proto.load_state_dict(proto_dm_state)

            auc_this, ap_this, rec_this, auprc_this = self.sample(self.dm_proto, self.dm, data)
            dm_auc.append(auc_this)
            dm_ap.append(ap_this)
            dm_rec.append(rec_this)
            dm_auprc.append(auprc_this)
            trial_bar.set_postfix(auc=f"{auc_this:.4f}", ap=f"{ap_this:.4f}", rec=f"{rec_this:.4f}")

            final_state = {
                "ae": copy.deepcopy(self.ae.state_dict()),
                "dm": copy.deepcopy(self.dm.state_dict()),
                "dm_proto": copy.deepcopy(self.dm_proto.state_dict()),
                "prototype": self.proto.detach().cpu() if torch.is_tensor(self.proto) else self.proto,
                "params": self._params_dict(),
            }

        if self.save_ckpt and final_state is not None:
            save_dir = self.save_dir_root / str(self.dataset)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(final_state, save_dir / "best.pt")

        dm_auc = torch.tensor(dm_auc)
        dm_ap = torch.tensor(dm_ap)
        dm_rec = torch.tensor(dm_rec)
        dm_auprc = torch.tensor(dm_auprc)

        print(
            "Final AUC: {:.4f}±{:.4f} ({:.4f})\t"
            "Final AP: {:.4f}±{:.4f} ({:.4f})\t"
            "Final Recall: {:.4f}±{:.4f} ({:.4f})\t"
            "Final AUPRC: {:.4f}±{:.4f} ({:.4f})".format(
                torch.mean(dm_auc), torch.std(dm_auc), torch.max(dm_auc),
                torch.mean(dm_ap), torch.std(dm_ap), torch.max(dm_ap),
                torch.mean(dm_rec), torch.std(dm_rec), torch.max(dm_rec),
                torch.mean(dm_auprc), torch.std(dm_auprc), torch.max(dm_auprc),
            )
        )

    def train_ae(self, data):
        optimizer = torch.optim.Adam(self.ae.parameters(), self.ae_lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        auc_list, ap_list, rec_list, states = [], [], [], []

        trials = tqdm(range(self.num_trials), desc="AE trials", dynamic_ncols=True, leave=False)
        for _ in trials:
            epochs = tqdm(range(1, self.ae_epochs + 1), desc="AE epochs", dynamic_ncols=True, leave=False)
            last_auc = last_ap = last_rec = 0.0
            for epoch in epochs:
                self.ae.train()
                x = data.x.to(self.device, dtype=torch.float32)
                edge_index = data.edge_index.to(self.device)
                y = data.y.bool()
                s = to_dense_adj(edge_index)[0].to(self.device)

                x_, s_, _ = self.ae(x, edge_index)
                score = self.ae.loss_func(x, x_, s, s_, self.ae_alpha)
                loss = torch.mean(score)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                last_auc = eval_roc_auc(y, score.detach().cpu())
                last_ap = eval_average_precision(y, score.detach().cpu())
                last_rec = eval_recall_at_k(y, score.detach().cpu(), int(sum(y)))
                epochs.set_postfix(loss=f"{loss.item():.5f}", auc=f"{last_auc:.4f}")

            auc_list.append(last_auc)
            ap_list.append(last_ap)
            rec_list.append(last_rec)
            states.append(copy.deepcopy(self.ae.state_dict()))
            trials.set_postfix(auc=f"{last_auc:.4f}", ap=f"{last_ap:.4f}", rec=f"{last_rec:.4f}")

        best_id = int(np.argmax(auc_list))
        return states[best_id]

    def train_dm(self, data):
        optimizer = torch.optim.Adam(self.dm.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        self.dm.train()
        best_loss = float("inf")
        patience = 0
        proto = None
        best_state = copy.deepcopy(self.dm.state_dict())
        best_proto = None

        epochs = tqdm(range(self.diff_epochs), desc="DM epochs", dynamic_ncols=True, leave=False)
        for epoch in epochs:
            x = data.x.to(self.device, dtype=torch.float32)
            edge_index = data.edge_index.to(self.device)
            inputs = self.ae.encode(x, edge_index)

            if epoch == 0:
                proto = torch.mean(inputs, dim=0)

            loss, _, reconstructed = self.dm(inputs)
            loss = loss.mean()

            if epoch > 0:
                s_v = self.cos(proto, reconstructed)
                weight = softmax_with_temperature(s_v, t=5).reshape(1, -1)
                proto = torch.mm(weight, reconstructed).detach()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dm.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epochs.set_postfix(loss=f"{loss.item():.5f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
                best_state = copy.deepcopy(self.dm.state_dict())
                best_proto = proto.detach().clone()
            else:
                patience += 1
                if patience == self.patience:
                    break

        return best_state, best_proto

    def train_dm_proto(self, data):
        optimizer = torch.optim.Adam(self.dm_proto.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        self.dm_proto.train()
        best_loss = float("inf")
        patience = 0
        best_state = copy.deepcopy(self.dm_proto.state_dict())

        epochs = tqdm(range(self.diff_epochs), desc="Proto-DM epochs", dynamic_ncols=True, leave=False)
        for _ in epochs:
            x = data.x.to(self.device, dtype=torch.float32)
            edge_index = data.edge_index.to(self.device)
            inputs = self.ae.encode(x, edge_index)
            loss, _, _ = self.dm_proto(inputs, proto=self.proto, proto_alpha=self.proto_alpha)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dm_proto.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epochs.set_postfix(loss=f"{loss.item():.5f}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience = 0
                best_state = copy.deepcopy(self.dm_proto.state_dict())
            else:
                patience += 1
                if patience == self.patience:
                    break

        return best_state

    def sample(self, proto_model, free_model, data):
        self.ae.eval()
        proto_model.eval()
        free_model.eval()
        proto_net = proto_model.denoise_fn_D
        free_net = free_model.denoise_fn_D

        x = data.x.to(self.device, dtype=torch.float32)
        edge_index = data.edge_index.to(self.device)
        y = data.y.bool()
        z_0 = self.ae.encode(x, edge_index)
        noise = torch.randn_like(z_0)

        auc_pygod, ap, rec, auprc = [], [], [], []
        timesteps = tqdm(range(self.timesteps), desc="Sampling", dynamic_ncols=True, leave=False)
        with torch.no_grad():
            for i in timesteps:
                t = torch.tensor([i] * z_0.size(0), dtype=torch.long, device=self.device)
                sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, z_0.shape)
                sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, t, z_0.shape)
                z_t = sqrt_alphas_cumprod_t * z_0 + sqrt_one_minus_alphas_cumprod_t * noise

                reconstructed = sample_dm_free(
                    proto_net,
                    free_net,
                    z_t,
                    self.sample_steps,
                    proto=self.proto,
                    proto_alpha=self.proto_alpha,
                    weight=self.weight,
                )
                s = to_dense_adj(edge_index)[0].to(self.device)
                x_, s_ = self.ae.decode(reconstructed, edge_index)
                score = self.ae.loss_func(x, x_, s, s_, self.ae_alpha)

                pyg_auc = eval_roc_auc(y, score.detach().cpu())
                pyg_ap = eval_average_precision(y, score.detach().cpu())
                pyg_rec = eval_recall_at_k(y, score.detach().cpu(), int(sum(y)))
                p, r, _ = precision_recall_curve(y.numpy(), score.detach().cpu().numpy())
                pyg_auprc = auc(r, p)

                auc_pygod.append(pyg_auc)
                ap.append(pyg_ap)
                rec.append(pyg_rec)
                auprc.append(pyg_auprc)
                timesteps.set_postfix(auc=f"{pyg_auc:.4f}", ap=f"{pyg_ap:.4f}", rec=f"{pyg_rec:.4f}")

        return np.max(auc_pygod), np.max(ap), np.max(rec), np.max(auprc)

    def _params_dict(self):
        return {
            "dataset": self.dataset,
            "hid_dim": self.hid_dim,
            "diff_dim": self.diff_dim,
            "ae_epochs": self.ae_epochs,
            "diff_epochs": self.diff_epochs,
            "patience": self.patience,
            "lr": self.lr,
            "wd": self.wd,
            "weight": self.weight,
            "sample_steps": self.sample_steps,
            "ae_dropout": self.ae_dropout,
            "ae_lr": self.ae_lr,
            "ae_alpha": self.ae_alpha,
            "proto_alpha": self.proto_alpha,
            "data_dir": self.data_dir,
        }
