import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

D_MODEL = 384
E5_SMALL = "intfloat/e5-small-v2"
N_REG_BINS = 100


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class NumericalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar_embedder = nn.Sequential(
            nn.Linear(1, D_MODEL * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D_MODEL * 2, D_MODEL),
        )
        self.fusion_block = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=2, dim_feedforward=D_MODEL * 4,
            dropout=0.1, activation="relu", batch_first=True, norm_first=True,
        )

    def forward(self, textual_embeddings: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = textual_embeddings.shape
        num_embeddings = self.scalar_embedder(x_num.unsqueeze(-1))
        fusion_input = torch.stack([textual_embeddings, num_embeddings], dim=2)
        fusion_input = fusion_input.view(batch_size * seq_len, 2, d_model)
        fused = self.fusion_block(fusion_input)
        return fused.view(batch_size, seq_len, 2, d_model).mean(dim=2)


class MAB(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        normed_q = self.norm1(Q)
        normed_k = self.norm1(K)
        Q = Q + self.attn(normed_q, normed_k, normed_k)[0]
        Q = Q + self.ff(self.norm2(Q))
        return Q


class SAB(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.mab = MAB(d_model, nhead, dim_feedforward, dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class NumericalFusionSetTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scalar_embedder = nn.Sequential(
            nn.Linear(1, D_MODEL * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(D_MODEL * 2, D_MODEL),
        )
        self.fusion_block = SAB(
            d_model=D_MODEL, nhead=2, dim_feedforward=D_MODEL * 4, dropout=0.1,
        )

    def forward(self, textual_embeddings: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = textual_embeddings.shape
        num_embeddings = self.scalar_embedder(x_num.unsqueeze(-1))
        fusion_input = torch.stack([textual_embeddings, num_embeddings], dim=2)
        fusion_input = fusion_input.view(batch_size * seq_len, 2, d_model)
        fused = self.fusion_block(fusion_input)
        return fused.view(batch_size, seq_len, 2, d_model).mean(dim=2)


class InteractionEncoder(nn.Module):
    def __init__(self, num_layers: int = 6, d_model: int = D_MODEL,
                 num_heads_factor: int = 64, ffn_d_hidden_multiplier: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=d_model // num_heads_factor,
            dim_feedforward=d_model * ffn_d_hidden_multiplier,
            dropout=dropout, activation="relu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PredictionHead(nn.Module):
    def __init__(self, input_size: int = D_MODEL):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, input_size * 4),
            nn.ReLU(),
            nn.Linear(input_size * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _freeze_text_encoder(text_encoder, layers_to_unfreeze: int = 0):
    if layers_to_unfreeze == 0:
        for p in text_encoder.parameters():
            p.requires_grad = False
        return
    for p in text_encoder.pooler.parameters():
        p.requires_grad = True
    for p in text_encoder.embeddings.parameters():
        p.requires_grad = False
    n_layers = text_encoder.config.num_hidden_layers
    unfreeze_indices = set(range(n_layers - layers_to_unfreeze, n_layers))
    for name, p in text_encoder.encoder.named_parameters():
        layer_num = None
        for part in name.split("."):
            if part.isdigit():
                layer_num = int(part)
                break
        p.requires_grad = (layer_num in unfreeze_indices)


class DiscretizedHead(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_bins: int = N_REG_BINS):
        super().__init__()
        self.cls_head = PredictionHead(d_model)
        self.reg_proj = nn.Linear(d_model, n_bins)

    def forward_cls(self, target_tokens: torch.Tensor) -> torch.Tensor:
        return self.cls_head(target_tokens).squeeze(dim=-1)

    def forward_reg(self, target_token: torch.Tensor) -> torch.Tensor:
        return self.reg_proj(target_token.squeeze(1))


class MoGHead(nn.Module):
    def __init__(self, d_model: int = D_MODEL, K: int = 10):
        super().__init__()
        self.K = K
        self.cls_head = PredictionHead(d_model)
        self.pi     = nn.Linear(d_model, K)
        self.mu     = nn.Linear(d_model, K)
        self.logvar = nn.Linear(d_model, K)

    def forward_cls(self, target_tokens: torch.Tensor) -> torch.Tensor:
        return self.cls_head(target_tokens).squeeze(dim=-1)

    def forward_reg(self, target_token: torch.Tensor) -> torch.Tensor:
        h = target_token.squeeze(1)
        return torch.cat([self.pi(h), self.mu(h), self.logvar(h)], dim=-1)


class RowPooler(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, row_tokens: torch.Tensor) -> torch.Tensor:
        B = row_tokens.size(0)
        q = self.query.expand(B, -1, -1)
        pooled, _ = self.attn(q, row_tokens, row_tokens)
        return self.norm(pooled.squeeze(1))


class Backbone(nn.Module):
    def __init__(self, d_model: int = D_MODEL, num_interaction_layers: int = 6,
                 unfreeze_layers: int = 3, n_reg_bins: int = N_REG_BINS,
                 reg_head: str = "discretized", n_mog: int = 10):
        super().__init__()
        self.d_model = d_model
        self.n_reg_bins = n_reg_bins
        self.reg_head_type = reg_head

        e5_path = os.getenv("E5_SMALL_LOCAL_PATH") or E5_SMALL
        self.text_encoder = AutoModel.from_pretrained(e5_path)
        self.tokenizer    = AutoTokenizer.from_pretrained(e5_path)
        self.numerical_fusion = NumericalFusion()
        self.tabular_encoder  = InteractionEncoder(num_layers=num_interaction_layers)

        if reg_head == "mog":
            self.head = MoGHead(d_model=d_model, K=n_mog)
        else:
            self.head = DiscretizedHead(d_model=d_model, n_bins=n_reg_bins)

        self.row_pooler = RowPooler(d_model=d_model)

        self.support_type_emb = nn.Parameter(torch.zeros(d_model))
        self.query_type_emb   = nn.Parameter(torch.zeros(d_model))
        self.label_type_emb   = nn.Parameter(torch.zeros(d_model))

        self.desc_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.desc_proj.weight, std=0.01)
        nn.init.zeros_(self.desc_proj.bias)

        _freeze_text_encoder(self.text_encoder, layers_to_unfreeze=unfreeze_layers)
        self._txt_cache: Dict[str, torch.Tensor] = {}
        self._unfreeze_layers = unfreeze_layers

    def _encode_texts_train(self, x_txt: np.ndarray) -> torch.Tensor:
        batch_size, seq_len = x_txt.shape
        unique_texts, inv = np.unique(x_txt, return_inverse=True)
        all_embs = []
        for i in range(0, len(unique_texts), 128):
            batch = unique_texts[i:i+128].tolist()
            inp = self.tokenizer(batch, padding=True, return_tensors="pt", truncation=True)
            inp = {k: v.to(next(self.text_encoder.parameters()).device) for k, v in inp.items()}
            out = self.text_encoder(**inp)
            all_embs.append(out.last_hidden_state[:, 0, :])
        all_embs = torch.cat(all_embs, dim=0)
        inv_t = torch.tensor(inv, dtype=torch.long, device=all_embs.device)
        return all_embs[inv_t].view(batch_size, seq_len, -1)

    @torch.no_grad()
    def _encode_texts_cached(self, x_txt: np.ndarray) -> torch.Tensor:
        batch_size, seq_len = x_txt.shape
        device = next(self.text_encoder.parameters()).device
        unique_texts, inv = np.unique(x_txt, return_inverse=True)
        uncached = [t for t in unique_texts if t not in self._txt_cache]
        if uncached:
            for i in range(0, len(uncached), 128):
                batch = uncached[i:i+128]
                inp = self.tokenizer(list(batch), padding=True, return_tensors="pt", truncation=True)
                inp = {k: v.to(device) for k, v in inp.items()}
                cls_embs = self.text_encoder(**inp).last_hidden_state[:, 0, :]
                for j, t in enumerate(batch):
                    self._txt_cache[t] = cls_embs[j].detach().cpu()
        all_embs = torch.stack([self._txt_cache[t] for t in unique_texts]).to(device)
        inv_t = torch.tensor(inv, dtype=torch.long, device=device)
        return all_embs[inv_t].view(batch_size, seq_len, -1)

    def _encode_texts(self, x_txt: np.ndarray) -> torch.Tensor:
        if self.training and self._unfreeze_layers > 0:
            return self._encode_texts_train(x_txt)
        return self._encode_texts_cached(x_txt)

    def _pool_support(self, support_x_txt: np.ndarray, support_x_num: np.ndarray,
                      K: int) -> torch.Tensor:
        B, _, M_sup = support_x_txt.shape
        flat_txt = support_x_txt.reshape(B * K, M_sup)
        flat_num = support_x_num.reshape(B * K, M_sup)
        text_emb = self._encode_texts(flat_txt)
        device   = text_emb.device
        num_t    = torch.as_tensor(flat_num, dtype=text_emb.dtype, device=device)
        row_emb  = self.numerical_fusion(textual_embeddings=text_emb, x_num=num_t)
        label_mask = torch.zeros(row_emb.size(1), 1, device=device, dtype=row_emb.dtype)
        label_mask[-1] = 1.0
        row_emb = row_emb + label_mask.unsqueeze(0) * self.label_type_emb
        pooled  = self.row_pooler(row_emb) + self.support_type_emb
        return pooled.view(B, K, -1)

    def forward(
        self,
        x_txt: np.ndarray,
        x_num: np.ndarray,
        d_output: int,
        target_type: str,
        support_x_txt: Optional[np.ndarray] = None,
        support_x_num: Optional[np.ndarray] = None,
        desc_txt: Optional[np.ndarray] = None,
        return_encoded: bool = False,
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        text_emb  = self._encode_texts(x_txt)
        num_t     = torch.as_tensor(x_num, dtype=text_emb.dtype, device=device)
        query_emb = self.numerical_fusion(textual_embeddings=text_emb, x_num=num_t)
        query_emb = query_emb + self.query_type_emb

        desc_cond = None
        if desc_txt is not None:
            desc_arr  = np.array(desc_txt, dtype=object).reshape(-1, 1)
            desc_emb  = self._encode_texts_cached(desc_arr)
            desc_cond = self.desc_proj(desc_emb.squeeze(1))
            query_emb = query_emb + desc_cond.unsqueeze(1)

        seq_len = query_emb.size(1)

        if support_x_txt is not None and support_x_txt.shape[1] > 0:
            K          = support_x_txt.shape[1]
            sup_pooled = self._pool_support(support_x_txt, support_x_num, K)
            if desc_cond is not None:
                sup_pooled = sup_pooled + desc_cond.unsqueeze(1)
            combined = torch.cat([query_emb, sup_pooled], dim=1)
        else:
            combined = query_emb

        encoded       = self.tabular_encoder(combined)
        query_encoded = encoded[:, :seq_len]
        target_tokens = query_encoded[:, :d_output]

        if target_type == "cat":
            scores = self.head.forward_cls(target_tokens)
        else:
            scores = self.head.forward_reg(target_tokens[:, :1])

        if return_encoded:
            return scores, query_encoded
        return scores


@dataclass
class DatasetBundle:
    name: str
    df: pd.DataFrame
    columns: List[str]
    col_types: Dict[str, str]
    feature_descs: Dict[str, str]
    dataset_desc: str
    num_scalers: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    reg_bin_edges: Dict[str, np.ndarray] = field(default_factory=dict)
    fixed_targets: List[str] = field(default_factory=list)


def _fit_reg_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    qs    = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(values.astype(np.float64), qs)
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([values.min() - 1e-6, values.max() + 1e-6])
    return edges


class LinearHead(nn.Module):
    def __init__(self, model_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(model_dim, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


def _decode_mog(preds_np: np.ndarray) -> np.ndarray:
    K  = preds_np.shape[-1] // 3
    pi = torch.softmax(torch.tensor(preds_np[:, :K]), dim=-1).numpy()
    mu = preds_np[:, K: 2 * K]
    return (pi * mu).sum(axis=1)


def _decode_bins(bin_preds: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bin_preds = np.clip(bin_preds, 0, len(edges) - 2)
    return (edges[bin_preds] + edges[bin_preds + 1]) / 2.0


PAD_TOKEN = "Predictive Feature: padding\nFeature Value: none"


def verbalize_cell(col_name: str, col_idx: int, value: Any,
                   feat_desc: str = None, anon: bool = False) -> str:
    if feat_desc:
        col_clean = feat_desc
    elif anon:
        col_clean = f"f{col_idx}"
    else:
        col_clean = str(col_name).replace("_", " ").strip()
    return f"Predictive Feature: {col_clean}\nFeature Value: {str(value).strip()}"


def verbalize_target_tokens(col_name: str, col_idx: int, col_type: str,
                             choices: Optional[List[str]] = None,
                             feat_desc: str = None, anon: bool = False) -> List[str]:
    if feat_desc:
        col_clean = feat_desc
    elif anon:
        col_clean = f"f{col_idx}"
    else:
        col_clean = str(col_name).replace("_", " ").strip()
    if col_type == "cat" and choices:
        return [f"Target Feature: {col_clean}\nFeature Value: {v}" for v in choices]
    return [f"Numerical Target Feature: {col_clean}"]


def _num_scale(val: Any, col: str, bundle: DatasetBundle) -> float:
    try:
        raw = float(val)
        mu, sigma = bundle.num_scalers.get(col, (0.0, 1.0))
        return float(np.clip((raw - mu) / sigma, -5.0, 5.0))
    except (ValueError, TypeError):
        return 0.0


def prepare_query_row(bundle: DatasetBundle, row: pd.Series, target_col: str,
                       target_choices: Optional[List[str]] = None,
                       mask_prob: float = 0.0,
                       use_feat_descs: bool = False,
                       anon_cols: bool = False) -> Tuple[np.ndarray, np.ndarray, int]:
    ttype = bundle.col_types[target_col]
    tfd = bundle.feature_descs.get(target_col) if use_feat_descs else None
    t_idx = bundle.columns.index(target_col) if target_col in bundle.columns else 0
    target_texts = verbalize_target_tokens(target_col, t_idx, ttype, target_choices,
                                            feat_desc=tfd, anon=anon_cols)
    d_output = len(target_texts)

    feat_texts, feat_nums = [], []
    for i, col in enumerate(bundle.columns):
        if col == target_col:
            continue
        val = row.get(col, np.nan)
        if pd.isna(val):
            continue
        if mask_prob > 0 and random.random() < mask_prob:
            continue
        fd = bundle.feature_descs.get(col) if use_feat_descs else None
        feat_texts.append(verbalize_cell(col, i, val, feat_desc=fd, anon=anon_cols))
        feat_nums.append(_num_scale(val, col, bundle) if bundle.col_types[col] == "num" else 0.0)

    texts = list(target_texts)
    nums = [0.0] * d_output
    texts.extend(feat_texts)
    nums.extend(feat_nums)
    return np.array(texts, dtype=object), np.array(nums, dtype=np.float32), d_output


def prepare_support_row(bundle: DatasetBundle, row: pd.Series,
                         target_col: str,
                         use_feat_descs: bool = False,
                         anon_cols: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    target_val = row.get(target_col, "unknown")
    texts, nums = [], []
    for i, col in enumerate(bundle.columns):
        if col == target_col:
            continue
        val = row.get(col, np.nan)
        if pd.isna(val):
            val = "unknown"
        fd = bundle.feature_descs.get(col) if use_feat_descs else None
        texts.append(verbalize_cell(col, i, val, feat_desc=fd, anon=anon_cols))
        nums.append(_num_scale(val, col, bundle) if bundle.col_types[col] == "num" else 0.0)
    tfd = bundle.feature_descs.get(target_col) if use_feat_descs else None
    t_idx = bundle.columns.index(target_col) if target_col in bundle.columns else 0
    if tfd:
        tc = tfd
    elif anon_cols:
        tc = f"f{t_idx}"
    else:
        tc = str(target_col).replace("_", " ").strip()
    if bundle.col_types[target_col] == "cat":
        label_txt = f"LABEL: {tc} = {str(target_val).strip()}"
        label_num = 0.0
    else:
        label_txt = f"LABEL: {tc}"
        label_num = _num_scale(target_val, target_col, bundle)
    texts.append(label_txt)
    nums.append(label_num)
    return np.array(texts, dtype=object), np.array(nums, dtype=np.float32)


def _bin_reg_target(val: float, bundle: DatasetBundle, col: str) -> int:
    edges = bundle.reg_bin_edges.get(col)
    if edges is None:
        return 0
    idx = int(np.digitize([val], edges[1:-1])[0])
    return max(0, min(idx, N_REG_BINS - 1))


def prepare_batch(bundle: DatasetBundle, rows: List[pd.Series], target_col: str,
                  support_rows: Optional[List[pd.Series]] = None,
                  mask_prob: float = 0.0,
                  include_desc: bool = False,
                  use_feat_descs: bool = False,
                  anon_cols: bool = False) -> Dict:
    ttype = bundle.col_types[target_col]
    if ttype == "cat":
        target_choices = sorted(bundle.df[target_col].dropna().astype(str).unique().tolist())
    else:
        target_choices = None

    q_txt, q_num, q_y, q_y_cont = [], [], [], []
    d_output = None
    for row in rows:
        xt, xn, d_out = prepare_query_row(bundle, row, target_col, target_choices,
                                           mask_prob=mask_prob,
                                           use_feat_descs=use_feat_descs,
                                           anon_cols=anon_cols)
        q_txt.append(xt)
        q_num.append(xn)
        d_output = d_out
        tv = row[target_col]
        if ttype == "cat":
            try:
                q_y.append(target_choices.index(str(tv)))
            except ValueError:
                q_y.append(0)
            q_y_cont.append(0.0)
        else:
            q_y.append(_bin_reg_target(float(tv), bundle, target_col))
            mu_s, sigma_s = bundle.num_scalers.get(target_col, (0.0, 1.0))
            q_y_cont.append((float(tv) - mu_s) / max(sigma_s, 1e-8))

    max_len = max(len(t) for t in q_txt)
    pad_txt, pad_num = [], []
    for t, n in zip(q_txt, q_num):
        if len(t) < max_len:
            pad = max_len - len(t)
            t = np.concatenate([t, np.array([PAD_TOKEN] * pad, dtype=object)])
            n = np.concatenate([n, np.zeros(pad, dtype=np.float32)])
        pad_txt.append(t)
        pad_num.append(n)

    B = len(rows)
    result = {
        "x_txt": np.stack(pad_txt),
        "x_num": np.stack(pad_num),
        "d_output": d_output,
        "y": np.array(q_y, dtype=np.int64),
        "y_cont": np.array(q_y_cont, dtype=np.float32),
        "target_type": ttype,
        "desc_txt": np.array([bundle.dataset_desc] * B, dtype=object) if include_desc else None,
    }

    if support_rows and len(support_rows) > 0:
        sup_txts, sup_nums = [], []
        for sr in support_rows:
            st, sn = prepare_support_row(bundle, sr, target_col,
                                          use_feat_descs=use_feat_descs,
                                          anon_cols=anon_cols)
            sup_txts.append(st)
            sup_nums.append(sn)
        max_sup = max(len(t) for t in sup_txts)
        pad_sup_txt, pad_sup_num = [], []
        for t, n in zip(sup_txts, sup_nums):
            if len(t) < max_sup:
                pad = max_sup - len(t)
                label_t, label_n = t[-1:], n[-1:]
                body_t, body_n = t[:-1], n[:-1]
                body_t = np.concatenate([body_t, np.array([PAD_TOKEN] * pad, dtype=object)])
                body_n = np.concatenate([body_n, np.zeros(pad, dtype=np.float32)])
                t = np.concatenate([body_t, label_t])
                n = np.concatenate([body_n, label_n])
            pad_sup_txt.append(t)
            pad_sup_num.append(n)
        sup_txt_arr = np.stack(pad_sup_txt)
        sup_num_arr = np.stack(pad_sup_num)
        result["support_x_txt"] = np.broadcast_to(sup_txt_arr[None], (B,) + sup_txt_arr.shape).copy()
        result["support_x_num"] = np.broadcast_to(sup_num_arr[None], (B,) + sup_num_arr.shape).copy()
    else:
        result["support_x_txt"] = None
        result["support_x_num"] = None
    return result


def compute_loss(predictions: torch.Tensor, y: np.ndarray, target_type: str,
                 device: torch.device, label_smoothing: float = 0.0,
                 reg_head: str = "discretized", y_cont: np.ndarray = None) -> torch.Tensor:
    if target_type == "num" and reg_head == "mog":
        K = predictions.shape[-1] // 3
        pi     = F.softmax(predictions[:, :K], dim=-1)
        mu     = predictions[:, K:2*K]
        logvar = predictions[:, 2*K:].clamp(-10, 10)
        sigma2 = torch.exp(logvar).clamp(min=1e-6)
        y_t    = torch.tensor(y_cont, dtype=torch.float32, device=device)
        log_g  = -0.5 * ((y_t.unsqueeze(1) - mu) ** 2 / sigma2
                         + logvar + math.log(2 * math.pi))
        nll    = -torch.logsumexp(torch.log(pi + 1e-8) + log_g, dim=1)
        return nll.mean()
    y_t = torch.tensor(y, dtype=torch.long, device=device)
    return F.cross_entropy(predictions, y_t, label_smoothing=label_smoothing)
