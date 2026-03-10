import logging
import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seeds(42)


@dataclass
class Feature:
    """Metadata describing a tabular feature."""

    name: str
    description: str
    dtype: str
    choices: Optional[List[str]] = None
    value_range: Optional[Tuple[float, float]] = None


@dataclass
class Example:
    """Single ASPIRE training/inference example."""

    features: List[Feature]
    values: List[Any]
    target_indices: List[int]
    dataset_context: str
    support_examples: Optional[List["Example"]] = None


def train_examples(
    model: nn.Module,
    examples: List["Example"],
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    random_state: Optional[int] = None,
    lr_head: Optional[float] = None,
    lr_backbone: Optional[float] = None,
    freeze_bert: bool = False,
    num_support: int = 0,
    val_fraction: float = 0.0,
    patience: int = 0,
    weight_decay: float = 0.01,
) -> nn.Module:
    """
    Train model on a list of ASPIRE examples.

    Supports both:
    - vanilla training (single learning rate)
    - finetuning-style training with split LRs, optional BERT freezing,
      support examples, validation split, and early stopping.
    """
    if not examples:
        raise ValueError("No training examples were provided.")

    if random_state is not None:
        set_seeds(random_state)

    train_examples_, val_examples_ = _split_examples(
        examples=examples,
        val_fraction=val_fraction,
        random_state=random_state,
    )

    if num_support > 0:
        _attach_support_examples(train_examples_, train_examples_, num_support=num_support)
        if val_examples_:
            _attach_support_examples(val_examples_, train_examples_, num_support=num_support)

    optimizer = _build_optimizer(
        model=model,
        learning_rate=learning_rate,
        lr_head=lr_head,
        lr_backbone=lr_backbone,
        freeze_bert=freeze_bert,
        weight_decay=weight_decay,
    )

    best_state = None
    best_metric = float("inf")
    epochs_without_improvement = 0

    for _ in range(max(1, num_epochs)):
        model.train()
        shuffled_examples = list(train_examples_)
        random.shuffle(shuffled_examples)
        epoch_loss = 0.0
        n_batches = 0

        for idx in range(0, len(shuffled_examples), max(1, batch_size)):
            batch = shuffled_examples[idx:idx + max(1, batch_size)]
            optimizer.zero_grad()
            loss = model(batch)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        train_metric = epoch_loss / max(1, n_batches)
        eval_metric = train_metric

        if val_examples_:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for idx in range(0, len(val_examples_), max(1, batch_size)):
                    batch = val_examples_[idx:idx + max(1, batch_size)]
                    loss = model(batch)
                    if torch.isnan(loss):
                        continue
                    val_loss += loss.item()
                    val_batches += 1
            eval_metric = val_loss / max(1, val_batches)

        if eval_metric < best_metric:
            best_metric = eval_metric
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if val_examples_ and patience > 0 and epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def _split_examples(
    examples: List["Example"],
    val_fraction: float = 0.0,
    random_state: Optional[int] = None,
) -> Tuple[List["Example"], List["Example"]]:
    """Split examples into train/validation lists."""
    if val_fraction <= 0:
        return list(examples), []
    if val_fraction >= 1:
        raise ValueError("val_fraction must be in [0, 1).")

    shuffled = list(examples)
    if random_state is not None:
        rng = random.Random(random_state)
        rng.shuffle(shuffled)
    else:
        random.shuffle(shuffled)

    n_total = len(shuffled)
    n_val = int(n_total * val_fraction)
    if n_total >= 5:
        n_val = max(1, n_val)
    else:
        n_val = 0

    if n_val <= 0:
        return shuffled, []
    if n_val >= n_total:
        n_val = n_total - 1

    return shuffled[n_val:], shuffled[:n_val]


def _attach_support_examples(
    query_examples: Sequence["Example"],
    pool_examples: Sequence["Example"],
    num_support: int = 5,
) -> None:
    """Attach randomly sampled support examples to each query example."""
    if num_support <= 0 or not query_examples or not pool_examples:
        return

    for example in query_examples:
        candidates = [candidate for candidate in pool_examples if candidate is not example]
        if not candidates:
            example.support_examples = []
            continue

        if len(candidates) < num_support:
            repeats = (num_support // len(candidates)) + 1
            candidates = list(candidates) * repeats

        support = random.sample(candidates, k=min(num_support, len(candidates)))
        example.support_examples = [
            Example(
                features=s.features,
                values=s.values,
                target_indices=[],
                dataset_context=s.dataset_context,
                support_examples=None,
            )
            for s in support
        ]


def _build_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-3,
    lr_head: Optional[float] = None,
    lr_backbone: Optional[float] = None,
    freeze_bert: bool = False,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """Build optimizer with optional split learning rates for finetuning."""
    resolved_lr_head = learning_rate if lr_head is None else lr_head
    resolved_lr_backbone = learning_rate if lr_backbone is None else lr_backbone

    if hasattr(model, "shared_text"):
        for parameter in model.shared_text.parameters():
            parameter.requires_grad = not freeze_bert
    else:
        for name, parameter in model.named_parameters():
            if "semantic_grounding.bert" in name or "context_bert" in name:
                parameter.requires_grad = not freeze_bert

    head_params = []
    backbone_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("reg_head.") or name.startswith("cls_head.") or name.startswith("prediction_heads."):
            head_params.append(parameter)
        else:
            backbone_params.append(parameter)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": resolved_lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": resolved_lr_head})

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


class MAB(nn.Module):
    """Multihead Attention Block."""

    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        h = self.num_heads
        dv = self.dim_V // h
        Q_ = torch.cat(Q.split(dv, dim=2), dim=0)
        K_ = torch.cat(K.split(dv, dim=2), dim=0)
        V_ = torch.cat(V.split(dv, dim=2), dim=0)
        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), dim=2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), dim=0), dim=2)
        O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O)
        return O


class ISAB(nn.Module):
    """Induced Set Attention Block."""

    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class SetTransformer(nn.Module):
    """Permutation-equivariant set encoder using ISAB layers."""

    def __init__(self, dim: int, num_heads: int = 4, num_inds: int = 32, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([ISAB(dim, dim, num_heads, num_inds) for _ in range(num_layers)])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            X = layer(X)
        return X


class SharedTextEncoder(nn.Module):
    """Shared BERT encoder for feature descriptions, values, and context."""

    def __init__(self, model_name: str = "bert-base-uncased", max_len_desc: int = 64, max_len_ctx: int = 128):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.eval()
        self.max_len_desc = max_len_desc
        self.max_len_ctx = max_len_ctx
        self._cache = {}

    @torch.no_grad()
    def encode_text(
        self,
        text: str,
        is_context: bool = False,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if text is None or (isinstance(text, float) and math.isnan(text)):
            text = "unknown"
        text = str(text)

        key = f"{'CTX' if is_context else 'TXT'}::{text}"
        if key in self._cache:
            out = self._cache[key]
            return out.to(device) if device is not None else out

        max_len = self.max_len_ctx if is_context else self.max_len_desc
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        bert_device = next(self.bert.parameters()).device
        inputs = {k: v.to(bert_device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        self._cache[key] = pooled.cpu()
        return pooled.to(device) if device is not None else pooled

    @torch.no_grad()
    def encode_text_sequence(
        self,
        text: str,
        is_context: bool = False,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode text and return token embeddings."""
        if text is None or (isinstance(text, float) and math.isnan(text)):
            text = "unknown"
        text = str(text)

        max_len = self.max_len_ctx if is_context else self.max_len_desc
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        bert_device = next(self.bert.parameters()).device
        inputs = {k: v.to(bert_device) for k, v in inputs.items()}
        outputs = self.bert(**inputs)
        tokens = outputs.last_hidden_state.squeeze(0)
        return tokens.to(device) if device is not None else tokens


class SemanticFeatureGrounding(nn.Module):
    """Semantic grounding for feature descriptors."""

    def __init__(self, model_dim: int, shared_text: SharedTextEncoder):
        super().__init__()
        self.shared_text = shared_text
        self.model_dim = model_dim
        bert_dim = self.shared_text.bert.config.hidden_size

        self.desc_proj = nn.Linear(bert_dim, model_dim)
        self.type_embed = nn.Embedding(2, model_dim)
        self.choices_proj = nn.Linear(bert_dim, model_dim)

    def forward(self, feature: Feature, device: torch.device) -> torch.Tensor:
        desc_emb = self.shared_text.encode_text(feature.description or feature.name, is_context=False, device=device)
        E_desc = self.desc_proj(desc_emb)

        type_idx = 0 if feature.dtype == "continuous" else 1
        E_dtype = self.type_embed(torch.tensor(type_idx, device=device))

        E_choices = torch.zeros(self.model_dim, device=device)
        if feature.dtype == "categorical" and feature.choices:
            choice_embs = []
            for choice in feature.choices:
                emb = self.shared_text.encode_text(str(choice), is_context=False, device=device)
                choice_embs.append(emb)
            choice_embs = torch.stack(choice_embs, dim=0)
            E_choices = self.choices_proj(choice_embs.mean(dim=0))

        return E_desc + E_dtype + E_choices

    def forward_tokens(self, feature: Feature, device: torch.device) -> torch.Tensor:
        """Return token-level feature description embeddings."""
        desc_text = feature.description or feature.name
        token_embs = self.shared_text.encode_text_sequence(desc_text, is_context=False, device=device)
        return self.desc_proj(token_embs)


class AtomProcessing(nn.Module):
    """Atom-level feature/value processing."""

    def __init__(self, model_dim: int, shared_text: SharedTextEncoder, num_fourier_features: int = 256):
        super().__init__()
        self.model_dim = model_dim
        self.shared_text = shared_text

        freqs = torch.logspace(np.log10(0.1), np.log10(10.0), num_fourier_features)
        self.register_buffer("fourier_freqs", freqs)
        self.fourier_proj = nn.Linear(num_fourier_features, model_dim)

        self.missing_embed = nn.Parameter(torch.randn(model_dim))

        bert_dim = self.shared_text.bert.config.hidden_size
        self.cat_proj = nn.Linear(bert_dim, model_dim)

        self.atom_mlp = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
        )

    def _encode_value(self, feature: Feature, value: Any, device: torch.device) -> torch.Tensor:
        if value is None or (isinstance(value, float) and (math.isnan(value) or not math.isfinite(value))):
            return self.missing_embed

        if feature.dtype == "continuous":
            try:
                val = float(value)
            except Exception:
                return self.missing_embed

            if feature.value_range:
                vmin, vmax = feature.value_range
                if vmax <= vmin:
                    norm = 0.5
                else:
                    norm = (val - vmin) / max(1e-8, (vmax - vmin))
                    norm = float(np.clip(norm, 0.0, 1.0))
            else:
                norm = (math.tanh(val / 100.0) + 1.0) / 2.0

            val_tensor = torch.tensor(norm, device=device, dtype=torch.float32)
            fourier_feat = torch.cos(2.0 * math.pi * self.fourier_freqs * val_tensor)
            return self.fourier_proj(fourier_feat)

        txt = str(value)
        emb = self.shared_text.encode_text(txt, is_context=False, device=device)
        return self.cat_proj(emb)

    def forward(
        self,
        feature: Feature,
        value: Any,
        phi_feature: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        nu_value = self._encode_value(feature, value, device)
        scale = self.model_dim ** 0.5
        phi_n = F.normalize(phi_feature, dim=-1) * scale
        nu_n = F.normalize(nu_value, dim=-1) * scale
        combined = torch.cat([phi_n, nu_n], dim=-1)
        atom_emb = self.atom_mlp(combined)
        return atom_emb


class IntraInstanceSet2Set(nn.Module):
    """Optional intra-instance set transformer."""

    def __init__(self, model_dim: int, num_heads: int = 4, num_inds: int = 16, num_layers: int = 2):
        super().__init__()
        self.set_transformer = SetTransformer(
            dim=model_dim,
            num_heads=num_heads,
            num_inds=num_inds,
            num_layers=num_layers,
        )

    def forward(self, atom_embeddings: torch.Tensor) -> torch.Tensor:
        if atom_embeddings.dim() == 2:
            atom_embeddings = atom_embeddings.unsqueeze(0)
            return self.set_transformer(atom_embeddings).squeeze(0)
        return self.set_transformer(atom_embeddings)


class InterInstanceAggregator(nn.Module):
    """Inter-instance aggregation module for conditioning."""

    def __init__(self, model_dim: int, bert_dim: int, num_heads: int = 8, num_inds: int = 32, num_layers: int = 2):
        super().__init__()
        self.model_dim = model_dim

        self.type_embeddings = nn.ParameterDict(
            {
                "query": nn.Parameter(torch.randn(model_dim) * 0.02),
                "target": nn.Parameter(torch.randn(model_dim) * 0.02),
                "shot": nn.Parameter(torch.randn(model_dim) * 0.02),
                "context_data": nn.Parameter(torch.randn(model_dim) * 0.02),
                "context_target": nn.Parameter(torch.randn(model_dim) * 0.02),
            }
        )

        self.cls_token = nn.Parameter(torch.randn(model_dim) * 0.02)
        self.context_proj = nn.Linear(bert_dim, model_dim)

        self.aggregator = SetTransformer(
            dim=model_dim,
            num_heads=num_heads,
            num_inds=num_inds,
            num_layers=num_layers,
        )

    def forward(
        self,
        query_atoms: torch.Tensor,
        target_atoms: torch.Tensor,
        support_atoms: List[torch.Tensor],
        context_data_tokens: torch.Tensor,
        context_target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        tokens: List[torch.Tensor] = []

        for i in range(query_atoms.size(0)):
            tokens.append(query_atoms[i] + self.type_embeddings["query"])

        for i in range(target_atoms.size(0)):
            combined = target_atoms[i] + self.cls_token + self.type_embeddings["target"]
            tokens.append(combined)

        for support_inst in support_atoms:
            for j in range(support_inst.size(0)):
                tokens.append(support_inst[j] + self.type_embeddings["shot"])

        if context_data_tokens.size(0) > 0:
            ctx_data = self.context_proj(context_data_tokens)
            for i in range(ctx_data.size(0)):
                tokens.append(ctx_data[i] + self.type_embeddings["context_data"])

        if context_target_tokens.size(0) > 0:
            ctx_target = self.context_proj(context_target_tokens)
            for i in range(ctx_target.size(0)):
                tokens.append(ctx_target[i] + self.type_embeddings["context_target"])

        all_tokens = torch.stack(tokens, dim=0).unsqueeze(0)
        aggregated = self.aggregator(all_tokens).squeeze(0)

        num_query = query_atoms.size(0)
        num_targets = target_atoms.size(0)
        rt = aggregated[num_query:num_query + num_targets]

        return rt


class MoGRegressionHead(nn.Module):
    """Mixture-of-Gaussians regression head."""

    def __init__(self, model_dim: int, K: int = 3):
        super().__init__()
        self.K = K
        self.pi = nn.Linear(model_dim, K)
        self.mu = nn.Linear(model_dim, K)
        self.logvar = nn.Linear(model_dim, K)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi = F.softmax(self.pi(h), dim=-1)
        mu = torch.sigmoid(self.mu(h))
        logvar = self.logvar(h)
        return pi, mu, logvar

    def nll(self, h: torch.Tensor, target_norm: torch.Tensor) -> torch.Tensor:
        pi, mu, logvar = self.forward(h)
        var = torch.exp(logvar) + 1e-8
        y = target_norm.unsqueeze(-1)
        log_probs = -0.5 * (torch.log(2 * torch.pi * var) + (y - mu) ** 2 / var)
        log_mix = torch.logsumexp(torch.log(pi + 1e-12) + log_probs, dim=-1)
        return -log_mix.mean()

    @torch.no_grad()
    def predict(self, h: torch.Tensor) -> torch.Tensor:
        pi, mu, _ = self.forward(h)
        return (pi * mu).sum(dim=-1)


class ClassificationHead(nn.Module):
    """Classification head with temperature scaling."""

    def __init__(self, model_dim: int, bert_hidden: int):
        super().__init__()
        self.category_proj = nn.Linear(bert_hidden, model_dim)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(10.0)))

    def logits(
        self,
        h: torch.Tensor,
        label_embs: torch.Tensor,
        context_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h_n = F.normalize(h, dim=-1)

        l_n = F.normalize(label_embs, dim=-1)
        temp = torch.exp(self.log_temperature)
        return h_n @ l_n.T * temp


class ASPIREEnhanced(nn.Module):
    """ASPIRE v2 architecture compatible with old checkpoints."""

    def __init__(
        self,
        model_dim: int = 768,
        num_heads: int = 8,
        num_inds: int = 32,
        mask_prob: float = 0.40,
        max_targets: int = 3,
        intra_layers: int = 2,
        inter_layers: int = 2,
        shared_bert: str = "bert-base-uncased",
        use_intra_set2set: bool = True,
        use_dataset_description: bool = True,
        use_echoices: bool = True,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.mask_prob = mask_prob
        self.max_targets = max_targets
        self._label_smoothing = 0.1
        self._cls_loss_weight = 2.0
        self.use_intra_set2set = use_intra_set2set
        self.use_dataset_description = use_dataset_description
        self.use_echoices = use_echoices

        self.shared_text = SharedTextEncoder(shared_bert)
        bert_dim = self.shared_text.bert.config.hidden_size

        self.semantic_grounding = SemanticFeatureGrounding(model_dim, self.shared_text)
        self.atom_processing = AtomProcessing(model_dim, self.shared_text)

        if use_intra_set2set:
            self.intra_set2set = IntraInstanceSet2Set(
                model_dim=model_dim,
                num_heads=max(1, num_heads // 2),
                num_inds=max(1, num_inds // 2),
                num_layers=intra_layers,
            )
        else:
            self.intra_set2set = None

        self.inter_aggregator = InterInstanceAggregator(
            model_dim=model_dim,
            bert_dim=bert_dim,
            num_heads=num_heads,
            num_inds=num_inds,
            num_layers=inter_layers,
        )

        self.reg_head = MoGRegressionHead(model_dim, K=3)
        self.cls_head = ClassificationHead(model_dim, bert_dim)

        self.mask_token = nn.Parameter(torch.randn(model_dim))

    def _select_targets(self, num_features: int, existing: Optional[List[int]] = None) -> List[int]:
        if existing and len(existing) > 0:
            selected = list(existing)
        else:
            selected = [random.randint(0, num_features - 1)]
        remaining = [i for i in range(num_features) if i not in selected]
        add = max(0, min(random.randint(1, self.max_targets) - len(selected), len(remaining)))
        if add > 0:
            selected.extend(random.sample(remaining, add))
        return selected

    def _mask_atoms(self, atoms: torch.Tensor, target_indices: List[int]) -> torch.Tensor:
        if not self.training:
            return atoms
        out = atoms.clone()
        non_target_indices = [i for i in range(atoms.size(0)) if i not in target_indices]
        kept = 0
        for i in non_target_indices:
            if random.random() < self.mask_prob:
                out[i] = self.mask_token
            else:
                kept += 1
        if non_target_indices and kept == 0:
            keep_idx = random.choice(non_target_indices)
            out[keep_idx] = atoms[keep_idx]
        return out

    def _process_instance_atoms(
        self,
        features: List[Feature],
        values: List[Any],
        device: torch.device,
    ) -> torch.Tensor:
        phi = [self.semantic_grounding(f, device) for f in features]
        atoms = [
            self.atom_processing(f, v, phi[i], device)
            for i, (f, v) in enumerate(zip(features, values))
        ]
        return torch.stack(atoms, dim=0)

    def _apply_intra_set2set(self, atoms: torch.Tensor) -> torch.Tensor:
        if self.use_intra_set2set and self.intra_set2set is not None:
            return self.intra_set2set(atoms)
        return atoms

    def _encode_context(self, context_text: str, device: torch.device) -> torch.Tensor:
        if not self.use_dataset_description or not context_text:
            return torch.zeros(1, self.shared_text.bert.config.hidden_size, device=device)
        return self.shared_text.encode_text_sequence(context_text, is_context=True, device=device)

    def _encode_target_description(
        self,
        features: List[Feature],
        target_indices: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        descriptions = []
        for idx in target_indices:
            descriptions.append(features[idx].description or features[idx].name)
        combined_desc = " ".join(descriptions)
        return self.shared_text.encode_text_sequence(combined_desc, is_context=False, device=device)

    def _build_desc_context_tokens(
        self,
        context_data_tokens: torch.Tensor,
    ) -> torch.Tensor:
        return context_data_tokens

    def forward(self, batch: List[Example]) -> torch.Tensor:
        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        total_targets = 0

        for example in batch:
            if not example.features:
                continue

            if self.training:
                targets = self._select_targets(len(example.features), example.target_indices)
            else:
                targets = list(example.target_indices)

            if not targets:
                continue

            inst_atoms = self._process_instance_atoms(example.features, example.values, device)
            if self.training:
                inst_atoms = self._mask_atoms(inst_atoms, targets)
            inst_atoms = self._apply_intra_set2set(inst_atoms)

            observed_idx = [i for i in range(len(example.features)) if i not in targets]
            query_atoms = inst_atoms[observed_idx]
            target_atoms = inst_atoms[targets]

            support_atoms_list = []
            if example.support_examples:
                for sup in example.support_examples:
                    sup_atoms = self._process_instance_atoms(sup.features, sup.values, device)
                    sup_atoms = self._apply_intra_set2set(sup_atoms)
                    support_atoms_list.append(sup_atoms)

            context_data_tokens = self._encode_context(example.dataset_context, device)
            context_target_tokens = self._encode_target_description(example.features, targets, device)
            desc_ctx_tokens = self._build_desc_context_tokens(context_data_tokens)

            target_reprs = self.inter_aggregator(
                query_atoms=query_atoms,
                target_atoms=target_atoms,
                support_atoms=support_atoms_list,
                context_data_tokens=context_data_tokens,
                context_target_tokens=context_target_tokens,
            )

            for ti, feat_idx in enumerate(targets):
                feature = example.features[feat_idx]
                value = example.values[feat_idx]
                h = target_reprs[ti]

                if feature.dtype == "continuous":
                    try:
                        raw = float(value)
                    except Exception:
                        continue

                    if feature.value_range:
                        vmin, vmax = feature.value_range
                        if vmax <= vmin:
                            tnorm = 0.5
                        else:
                            tnorm = (raw - vmin) / max(1e-8, (vmax - vmin))
                            tnorm = float(np.clip(tnorm, 0.0, 1.0))
                    else:
                        tnorm = (math.tanh(raw / 100.0) + 1.0) / 2.0

                    target_norm = torch.tensor([tnorm], device=device, dtype=torch.float32)
                    loss = self.reg_head.nll(h.unsqueeze(0), target_norm)
                else:
                    if not feature.choices:
                        continue

                    categories = list(feature.choices)
                    if "[UNK]" not in categories:
                        categories = categories + ["[UNK]"]

                    label_vecs = []
                    for cat in categories:
                        emb = self.shared_text.encode_text(str(cat), is_context=False, device=device)
                        label_vecs.append(emb)
                    label_vecs = torch.stack(label_vecs, dim=0)
                    label_embs = self.cls_head.category_proj(label_vecs)

                    try:
                        idx = categories.index(str(value))
                    except ValueError:
                        idx = categories.index("[UNK]")

                    logits = self.cls_head.logits(
                        h.unsqueeze(0),
                        label_embs,
                        context_tokens=desc_ctx_tokens,
                    )
                    target_tensor = torch.tensor([idx], device=device, dtype=torch.long)
                    loss = F.cross_entropy(logits, target_tensor, label_smoothing=self._label_smoothing)
                    loss = loss * self._cls_loss_weight

                total_loss = total_loss + loss
                total_targets += 1

        return total_loss / max(1, total_targets)

    @torch.no_grad()
    def predict(self, example: Example, return_probs: bool = False):
        """Predict target values for an example."""
        self.eval()
        device = next(self.parameters()).device

        inst_atoms = self._process_instance_atoms(example.features, example.values, device)
        inst_atoms = self._apply_intra_set2set(inst_atoms)

        query_idx = [i for i in range(len(example.features)) if i not in example.target_indices]
        query_atoms = inst_atoms[query_idx]
        target_atoms = inst_atoms[example.target_indices]

        support_atoms_list = []
        if example.support_examples:
            for support in example.support_examples:
                sup_atoms = self._process_instance_atoms(support.features, support.values, device)
                sup_atoms = self._apply_intra_set2set(sup_atoms)
                support_atoms_list.append(sup_atoms)

        context_data_tokens = self._encode_context(example.dataset_context, device)
        context_target_tokens = self._encode_target_description(example.features, example.target_indices, device)
        desc_ctx_tokens = self._build_desc_context_tokens(context_data_tokens)

        target_reprs = self.inter_aggregator(
            query_atoms=query_atoms,
            target_atoms=target_atoms,
            support_atoms=support_atoms_list,
            context_data_tokens=context_data_tokens,
            context_target_tokens=context_target_tokens,
        )

        preds = []
        probs_list = []
        for ti, feat_idx in enumerate(example.target_indices):
            feature = example.features[feat_idx]
            h = target_reprs[ti]

            if feature.dtype == "continuous":
                y_norm = self.reg_head.predict(h.unsqueeze(0)).squeeze(0).item()
                if feature.value_range:
                    vmin, vmax = feature.value_range
                    pred = y_norm * (vmax - vmin) + vmin
                else:
                    pred = (y_norm - 0.5) * 200.0
                preds.append(float(pred))
                probs_list.append(None)
            else:
                categories = list(feature.choices) if feature.choices else []
                if "[UNK]" not in categories:
                    categories = categories + ["[UNK]"]

                label_vecs = []
                for cat in categories:
                    emb = self.shared_text.encode_text(str(cat), is_context=False, device=device)
                    label_vecs.append(emb)
                label_vecs = torch.stack(label_vecs, dim=0)
                label_embs = self.cls_head.category_proj(label_vecs)

                logits = self.cls_head.logits(
                    h.unsqueeze(0),
                    label_embs,
                    context_tokens=desc_ctx_tokens,
                )
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                idx = int(torch.argmax(probs).item())
                preds.append(categories[idx])
                probs_list.append(probs.detach().cpu().numpy())

        return (preds, probs_list) if return_probs else preds
