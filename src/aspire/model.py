import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import math
import numpy as np
import pandas as pd
import random
import glob
import logging
import argparse
import os
import json
from typing import List, Optional
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seeds()

# Data Structures
@dataclass
class Feature:
    name: str
    description: str
    dtype: str
    choices: Optional[List[str]] = None
    value_range: Optional[tuple] = None

@dataclass
class Example:
    features: List[Feature]
    values: List
    target_indices: List[int]
    dataset_context: str
    support_examples: Optional[List] = None

# Set Transformer Components
class MAB(nn.Module):
    """Multihead Attention Block"""
    def __init__(self, dim_Q, dim_K, dim_V, num_heads):
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.ln0 = nn.LayerNorm(dim_V)
        self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)
        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O)
        return O

class ISAB(nn.Module):
    """Induced Set Attention Block"""
    def __init__(self, dim_in, dim_out, num_heads, num_inds):
        super().__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class SetTransformer(nn.Module):
    """Set Transformer for permutation-equivariant processing"""
    def __init__(self, dim, num_heads=4, num_inds=32, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            ISAB(dim, dim, num_heads, num_inds) for _ in range(num_layers)
        ])
    
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

# ASPIRE Core Components
class SemanticFeatureGrounding(nn.Module):
    """Semantic Feature Grounding with BERT"""
    def __init__(self, model_dim=768, bert_model='bert-base-uncased'):
        super().__init__()
        self.model_dim = model_dim
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model)
        self.bert.eval()
        bert_dim = self.bert.config.hidden_size
        self.desc_proj = nn.Linear(bert_dim, model_dim)
        self.type_embed = nn.Embedding(2, model_dim)
        self.choice_proj = nn.Linear(bert_dim, model_dim)
        self._text_cache = {}
        
    def _encode_text(self, text: str) -> torch.Tensor:
        device = next(self.parameters()).device
        if text is None or (isinstance(text, float) and math.isnan(text)):
            text = "unknown"
        else:
            text = str(text)
        if text in self._text_cache:
            return self._text_cache[text].to(device)
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.bert(**inputs)
            pooled = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            self._text_cache[text] = pooled.cpu()
        return pooled
    
    def forward(self, feature: Feature) -> torch.Tensor:
        device = next(self.parameters()).device
        desc_emb = self._encode_text(feature.description)
        E_desc = self.desc_proj(desc_emb)
        type_idx = 0 if feature.dtype == 'continuous' else 1
        E_dtype = self.type_embed(torch.tensor(type_idx, device=device))
        E_choices = torch.zeros(self.model_dim, device=device)
        phi = E_desc + E_dtype + E_choices
        return phi

class AtomProcessing(nn.Module):
    """Feature-Value Atom Processing"""
    def __init__(self, model_dim=768, num_fourier_features=256):
        super().__init__()
        self.model_dim = model_dim
        freqs = torch.logspace(np.log10(0.1), np.log10(10.0), num_fourier_features)
        self.register_buffer('fourier_freqs', freqs)
        self.fourier_proj = nn.Linear(num_fourier_features, model_dim)
        self.missing_embed = nn.Parameter(torch.randn(model_dim))
        self.cat_proj = nn.Linear(768, model_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim)
        )
    
    def _encode_value(self, feature: Feature, value, bert_encoder) -> torch.Tensor:
        device = self.missing_embed.device
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return self.missing_embed
        if feature.dtype == 'continuous':
            val = float(value)
            if feature.value_range:
                vmin, vmax = feature.value_range
                val = (val - vmin) / (vmax - vmin + 1e-8)
            val_tensor = torch.tensor(val, device=device, dtype=torch.float32)
            fourier_feat = torch.cos(2 * math.pi * self.fourier_freqs * val_tensor)
            return self.fourier_proj(fourier_feat)
        else:
            cat_emb = bert_encoder(str(value))
            return self.cat_proj(cat_emb)
    
    def forward(self, feature: Feature, value, phi_feature: torch.Tensor, bert_encoder) -> torch.Tensor:
        nu_value = self._encode_value(feature, value, bert_encoder)
        combined = torch.cat([phi_feature, nu_value], dim=-1)
        atom_emb = self.atom_mlp(combined)
        return atom_emb

class UniversalInferenceModule(nn.Module):
    """Universal Inference Architecture with attention-based aggregation"""
    def __init__(self, model_dim=768, num_heads=8):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        
        # Type embeddings for tagging (keep from original)
        self.type_embeddings = nn.ParameterDict({
            'query': nn.Parameter(torch.randn(model_dim)),
            'target': nn.Parameter(torch.randn(model_dim)),
            'shot': nn.Parameter(torch.randn(model_dim)),
            'context': nn.Parameter(torch.randn(model_dim))
        })
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Tanh(),
            nn.Linear(model_dim, 1)
        )
        self.output_norm = nn.LayerNorm(model_dim)
    
    def forward(self, query_atoms, target_atoms, support_atoms, context_tokens):
        tagged_query = query_atoms + self.type_embeddings['query']
        tagged_target = target_atoms + self.type_embeddings['target']
        tagged_support = support_atoms + self.type_embeddings['shot']
        tagged_context = context_tokens + self.type_embeddings['context']
        all_tokens = torch.cat([tagged_query, tagged_target, tagged_support, tagged_context], dim=1)
        
        if all_tokens.size(1) == 0:
            return torch.zeros(target_atoms.size(0), target_atoms.size(1), self.model_dim, device=target_atoms.device)
        
        attn_weights = self.attention(all_tokens)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_features = all_tokens * attn_weights
        aggregated = weighted_features.sum(dim=1, keepdim=True)
        
        # Residual with mean pooling
        mean_pooled = all_tokens.mean(dim=1, keepdim=True)
        processed = aggregated + mean_pooled
        processed = self.output_norm(processed)
        
        # Broadcast to target size
        num_target = target_atoms.size(1)
        target_reprs = processed.expand(-1, num_target, -1)
        return target_reprs

class PredictionHeads(nn.Module):
    """Prediction heads for classification and regression"""
    def __init__(self, model_dim=768, bert_encoder=None, num_mixtures=10):
        super().__init__()
        self.model_dim = model_dim
        self.bert_encoder = bert_encoder
        self.num_mixtures = num_mixtures
        
        # Regression head - Mixture of Gaussians with multiple components
        self.reg_head_shared = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        # Mixture weights (logits)
        self.mixture_weights = nn.Linear(model_dim, self.num_mixtures)
        # Means for each mixture component
        self.mixture_means = nn.Linear(model_dim, self.num_mixtures)
        # Log variances for each mixture component (log for numerical stability)
        self.mixture_logvars = nn.Linear(model_dim, self.num_mixtures)
        
        # Enhanced classification head with MLP for better expressiveness
        self.classification_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(model_dim, model_dim)
        )
        # Project BERT embeddings (768) to model_dim if different
        self.category_proj = nn.Linear(768, model_dim)
    
    def predict_regression(self, repr, value, value_range=None):
        """
        Regression prediction using Mixture of Gaussians.
        Outputs are constrained to [0, 1] range using sigmoid.
        """
        device = repr.device
        raw_value = float(value)
        
        # Normalize target value to [0, 1] using column's min and max
        if value_range is not None:
            min_val, max_val = value_range
            # Avoid division by zero
            if abs(max_val - min_val) < 1e-8:
                normalized_target = 0.5
            else:
                # Min-max normalization: (value - min) / (max - min)
                normalized_target = (raw_value - min_val) / (max_val - min_val)
        else:
            # Fallback: simple scaling to [0, 1]
            normalized_target = raw_value / 100.0
        
        # Convert to tensor
        target = torch.tensor(normalized_target, device=device, dtype=torch.float32)
        
        # Get mixture components
        reg_features = self.reg_head_shared(repr)  # [model_dim]
        mixture_weights = F.softmax(self.mixture_weights(reg_features), dim=-1)  # [num_mixtures]
        mixture_means = self.mixture_means(reg_features)  # [num_mixtures]
        mixture_logvars = self.mixture_logvars(reg_features)  # [num_mixtures]
        
        # Apply sigmoid to means to constrain to [0, 1]
        mixture_means = torch.sigmoid(mixture_means)
        
        # Clamp logvars to prevent numerical instability
        mixture_logvars = torch.clamp(mixture_logvars, min=-10.0, max=10.0)
        mixture_vars = torch.exp(mixture_logvars)  # [num_mixtures]
        
        # Compute negative log-likelihood for mixture of Gaussians
        diff = target - mixture_means  # [num_mixtures]
        
        # Clamp diff to prevent extreme values
        diff = torch.clamp(diff, min=-100.0, max=100.0)
        
        # Gaussian likelihood for each component with numerical stability
        log_2pi = math.log(2 * math.pi)
        log_probs = -0.5 * (diff**2 / (mixture_vars + 1e-8) + mixture_logvars + log_2pi)  # [num_mixtures]
        
        # Clamp log_probs to prevent extreme values
        log_probs = torch.clamp(log_probs, min=-50.0, max=50.0)
        
        # Weighted mixture likelihood
        weighted_log_probs = log_probs + torch.log(mixture_weights + 1e-8)  # [num_mixtures]
        
        # Log-sum-exp for numerical stability
        loss = -torch.logsumexp(weighted_log_probs, dim=-1)  # scalar
        
        # Clamp regression loss to prevent negative values
        loss = torch.clamp(loss, min=0.0, max=100.0)
        
        # Prediction: weighted average of mixture means
        pred_normalized = torch.sum(mixture_weights * mixture_means, dim=-1)  # scalar, already in [0, 1]
        
        # Denormalize prediction back to original scale
        pred_value = pred_normalized.item()
        if value_range is not None:
            min_val, max_val = value_range
            # Inverse min-max: value = normalized * (max - min) + min
            denormalized_pred = pred_value * (max_val - min_val) + min_val
        else:
            denormalized_pred = pred_value * 100.0
        
        return loss, denormalized_pred
    
    def predict_classification(self, repr, categories, target_idx):
        """Enhanced classification with MLP head + dot-product"""
        repr_enhanced = self.classification_head(repr)
        
        if categories and self.bert_encoder:
            # BERT-based category representation
            cat_vectors = []
            for cat in categories:
                # Encode category using BERT
                cat_bert_emb = self.bert_encoder(str(cat))  # Returns 768-dim BERT embedding
                cat_proj = self.category_proj(cat_bert_emb)  # Project to model_dim
                cat_vectors.append(cat_proj)
            
            cat_tensor = torch.stack(cat_vectors)  # [num_categories, model_dim]
            
            # Cosine similarity for better generalization
            # Normalize both representation and category vectors
            repr_norm = F.normalize(repr_enhanced, dim=-1)
            cat_norm = F.normalize(cat_tensor, dim=-1)
            
            # Scaled cosine similarity
            # CRITICAL FIX: Lower temperature to reduce overconfidence  
            temperature = 3.0 if not self.training else 20.0  # Use 3.0 at inference, 20.0 during training
            logits = torch.matmul(repr_norm, cat_norm.T) * temperature
            
            # DEBUG: Log logits distribution
            if not self.training:
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Logits: {logits.cpu().numpy()}, Predicted: {torch.argmax(logits).item()}")
            
            target = torch.tensor([target_idx], device=logits.device, dtype=torch.long)
            # Reduce label smoothing - it hurts on imbalanced classes
            loss = F.cross_entropy(logits.unsqueeze(0), target, label_smoothing=0.05)
            prediction = torch.argmax(logits).item()
            return loss, prediction
        else:
            return torch.tensor(0.0, device=repr.device), 0

# Enhanced ASPIRE Model with Masking and Multi-Target
class ASPIREEnhanced(nn.Module):
    """Full ASPIRE with Enhanced Features: masking and arbitrary targets"""
    def __init__(self, model_dim=768, num_heads=8, num_inds=32, mask_prob=0.15, max_targets=3):
        super().__init__()
        self.model_dim = model_dim
        self.mask_prob = mask_prob  # ENHANCEMENT: Random masking probability
        self.max_targets = max_targets  # ENHANCEMENT: Max targets to predict
        
        # Core ASPIRE components
        self.semantic_grounding = SemanticFeatureGrounding(model_dim)
        self.atom_processing = AtomProcessing(model_dim)
        self.instance_transformer = SetTransformer(dim=model_dim, num_heads=num_heads, num_inds=num_inds, num_layers=2)
        self.universal_inference = UniversalInferenceModule(model_dim, num_heads)
        # Pass BERT encoder to prediction heads for classification
        self.prediction_heads = PredictionHeads(model_dim, bert_encoder=self.semantic_grounding._encode_text)
        
        # Context encoder
        self.context_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.context_bert = AutoModel.from_pretrained('bert-base-uncased')
        self.context_bert.eval()
        bert_dim = self.context_bert.config.hidden_size
        self.context_proj = nn.Linear(bert_dim, model_dim)
        
        # ENHANCEMENT: Mask token for masked features
        self.mask_token = nn.Parameter(torch.randn(model_dim))
    
    def _select_targets(self, num_features: int, existing_targets: List[int] = None) -> List[int]:
        """ENHANCEMENT: Select arbitrary targets from available features"""
        available_indices = list(range(num_features))
        if existing_targets:
            selected_targets = existing_targets[:]
        else:
            selected_targets = []
        remaining_indices = [i for i in available_indices if i not in selected_targets]
        if remaining_indices:
            num_additional = min(
                random.randint(1, self.max_targets) - len(selected_targets),
                len(remaining_indices)
            )
            if num_additional > 0:
                additional_targets = random.sample(remaining_indices, num_additional)
                selected_targets.extend(additional_targets)
        return selected_targets if selected_targets else [random.randint(0, num_features - 1)]
    
    def _mask_atoms(self, atoms: torch.Tensor, target_indices: List[int]) -> torch.Tensor:
        """ENHANCEMENT: Randomly mask atom features during training"""
        if not self.training:
            return atoms
        masked_atoms = atoms.clone()
        num_atoms = atoms.size(0)
        for i in range(num_atoms):
            if i not in target_indices:
                if random.random() < self.mask_prob:
                    masked_atoms[i] = self.mask_token
        return masked_atoms
    
    def process_instance(self, features: List[Feature], values: List, target_indices: List[int] = None) -> torch.Tensor:
        """Process instance as set of atoms with optional masking"""
        phi_features = [self.semantic_grounding(f) for f in features]
        atoms = []
        for i, (feature, value) in enumerate(zip(features, values)):
            atom = self.atom_processing(feature, value, phi_features[i], self.semantic_grounding._encode_text)
            atoms.append(atom)
        atom_tensor = torch.stack(atoms, dim=0)
        if self.training and target_indices is not None:
            atom_tensor = self._mask_atoms(atom_tensor, target_indices)
        atom_batch = atom_tensor.unsqueeze(0)
        instance_atoms = self.instance_transformer(atom_batch)
        instance_atoms = instance_atoms.squeeze(0)
        return instance_atoms
    
    def encode_context(self, context_text: str) -> torch.Tensor:
        """Encode dataset description"""
        device = next(self.parameters()).device
        with torch.no_grad():
            inputs = self.context_tokenizer(context_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = self.context_bert(**inputs)
            tokens = outputs.last_hidden_state
        tokens = self.context_proj(tokens)
        return tokens
    
    def forward(self, batch: List[Example]):
        """Forward pass with arbitrary conditioning and masking"""
        device = next(self.parameters()).device
        total_loss = 0.0
        num_targets_processed = 0
        
        for example in batch:
            # ENHANCEMENT: Select arbitrary targets during training
            if self.training:
                target_indices = self._select_targets(len(example.features), example.target_indices)
            else:
                target_indices = example.target_indices
            
            # Process instance with masking
            instance_atoms = self.process_instance(example.features, example.values, target_indices)
            
            # Split observed/target
            observed_indices = [i for i in range(len(example.features)) if i not in target_indices]
            query_atoms = instance_atoms[observed_indices]
            target_atoms = instance_atoms[target_indices]
            
            # Process support set
            support_atoms_list = []
            if example.support_examples:
                for support_ex in example.support_examples:
                    support_inst = self.process_instance(support_ex.features, support_ex.values)
                    support_atoms_list.append(support_inst)
            if support_atoms_list:
                support_atoms = torch.cat(support_atoms_list, dim=0).unsqueeze(0)
            else:
                support_atoms = torch.zeros(1, 0, self.model_dim, device=device)
            
            # Encode context
            context_tokens = self.encode_context(example.dataset_context)
            
            # Add batch dimension
            query_batch = query_atoms.unsqueeze(0)
            target_batch = target_atoms.unsqueeze(0)
            
            # Universal inference
            target_reprs = self.universal_inference(query_batch, target_batch, support_atoms, context_tokens)
            
            # Predictions
            for i, target_idx in enumerate(target_indices):
                feature = example.features[target_idx]
                value = example.values[target_idx]
                repr = target_reprs[0, i]
                
                if feature.dtype == 'continuous':
                    loss, _ = self.prediction_heads.predict_regression(repr, value, feature.value_range)
                else:
                    # Use BERT encoding + dot-product for classification
                    try:
                        target_cat_idx = feature.choices.index(str(value))
                    except ValueError:
                        target_cat_idx = 0
                    loss, _ = self.prediction_heads.predict_classification(repr, feature.choices, target_cat_idx)
                
                total_loss += loss
                num_targets_processed += 1
        
        return total_loss / max(num_targets_processed, 1)
    
    def predict(self, example: Example):
        """Make predictions for target features"""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            instance_atoms = self.process_instance(example.features, example.values)
            observed_indices = [i for i in range(len(example.features)) if i not in example.target_indices]
            query_atoms = instance_atoms[observed_indices].unsqueeze(0)
            target_atoms = instance_atoms[example.target_indices].unsqueeze(0)
            
            support_atoms_list = []
            if example.support_examples:
                for support_ex in example.support_examples:
                    support_inst = self.process_instance(support_ex.features, support_ex.values)
                    support_atoms_list.append(support_inst)
            if support_atoms_list:
                support_atoms = torch.cat(support_atoms_list, dim=0).unsqueeze(0)
            else:
                support_atoms = torch.zeros(1, 0, self.model_dim, device=device)
            
            context_tokens = self.encode_context(example.dataset_context)
            target_reprs = self.universal_inference(query_atoms, target_atoms, support_atoms, context_tokens)
            
            predictions = []
            for i, target_idx in enumerate(example.target_indices):
                feature = example.features[target_idx]
                repr = target_reprs[0, i]
                if feature.dtype == 'continuous':
                    _, pred = self.prediction_heads.predict_regression(repr, 0.0, feature.value_range)
                    predictions.append(float(pred) if hasattr(pred, 'item') else pred)
                else:
                    # Use BERT encoding + dot-product for classification
                    _, pred_idx = self.prediction_heads.predict_classification(repr, feature.choices, 0)
                    predictions.append(feature.choices[pred_idx])
        return predictions

# Data Loading
def load_single_dataset_batches(data_dir, max_datasets=15, multi_target=True, use_support=False, num_support=5):
    """Load data with multi-target support and metadata from JSON files"""
    logger.info(f"Loading datasets with metadata (multi_target={multi_target}, use_support={use_support}, num_support={num_support})")
    csv_files = glob.glob(f"{data_dir}/**/*.csv", recursive=True)
    if max_datasets:
        csv_files = csv_files[:max_datasets]
    dataset_examples = {}
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, low_memory=False)
            if len(df) < 15 or len(df.columns) < 3:
                continue
            
            # Try to load corresponding metadata JSON file
            csv_filename = os.path.basename(file_path)
            csv_dir = os.path.dirname(file_path)
            
            # Handle synthetic_ prefix in CSV filename
            if csv_filename.startswith('synthetic_'):
                base_name = csv_filename.replace('synthetic_', '').replace('.csv', '')
                metadata_path = os.path.join(csv_dir, f'metadata_{base_name}.json')
            else:
                base_name = csv_filename.replace('.csv', '')
                metadata_path = os.path.join(csv_dir, f'{base_name}.json')
            
            # Load metadata if available
            metadata = None
            dataset_description = base_name  # Default to filename
            feature_metadata = {}
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    dataset_description = metadata.get('description', base_name)
                    # Create feature lookup by name
                    for feat_meta in metadata.get('features', []):
                        feature_metadata[feat_meta['name']] = feat_meta
                    logger.info(f"Loaded metadata from {os.path.basename(metadata_path)}")
                except Exception as e:
                    logger.warning(f"Could not load metadata from {metadata_path}: {e}")
            else:
                logger.warning(f"No metadata found at {metadata_path}, using default descriptions")
            
            dataset_name = base_name
            examples = []
            sample_size = min(150, len(df))
            df_sample = df.sample(n=sample_size, random_state=42)
            
            features = []
            for col in df_sample.columns:
                col_data = df_sample[col].dropna()
                if len(col_data) < 10:
                    continue
                
                # Get metadata for this feature if available
                feat_meta = feature_metadata.get(col, {})
                feat_description = feat_meta.get('description', f"Feature {col}")
                feat_type = feat_meta.get('type', None)
                
                # Determine feature type from data or metadata
                if feat_type == 'discrete' or col_data.dtype in ['object', 'category']:
                    unique_vals = col_data.unique()
                    if 2 <= len(unique_vals) <= 30:
                        # Use categories from metadata if available, otherwise from data
                        choices = feat_meta.get('categories', [str(v) for v in unique_vals])
                        if not choices:  # Fallback if metadata categories is empty
                            choices = [str(v) for v in unique_vals]
                        features.append(Feature(col, feat_description, "categorical", choices))
                elif feat_type == 'continuous':
                    # Use range from metadata if available
                    if 'range' in feat_meta and feat_meta['range'] is not None:
                        min_val, max_val = feat_meta['range']
                    else:
                        # Fallback to data range
                        numeric_vals = pd.to_numeric(col_data, errors='coerce').dropna()
                        if len(numeric_vals) < len(col_data) * 0.8:
                            continue
                        min_val = float(numeric_vals.min())
                        max_val = float(numeric_vals.max())
                    features.append(Feature(col, feat_description, "continuous", value_range=(min_val, max_val)))
                else:
                    # Auto-detect if no metadata
                    try:
                        numeric_vals = pd.to_numeric(col_data, errors='coerce').dropna()
                        if len(numeric_vals) >= len(col_data) * 0.8:
                            min_val = float(numeric_vals.min())
                            max_val = float(numeric_vals.max())
                            features.append(Feature(col, feat_description, "continuous", value_range=(min_val, max_val)))
                    except:
                        continue
            
            if len(features) < 3:
                continue
            
            for _, row in df_sample.iterrows():
                values = []
                valid_indices = []
                for i, feature in enumerate(features):
                    val = row[feature.name]
                    if pd.isna(val):
                        continue
                    if feature.dtype == 'continuous':
                        try:
                            val = float(val)
                            if not math.isfinite(val) or abs(val) > 1e8:
                                continue
                        except:
                            continue
                    else:
                        val = str(val)
                        if val not in feature.choices:
                            continue
                    values.append(val)
                    valid_indices.append(i)
                
                if len(valid_indices) < 3:
                    continue
                
                if multi_target and len(valid_indices) >= 2:
                    num_targets = min(random.randint(1, 3), len(valid_indices))
                    selected_targets = random.sample(valid_indices, num_targets)
                    target_indices = [valid_indices.index(t) for t in selected_targets]
                else:
                    target_idx = random.choice(valid_indices)
                    target_indices = [valid_indices.index(target_idx)]
                
                filtered_features = [features[i] for i in valid_indices]
                example = Example(
                    features=filtered_features,
                    values=values,
                    target_indices=target_indices,
                    dataset_context=dataset_description, 
                    support_examples=None  
                )
                examples.append(example)
            
            if examples:
                # Add few-shot support examples to each training example
                if use_support:
                    for i, ex in enumerate(examples):
                        # Sample support from other examples in same dataset (exclude current)
                        other_examples = [e for j, e in enumerate(examples) if j != i]
                        if len(other_examples) >= num_support:
                            support_samples = random.sample(other_examples, num_support)
                            # Create support examples with full feature-value pairs
                            ex.support_examples = [
                                Example(
                                    features=s.features,
                                    values=s.values,
                                    target_indices=[],  # Support shows all features
                                    dataset_context=s.dataset_context,
                                    support_examples=None
                                ) for s in support_samples
                            ]
                    logger.info(f"Loaded {len(examples)} examples from {dataset_name} with {num_support}-shot support")
                else:
                    logger.info(f"Loaded {len(examples)} examples from {dataset_name} (zero-shot)")
                
                dataset_examples[dataset_name] = examples
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue
    
    logger.info(f"Loaded {len(dataset_examples)} datasets")
    return dataset_examples

# Training
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/playpen-nvme/scribble/shbhat/eddy_data/synthetic_data_1')
    parser.add_argument('--max_datasets', type=int, default=15)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_inds', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mask_prob', type=float, default=0.15)
    parser.add_argument('--max_targets', type=int, default=1)
    parser.add_argument('--multi_target', action='store_true')
    parser.add_argument('--use_support', action='store_true', help='Enable few-shot support examples')
    parser.add_argument('--num_support', type=int, default=5, help='Number of support examples per instance')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"ASPIRE Training")
    logger.info(f"Data: {args.data_dir}")
    logger.info(f"Device: {device}")
    
    dataset_examples = load_single_dataset_batches(
        data_dir=args.data_dir,
        max_datasets=args.max_datasets, 
        multi_target=args.multi_target,
        use_support=args.use_support,
        num_support=args.num_support
    )
    
    if not dataset_examples:
        logger.error("No datasets loaded!")
        return
    
    model = ASPIREEnhanced(
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_inds=args.num_inds,
        mask_prob=args.mask_prob,
        max_targets=args.max_targets
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Mask prob: {args.mask_prob}, Max targets: {args.max_targets}")
    
    dataset_names = list(dataset_examples.keys())
    best_eval_loss = float('inf')
    best_model_state = None
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        random.shuffle(dataset_names)
        
        for dataset_name in dataset_names:
            examples = dataset_examples[dataset_name]
            random.shuffle(examples)
            
            for i in range(0, len(examples), args.batch_size):
                batch = examples[i:i+args.batch_size]
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        
        # Evaluate
        model.eval()
        eval_loss = 0.0
        eval_batches = 0
        val_reg_preds, val_reg_targets = [], []
        val_cls_preds, val_cls_targets = [], []
        
        val_reg_preds_norm, val_reg_targets_norm = [], []
        
        with torch.no_grad():
            for dataset_name in dataset_names[:4]:
                examples = dataset_examples[dataset_name][:50]
                for i in range(0, len(examples), args.batch_size):
                    batch = examples[i:i+args.batch_size]
                    loss = model(batch)
                    eval_loss += loss.item()
                    eval_batches += 1
                
                for example in examples:
                    predictions = model.predict(example)
                    for j, target_idx in enumerate(example.target_indices):
                        if j < len(predictions) and target_idx < len(example.features):
                            feature = example.features[target_idx]
                            true_val = example.values[target_idx]
                            pred_val = predictions[j]
                            if feature.dtype == 'continuous' and pred_val is not None:
                                val_reg_preds.append(float(pred_val))
                                val_reg_targets.append(float(true_val))
                                
                                # Also track normalized values for better metric insight
                                if feature.value_range is not None:
                                    min_val, max_val = feature.value_range
                                    if abs(max_val - min_val) > 1e-8:
                                        norm_pred = (float(pred_val) - min_val) / (max_val - min_val)
                                        norm_true = (float(true_val) - min_val) / (max_val - min_val)
                                        val_reg_preds_norm.append(norm_pred)
                                        val_reg_targets_norm.append(norm_true)
                            elif feature.dtype == 'categorical' and pred_val is not None:
                                val_cls_preds.append(str(pred_val))
                                val_cls_targets.append(str(true_val))
        
        eval_avg = eval_loss / max(eval_batches, 1)
        
        # Calculate metrics
        mse_val = None
        mse_val_norm = None
        f1_val = None
        acc_val = None
        
        if val_reg_preds:
            mse_val = mean_squared_error(val_reg_targets, val_reg_preds)
        if val_reg_preds_norm:
            mse_val_norm = mean_squared_error(val_reg_targets_norm, val_reg_preds_norm)
        if val_cls_preds:
            acc_val = accuracy_score(val_cls_targets, val_cls_preds)
            f1_val = f1_score(val_cls_targets, val_cls_preds, average='weighted', zero_division=0)
            # Diagnostic: check if model is predicting same class
            unique_preds = len(set(val_cls_preds))
            unique_targets = len(set(val_cls_targets))
            if epoch == 0 or (epoch + 1) % 5 == 0:
                pass
                # logger.info(f"Classification Debug] Unique predictions: {unique_preds}/{unique_targets} classes, Acc: {acc_val:.3f}")
        
        best_marker = ""
        if eval_avg < best_eval_loss:
            best_eval_loss = eval_avg
            best_model_state = model.state_dict().copy()
            best_marker = "good"
            
            # Save best model
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model_state,
                'model_dim': args.model_dim,
                'num_heads': args.num_heads,
                'num_inds': args.num_inds,
                'mask_prob': args.mask_prob,
                'max_targets': args.max_targets,
                'best_eval_loss': best_eval_loss,
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_path)
            logger.info(f"Saved best model to {best_path}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'model_dim': args.model_dim,
                'num_heads': args.num_heads,
                'num_inds': args.num_inds,
                'mask_prob': args.mask_prob,
                'max_targets': args.max_targets,
                'val_loss': eval_avg,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Epoch summary
        metrics = [f"Epoch {epoch+1}/{args.num_epochs}", f"TrainLoss={avg_loss:.4f}", f"ValLoss={eval_avg:.4f}"]
        if mse_val_norm is not None:
            metrics.append(f"MSE={mse_val_norm:.4f}")
        if f1_val is not None:
            metrics.append(f"F1={f1_val:.4f}")
        logger.info(" | ".join(metrics) + f" {best_marker}")
    
    logger.info("Training completed!")
    
    # Final eval
    model.eval()
    all_reg_preds, all_reg_targets = [], []
    all_reg_preds_norm, all_reg_targets_norm = [], []
    all_cls_preds, all_cls_targets = [], []
    
    with torch.no_grad():
        for dataset_name in dataset_names:
            examples = dataset_examples[dataset_name][:25]
            for example in examples:
                predictions = model.predict(example)
                for i, target_idx in enumerate(example.target_indices):
                    feature = example.features[target_idx]
                    target_value = example.values[target_idx]
                    pred_value = predictions[i] if i < len(predictions) else None
                    if pred_value is None:
                        continue
                    if feature.dtype == 'continuous':
                        all_reg_preds.append(pred_value)
                        all_reg_targets.append(float(target_value))
                        
                        # Also track normalized
                        if feature.value_range is not None:
                            min_val, max_val = feature.value_range
                            if abs(max_val - min_val) > 1e-8:
                                norm_pred = (pred_value - min_val) / (max_val - min_val)
                                norm_true = (float(target_value) - min_val) / (max_val - min_val)
                                all_reg_preds_norm.append(norm_pred)
                                all_reg_targets_norm.append(norm_true)
                    else:
                        all_cls_preds.append(str(pred_value))
                        all_cls_targets.append(str(target_value))
    
    logger.info(f"\nFinal Evaluation:")
    if all_reg_preds_norm:
        rmse_norm = np.sqrt(mean_squared_error(all_reg_targets_norm, all_reg_preds_norm))
        logger.info(f"   Regression RMSE: {rmse_norm:.4f}")
    if all_cls_preds:
        accuracy = accuracy_score(all_cls_targets, all_cls_preds)
        f1 = f1_score(all_cls_targets, all_cls_preds, average='weighted', zero_division=0)
        logger.info(f"   Classification Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        logger.info(f"   Classification F1: {f1:.4f} ({f1*100:.1f}%)")
    
    # Summary of saved models
    logger.info(f"\nSaved Models:")
    logger.info(f"Best model: {os.path.join(args.save_dir, 'best_model.pt')}")
    logger.info(f"Best validation loss: {best_eval_loss:.4f}")
    logger.info(f"Checkpoints saved every 5 epochs in: {args.save_dir}")

if __name__ == "__main__":
    train()
