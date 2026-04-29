import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Any, Union, Callable
from torch import Tensor
import math
import numpy as np

DEFAULT_SUCCESS_CADENCE = (2, 3)

class RPPTransformer(nn.Module):

    def __init__(self,cfg):
        super().__init__()

        # initial
        seq_max = cfg['seq_max']
        d_model = cfg['d_model']
        nhead = cfg['nhead']
        num_layers = cfg['num_layers']
        dim_feedforward = cfg['dim_feedforward']
        dropout = cfg['dropout']
        self.rpp_feature_dim_dict = cfg['rpp_feature_dict']
        self.rpp_feature_selected = cfg['rpp_feature_selected']
        self.d_out = np.sum([self.rpp_feature_dim_dict[f] for f in self.rpp_feature_selected])
        self.seq_max = cfg['seq_max']

        self.cfg = cfg
        self.feature_name_to_idx = {name: idx for idx, name in enumerate(self.rpp_feature_selected)}
        self._value_vector_cache = {}
        self._cached_hidden = None
        
        start_tokens = []
        for feat_name in self.rpp_feature_selected:
            vocab = int(self.rpp_feature_dim_dict[feat_name])
            start_tokens.append(max(vocab - 1, 0))
        self.register_buffer('start_token_vector', torch.tensor(start_tokens, dtype=torch.long))
        feature_loss_cfg = cfg.get('feature_loss_weights', {}) or {}
        self.feature_loss_weights = {
            name: float(feature_loss_cfg.get(name, 1.0)) for name in self.rpp_feature_selected
        }
        
        # Embedding Module
        self.RPP_embeding = RppEmbedding(cfg)
        self.PE_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=seq_max,
            
        )  # positional encoding
        
        self.EncoderLayer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,batch_first=True)
        self.Encoder = nn.TransformerEncoder(self.EncoderLayer,num_layers=num_layers)
        
        self.PositionTransformerDecoderLayer = nn.TransformerDecoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout,batch_first=True)
        self.PositionTransformerDecoder = nn.TransformerDecoder(self.PositionTransformerDecoderLayer,num_layers=num_layers)
        
        # Output-Linear Module
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(in_features=2*d_model,out_features=d_model)
        self.linear2 = nn.Linear(in_features=d_model,out_features=d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.memory_dropout = nn.Dropout(dropout)
        self.input_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.post_decoder_norm = nn.LayerNorm(d_model)
        self.output_norm = nn.LayerNorm(d_model)
        self.feature_dims = {name: int(self.rpp_feature_dim_dict[name]) for name in self.rpp_feature_selected}
        self.time_features = [feat for feat in ['global_pos', 'bar', 'position', 'duration'] if feat in self.feature_dims]
        cascade_dim = int(cfg.get('cascade_embed_dim', d_model))
        self.cascade_embed_dim = cascade_dim
        
        # Dual-Stream Injection Embeddings
        if 'global_pos' in self.feature_dims:
            self.future_pos_embedding = nn.Embedding(self.feature_dims['global_pos'], cascade_dim, padding_idx=0)
            self.delta_embedding = nn.Embedding(512, cascade_dim) 
        
        self.feature_heads = nn.ModuleDict()
        for feat in self.rpp_feature_selected:
            in_dim = d_model
            if feat == 'global_pos':
                # Stream A: Skeleton (Global Pos) relies only on history (hidden state)
                pass 
            else:
                # Stream B: Detail Features get Future Pos + Delta injection
                if 'global_pos' in self.feature_dims:
                    in_dim += (cascade_dim * 2) # Future Pos Emb + Delta Emb

            self.feature_heads[feat] = nn.Linear(in_dim, self.feature_dims[feat])
            
       
        label_smoothing = float(cfg.get('label_smoothing', 0.0))
        self.loss_fuc = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=label_smoothing)
        self.label_smoothing = label_smoothing
        self.label_smoothing_sigma = float(cfg.get('label_smoothing_sigma', 1.0))
       
        self.distance_smoothed_features = set()



    def forward(self, tgt, tgt_gt=None, tgt_key_mask=None, future_pos_override=None, use_teacher_pos=True):
        seq_lenth = tgt.shape[1]
        
        if tgt.shape[1] < self.seq_max:
            pad_lenth = self.seq_max - tgt.shape[1]
            pad = (0, 0, 0, pad_lenth, 0, 0)
            tgt = nn.functional.pad(tgt, pad, "constant", 0) if pad_lenth > 0 else tgt
        
        key_mask = None
        if tgt_key_mask is not None:
            key_mask = self._align_mask(tgt_key_mask, tgt.shape[1], tgt.device)
        bool_mask = None if key_mask is None else key_mask > 0.5

        raw_feats = tgt.clone()
        teacher_inputs = self._build_autoregressive_inputs(tgt)
        tgt = self.RPP_embeding(teacher_inputs)
        
        tgt = self.embedding_dropout(tgt)
        tgt = self.input_norm(tgt)
        
        tgt_pe = self.PE_encoding(tgt, key_padding_mask=key_mask, raw_feats=raw_feats)

        tgt_mask = self._get_square_subsequent_mask(tgt.shape[1], device=tgt.device)
        tgt_padding_mask = None if key_mask is None else self._get_key_padding_mask(tgt_mask=key_mask)
        
        attr_memory = self.RPP_embeding(teacher_inputs)
        if bool_mask is not None:
            attr_memory = attr_memory * bool_mask.unsqueeze(-1).float()
        memory = attr_memory
        memory = self.memory_dropout(memory)
        
        memory = self.PE_encoding(memory, key_padding_mask=key_mask, raw_feats=None)
        memory = self.memory_norm(memory)
        
        memory = self.Encoder(memory, mask=tgt_mask)
        
       
        output = self.PositionTransformerDecoder(tgt=tgt_pe,
                                                 memory=memory,
                                                 tgt_mask=tgt_mask,
                                                 memory_mask=tgt_mask,
                                                 tgt_key_padding_mask=tgt_padding_mask)
        output = self.post_decoder_norm(output)

        # Skip connection with decoder input to stabilise training
        combined = torch.cat([output, tgt_pe], dim=-1)
        combined = self.dropout1(F.relu(self.linear1(combined)))
        hidden = self.dropout2(F.relu(self.linear2(combined)))                                # N X S X D
        hidden = self.output_norm(hidden)
        self._cached_hidden = hidden
        seq_len = hidden.shape[1]

        # Teacher tokens (current step) from ground truth if provided
        teacher_tokens = {}
        if tgt_gt is not None:
            for feat in self.time_features:
                idx = self.feature_name_to_idx.get(feat)
                if idx is None:
                    continue
                if idx < tgt_gt.shape[2]:
                    tokens = tgt_gt[:, :seq_len, idx].long()
                    if tokens.shape[1] < seq_len:
                        pad = tokens.new_zeros(tokens.shape[0], seq_len - tokens.shape[1])
                        tokens = torch.cat([tokens, pad], dim=1)
                    teacher_tokens[feat] = tokens

        def _shift_future(tok):
            if tok is None:
                return None
            future = tok.new_zeros(tok.shape)
            if tok.shape[1] > 1:
                future[:, :-1] = tok[:, 1:]
                future[:, -1] = tok[:, -1] + 16
            return future

        logits_map = {}

        def run_head(feature_name, extra_embs=None):
            if feature_name not in self.feature_heads:
                return
            head = self.feature_heads[feature_name]
            tensors = [hidden]
            if extra_embs:
                tensors.extend([emb for emb in extra_embs if emb is not None])
            head_input = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=-1)
            logits = head(head_input)
            logits_map[feature_name] = logits

        # Stream A: predict global_pos first (strictly causal via decoder mask)
        pos_logits = None
        if 'global_pos' in self.feature_heads:
            run_head('global_pos')
            pos_logits = logits_map.get('global_pos')

        # Build future position tokens for Stream B
        future_pos_tokens = None
        if future_pos_override is not None:
            future_pos_tokens = future_pos_override[:, :seq_len].long()
        elif use_teacher_pos and 'global_pos' in teacher_tokens:
            # Use Teacher Forcing for Delta/Position Context
            future_pos_tokens = _shift_future(teacher_tokens['global_pos'])
        elif pos_logits is not None:
             # Use Model Prediction (Autoregressive / Scheduled Sampling)
            cur_pred = torch.argmax(pos_logits, dim=-1)
            future_pos_tokens = _shift_future(cur_pred)

        future_pos_emb = None
        if future_pos_tokens is not None and hasattr(self, 'future_pos_embedding'):
            clipped = future_pos_tokens.clamp(min=0, max=self.future_pos_embedding.num_embeddings - 1)
            future_pos_emb = self.future_pos_embedding(clipped)

        delta_emb = None
        delta = None
        if hasattr(self, 'delta_embedding') and future_pos_tokens is not None:
            cur_pos_tokens = None
            idx_gp = self.feature_name_to_idx.get('global_pos')
            if idx_gp is not None:
                # Decide Current Position Source
                if raw_feats.shape[-1] > idx_gp and use_teacher_pos:
                     cur_pos_tokens = raw_feats[:, :seq_len, idx_gp].long()
                elif 'global_pos' in teacher_tokens and use_teacher_pos:
                     cur_pos_tokens = teacher_tokens['global_pos']
                elif pos_logits is not None:
                     
                     if raw_feats.shape[-1] > idx_gp:
                         cur_pos_tokens = raw_feats[:, :seq_len, idx_gp].long()

            if cur_pos_tokens is not None:
                delta = (future_pos_tokens - cur_pos_tokens).clamp(min=0, max=self.delta_embedding.num_embeddings - 1)
                delta_emb = self.delta_embedding(delta)

        # Stream B: run detail heads with injected future anchors
        for feat in self.rpp_feature_selected:
            if feat == 'global_pos':
                continue
            extras = []
            if future_pos_emb is not None:
                extras.append(future_pos_emb)
            if delta_emb is not None:
                extras.append(delta_emb)
            run_head(feat, extra_embs=extras)
            
            # Duration Force Non-Overlapping Mask
            if feat == 'duration' and delta is not None and 'duration' in logits_map:
                dur_logits = logits_map['duration'] 
                vocab_size = dur_logits.shape[-1]
                # mask indices k > delta
                k_indices = torch.arange(vocab_size, device=dur_logits.device).view(1, 1, -1)
                delta_expanded = delta.unsqueeze(-1)
                mask = k_indices > delta_expanded
                
                
                if tgt_gt is not None:
                    idx_dur = self.feature_name_to_idx.get('duration')
                    if idx_dur is not None and idx_dur < tgt_gt.shape[2]:
                        gt_dur = tgt_gt[:, :seq_len, idx_dur].long()
                        if gt_dur.shape[1] < seq_len:
                            pad = gt_dur.new_zeros(gt_dur.shape[0], seq_len - gt_dur.shape[1])
                            gt_dur = torch.cat([gt_dur, pad], dim=1)
                        gt_dur = gt_dur.clamp(0, vocab_size - 1)
                        mask.scatter_(2, gt_dur.unsqueeze(-1), False)

                logits_map['duration'] = dur_logits.masked_fill(mask, -1e9)

        ordered_logits = [logits_map[feat] for feat in self.rpp_feature_selected if feat in logits_map]
        if not ordered_logits:
            raise RuntimeError("No feature logits were produced; check feature configuration")
        output = torch.cat(ordered_logits, dim=-1)

        return output[:, :seq_lenth, :]


    def predict_transform(self,X):
        # Input:  X [batch,1,d_model]
        # Output: X [batch,d_feat]
        feature_size = [self.cfg['rpp_feature_dict'][f] for f in self.cfg['rpp_feature_selected']]
        div_index =[0] + [sum(feature_size[:i+1]) for i in range(len(feature_size))]
        Xs = [X[...,i:j] for i,j in zip(div_index[:-1],div_index[1:])]
        Xs = [torch.squeeze(x,dim=1) for x in Xs]
        Xs = [x/self.cfg['temperature'] for x in Xs]
        Xs = [F.softmax(x,dim=1) for x in Xs]
        Xs = [torch.multinomial(x,num_samples=1) for x in Xs]
        Xs = torch.cat(Xs,dim=1)   # [batch,token_seq] [8,4]
        return Xs

    def loss(self,predict,gt,mask=None,raw_feats=None,return_components=False):
        # X [batch,token_seq,sum(dim_feature)]   --------    Y [batch,token_seq,note_feature]
        # X [10,512,453]                         --------    Y [10,512,5]

        # Split Feature
        feature_size = [self.cfg['rpp_feature_dict'][f] for f in self.cfg['rpp_feature_selected']]
        assert len(feature_size) == gt.shape[2], 'in model.loss : Dim of Featureselected != Dim of GroundTruth[2]'
        assert sum(feature_size) == predict.shape[2], 'in model.Loss : Dim(predict[2]) != Dim(features)'
        div_index =[0] + [sum(feature_size[:i+1]) for i in range(len(feature_size))]
        predicts = [predict[...,i:j] for i,j in zip(div_index[:-1],div_index[1:])]
        gts = [gt[...,i] for i in range(gt.shape[2])]
        mask_tensor = None
        if mask is not None:
            mask_tensor = self._align_mask(mask, predict.shape[1], predict.device) > 0.5

        # Gather loss
        weighted_losses = []
        weight_total = 0.0
        feature_components = {}
        mse_losses = []
        for feat_name, pre, gt_feat in zip(self.rpp_feature_selected, predicts, gts):
            weight = float(self.feature_loss_weights.get(feat_name, 1.0))
            if feat_name in self.distance_smoothed_features and self.label_smoothing_sigma > 0:
                raw_loss = self._gaussian_label_loss(pre, gt_feat)
            else:
                raw_loss = self._loss_each_featrue(pre=pre, gt=gt_feat)
            feature_components[feat_name] = raw_loss
            weighted_losses.append(raw_loss * weight)
            weight_total += weight
        ce_loss = sum(weighted_losses) / max(weight_total, 1e-6)

    
        reg_loss = self._structural_regularizers(predicts, mask, raw_feats=raw_feats, hidden=self._cached_hidden)
        self._cached_hidden = None
        total_loss = ce_loss + reg_loss
        if return_components:
            comp_out = {name: loss_tensor.detach().item() for name, loss_tensor in feature_components.items()}
            return total_loss, comp_out
        return total_loss
    
    def loss_with_indices(self,predict,gt,indices,mask=None,raw_feats=None):
        # preidct [batch,token_seq,sum(dim_feature)]   --------    gt [batch,token_seq,note_feature]
        # X [10,512,453]                         --------    Y [10,512,5]
        
        # transform
        if torch.is_tensor(indices):
            select_idx = indices.tolist()
        else:
            select_idx = indices
        predict = predict[:,select_idx,:]
        gt = gt[:,select_idx,:]
        mask_sel = None
        if mask is not None:
            mask_sel = mask[:, select_idx]
        raw_sel = None
        if raw_feats is not None:
            raw_sel = raw_feats[:, select_idx, :]

        return self.loss(predict, gt, mask=mask_sel, raw_feats=raw_sel)

    def _build_autoregressive_inputs(self, tgt):
        shifted = tgt.clone()
        shifted[:, 1:, :] = tgt[:, :-1, :]
        start = self.start_token_vector.view(1, 1, -1).to(tgt.device)
        shifted[:, 0, :] = start
        return shifted


    def _get_edge_from_adj_batch(self,adj, transform=False):
        '''

        return:
            edges : list
            edge : 2 X M
        '''

        edges = []

        for b in range(adj.shape[0]):
            sub_adj = adj[b]

            # index
            src, dst = torch.where(sub_adj > 0)

            # stack
            edge = torch.stack((src, dst), dim=1).to(adj.device)

            if transform:
                edges.append(edge.t().contiguous())
            else:
                edges.append(edge)

        return edges



    def _loss_each_featrue(self,pre,gt):

        # pre [batch,seq,feature_dim] ---- gt [batch,seq]
        return self.loss_fuc(pre.reshape(-1, pre.shape[2]), gt.reshape(-1))

    def _gaussian_label_loss(self, logits, targets):
        if self.label_smoothing_sigma <= 0:
            return self._loss_each_featrue(logits, targets)

        vocab_size = logits.shape[-1]
        flat_logits = logits.reshape(-1, vocab_size)
        flat_targets = targets.reshape(-1)
        valid_mask = flat_targets > 0
        if not valid_mask.any():
            return logits.new_tensor(0.0)

        selected_logits = flat_logits[valid_mask]
        selected_targets = flat_targets[valid_mask].float()
        log_probs = F.log_softmax(selected_logits, dim=-1)

        class_indices = torch.arange(vocab_size, device=logits.device).float().view(1, -1)
        sigma = max(self.label_smoothing_sigma, 1e-4)
        diffs = class_indices - selected_targets.unsqueeze(1)
        weights = torch.exp(-0.5 * (diffs / sigma) ** 2)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        return F.kl_div(log_probs, weights, reduction='batchmean')

    def _global_pos_monotonic_loss(self, predicts, mask, epsilon=None):
        if 'global_pos' not in self.feature_name_to_idx:
            return predicts[0].new_tensor(0.0)

        idx = self.feature_name_to_idx['global_pos']
        if idx >= len(predicts):
            return predicts[0].new_tensor(0.0)

        logits = predicts[idx]
        vocab_size = logits.shape[-1]
        
        if epsilon is None:
            epsilon = 1.0 / max(1.0, float(vocab_size - 1))

        probs = F.softmax(logits, dim=-1)
        value_vec = self._value_vector(logits.shape[-1], logits.device)
        
        # E[pos] = sum(Prob * Val)
        expected_pos = (probs * value_vec.view(1, 1, -1)).sum(dim=-1) # [B, S]
        
        # Delta = E[pos]_{j+1} - E[pos]_j
        delta = expected_pos[:, 1:] - expected_pos[:, :-1]
        
        # Penalty = ReLU(epsilon - Delta)
        penalty = F.relu(epsilon - delta)
        
        if mask is not None:
            # mask [B, S] -> valid transitions
            valid_trans = mask[:, :-1] * mask[:, 1:]
            if valid_trans.sum() > 1e-6:
                return (penalty * valid_trans).sum() / valid_trans.sum()
            else:
                return penalty.mean()
        
        return penalty.mean()

    def _structural_regularizers(self, predicts, mask, raw_feats=None, hidden=None):
        device = predicts[0].device
        batch, seq_len = predicts[0].shape[:2]
        if mask is not None:
            valid_mask = self._align_mask(mask, seq_len, device).float()
        else:
            valid_mask = torch.ones(batch, seq_len, device=device)
            
        total = predicts[0].new_tensor(0.0)
        total = total + self._global_pos_monotonic_loss(predicts, valid_mask)
        return total

    def _similarity_alignment_loss(self, predicts, gt, mask, adj, cfg):
        if adj is None or mask is None or not cfg.get('enabled'):
            return predicts[0].new_tensor(0.0)

        required_feats = cfg.get('feature_names') or ['position', 'duration', 'rhythm_pattern', 'melody_contour']
        feat_indices = {name: self.feature_name_to_idx.get(name) for name in required_feats}
        if not any(idx is not None for idx in feat_indices.values()):
            return predicts[0].new_tensor(0.0)

        seq_len = predicts[0].shape[1]
        batch_size = predicts[0].shape[0]
        device = predicts[0].device

        adj_slice = adj[:, :seq_len, :seq_len].float()
        mask_aligned = self._align_mask(mask, seq_len, device).float()

        dir_threshold = float(cfg.get('dir_threshold', 1.05))
        proximity_window = int(cfg.get('proximity_window', 2))
        normalizer = float(cfg.get('normalizer', 10.0))
        loss_type = str(cfg.get('loss_type', 'mse')).lower()

        undirected = (adj_slice > 0) & (adj_slice < dir_threshold)
        pair_mask = (mask_aligned.unsqueeze(1) * mask_aligned.unsqueeze(2)) > 0.5
        lower_mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=-1).bool()
        lower_mask = lower_mask.unsqueeze(0).expand(batch_size, -1, -1)
        edge_mask = undirected & pair_mask & lower_mask
        if edge_mask.sum() <= 0:
            return predicts[0].new_tensor(0.0)

        gt_slice = gt[:, :seq_len, :]

        def _match_prob_matrix(feature_name):
            idx = self.feature_name_to_idx.get(feature_name)
            if idx is None:
                return None
            logits = predicts[idx][:, :seq_len, :]
            probs = F.softmax(logits, dim=-1)
            targets = gt_slice[:, :, idx].long().clamp(min=0)
            targets = targets.clamp(max=probs.shape[-1]-1)
            one_hot = F.one_hot(targets, num_classes=probs.shape[-1]).float()
            return torch.matmul(probs, one_hot.transpose(1, 2))

        eq_pos = _match_prob_matrix('position')
        eq_dur = _match_prob_matrix('duration')
        eq_rhythm = _match_prob_matrix('rhythm_pattern')
        eq_melody = _match_prob_matrix('melody_contour')

        components = [comp for comp in [eq_pos, eq_dur, eq_rhythm, eq_melody] if comp is not None]
        if not components:
            return predicts[0].new_tensor(0.0)

        feature_hits = sum(components)
        if eq_rhythm is not None:
            feature_hits = feature_hits + eq_rhythm
        if eq_melody is not None:
            feature_hits = feature_hits + eq_melody
        if eq_dur is not None:
            feature_hits = feature_hits + eq_dur

        distance = torch.arange(seq_len, device=device)
        distance = (distance.view(1, seq_len, 1) - distance.view(1, 1, seq_len)).abs().float()
        distance_hits = torch.zeros_like(distance)
        if proximity_window > 0:
            mask_dist = (distance > 0) & (distance <= proximity_window)
            distance_hits[mask_dist] = (proximity_window - distance[mask_dist] + 1)
        distance_hits = distance_hits.expand(batch_size, -1, -1)

        pred_weight = (feature_hits + distance_hits) / max(normalizer, 1e-6)
        edge_mask_float = edge_mask.float()
        target_weight = adj_slice * edge_mask_float

        diff = (pred_weight - adj_slice)
        if loss_type == 'l1':
            diff_val = diff.abs()
        else:  # default mse
            diff_val = diff.pow(2)

        loss_total = (diff_val * edge_mask_float).sum()
        denom = edge_mask_float.sum() + 1e-6
        return loss_total / denom

    def _feature_consistency_loss(self, predicts, mask, adj, cfg):
        if adj is None or mask is None:
            return predicts[0].new_tensor(0.0)

        rp_idx = self.feature_name_to_idx.get('rhythm_pattern')
        mc_idx = self.feature_name_to_idx.get('melody_contour')
        if rp_idx is None and mc_idx is None:
            return predicts[0].new_tensor(0.0)

        mask_bool = mask > 0.5
        seq_len = adj.shape[1]
        pair_mask = (mask_bool.unsqueeze(1) & mask_bool.unsqueeze(2))
        lower = torch.tril(torch.ones(seq_len, seq_len, device=adj.device), diagonal=-1).bool()
        pair_mask = pair_mask & lower.unsqueeze(0)

        adj_types = torch.round(adj).long()
        rp_edges = ((adj_types == 1) | (adj_types == 3)) & pair_mask
        mc_edges = ((adj_types == 2) | (adj_types == 3)) & pair_mask

        total = predicts[0].new_tensor(0.0)
        total_weight = 0.0
        rp_weight = float(cfg.get('rp_weight', 1.0))
        mc_weight = float(cfg.get('mc_weight', 1.0))

        if rp_idx is not None and rp_edges.any() and rp_weight > 0:
            rp_probs = F.softmax(predicts[rp_idx], dim=-1)
            total = total + rp_weight * self._pairwise_distribution_loss(rp_probs, rp_edges)
            total_weight += rp_weight

        if mc_idx is not None and mc_edges.any() and mc_weight > 0:
            mc_probs = F.softmax(predicts[mc_idx], dim=-1)
            total = total + mc_weight * self._pairwise_distribution_loss(mc_probs, mc_edges)
            total_weight += mc_weight

        if total_weight == 0:
            return predicts[0].new_tensor(0.0)
        return total / total_weight

    def _pairwise_distribution_loss(self, probs, mask):
        diff = (probs.unsqueeze(2) - probs.unsqueeze(1)).pow(2).sum(dim=-1)
        weights = mask.float()
        denom = weights.sum()
        if denom <= 0:
            return probs.new_tensor(0.0)
        return (diff * weights).sum() / (denom + 1e-6)

    def _directed_cadence_loss(self, predicts, mask, adj, cfg):
        idx = self.feature_name_to_idx.get('cadence_tag')
        if idx is None:
            return predicts[0].new_tensor(0.0)

        logits = predicts[idx]
        seq_len = logits.shape[1]
        if seq_len < 2:
            return predicts[0].new_tensor(0.0)

        dir_threshold = float(cfg.get('dir_threshold', 1.05))
        incoming = (adj[:, :-1, 1:] >= dir_threshold)
        has_directed = incoming.any(dim=1)

        cadence_mask = torch.zeros_like(mask)
        cadence_mask[:, :-1] = has_directed.float()
        cadence_mask = cadence_mask * mask
        total = cadence_mask.sum()
        if total <= 0:
            return predicts[0].new_tensor(0.0)

        log_probs = F.log_softmax(logits, dim=-1)
        success_values = cfg.get('success_values', DEFAULT_SUCCESS_CADENCE)
        valid_idx = [val for val in success_values if val < log_probs.shape[-1]]
        if not valid_idx:
            return predicts[0].new_tensor(0.0)

        success_logprob = torch.logsumexp(log_probs[..., valid_idx], dim=-1)
        return -(success_logprob * cadence_mask).sum() / (total + 1e-6)

    def _bar_monotonic_loss(self, predicts, mask, cfg):
        idx = self.feature_name_to_idx.get('bar')
        if idx is None:
            return predicts[0].new_tensor(0.0)

        logits = predicts[idx]
        if logits.shape[1] < 2:
            return predicts[0].new_tensor(0.0)

        probs = F.softmax(logits, dim=-1)
        values = self._value_vector(probs.shape[-1], probs.device)
        expected = (probs * values.view(1, 1, -1)).sum(dim=-1)
        diffs = expected[:, 1:] - expected[:, :-1]

        pair_mask = mask[:, 1:] * mask[:, :-1]
        if pair_mask.sum() <= 0:
            return predicts[0].new_tensor(0.0)

        violations = F.relu(-diffs) * pair_mask
        return violations.sum() / (pair_mask.sum() + 1e-6)

    def _edge_ranking_loss(self, raw_feats, mask, adj, cfg):
        if raw_feats is None or adj is None or mask is None:
            return raw_feats.new_tensor(0.0) if raw_feats is not None else adj.new_tensor(0.0)

        repr_tensor = self._build_feature_repr(raw_feats, cfg)
        if repr_tensor is None:
            return raw_feats.new_tensor(0.0)

        device = raw_feats.device
        seq_len = repr_tensor.shape[1]
        mask_bool = mask > 0.5
        pair_mask = (mask_bool.unsqueeze(1) & mask_bool.unsqueeze(2)).float()
        weights = adj * pair_mask
        eye = torch.eye(seq_len, device=device).unsqueeze(0)
        weights = weights * (1 - eye)

        high_threshold = float(cfg.get('high_threshold', 0.8))
        low_threshold = float(cfg.get('low_threshold', 0.3))
        high_mask = weights >= high_threshold
        low_mask = (weights > 0) & (weights <= low_threshold)

        diff_matrix = (repr_tensor.unsqueeze(2) - repr_tensor.unsqueeze(1)).abs().mean(dim=-1)
        margin = float(cfg.get('margin', 0.05))
        max_pairs = max(1, int(cfg.get('max_pairs', 128)))

        total = raw_feats.new_tensor(0.0)
        pair_count = 0
        batch_size = raw_feats.shape[0]

        for b in range(batch_size):
            anchor_indices = torch.nonzero(mask_bool[b], as_tuple=False).view(-1)
            if anchor_indices.numel() == 0:
                continue
            for anchor in anchor_indices.tolist():
                high_candidates = torch.nonzero(high_mask[b, anchor], as_tuple=False).view(-1)
                low_candidates = torch.nonzero(low_mask[b, anchor], as_tuple=False).view(-1)
                if high_candidates.numel() == 0 or low_candidates.numel() == 0:
                    continue

                hi_idx = high_candidates[torch.randint(high_candidates.shape[0], (1,), device=high_candidates.device)].item()
                lo_idx = low_candidates[torch.randint(low_candidates.shape[0], (1,), device=low_candidates.device)].item()

                d_high = diff_matrix[b, anchor, hi_idx]
                d_low = diff_matrix[b, anchor, lo_idx]
                total = total + F.relu(margin + d_high - d_low)
                pair_count += 1
                if pair_count >= max_pairs:
                    break
            if pair_count >= max_pairs:
                break

        if pair_count == 0:
            return raw_feats.new_tensor(0.0)

        return total / pair_count

    def _contrastive_loss(self, hidden, mask, adj, cfg):
        if hidden is None or mask is None or adj is None:
            return hidden.new_tensor(0.0) if hidden is not None else adj.new_tensor(0.0)

        seq_len = hidden.shape[1]
        mask_bool = mask > 0.5
        pair_mask = mask_bool.unsqueeze(1) & mask_bool.unsqueeze(2)

        eye = torch.eye(seq_len, device=hidden.device).unsqueeze(0).bool()
        pos_threshold = float(cfg.get('pos_threshold', 0.8))
        adj_slice = adj
        pos_mask = (adj_slice >= pos_threshold) & pair_mask & (~eye)
        pos_count = pos_mask.sum()
        if pos_count == 0:
            return hidden.new_tensor(0.0)

        temperature = float(cfg.get('temperature', 0.5))
        norm_hidden = F.normalize(hidden, dim=-1)
        sim = torch.matmul(norm_hidden, norm_hidden.transpose(1, 2)) / max(temperature, 1e-3)

        valid_pairs = pair_mask & (~eye)
        sim = sim.masked_fill(~valid_pairs, -1e9)
        log_probs = sim - torch.logsumexp(sim, dim=-1, keepdim=True)

        return -(log_probs[pos_mask].mean())

    def _build_feature_repr(self, raw_feats, cfg):
        feature_names = cfg.get('feature_names') or self.feature_consistency_cfg.get('feature_names') or self.rpp_feature_selected
        scales_cfg = cfg.get('scales', {}) or {}
        indices = []
        scales = []
        for name in feature_names:
            idx = self.feature_name_to_idx.get(name)
            if idx is None:
                continue
            scale = float(scales_cfg.get(name, 1.0))
            if scale <= 0:
                scale = 1.0
            indices.append(idx)
            scales.append(scale)
        if not indices:
            return None

        feats = raw_feats[..., indices].float()
        scale_tensor = torch.tensor(scales, device=feats.device).view(1, 1, -1)
        return feats / scale_tensor

    def _value_vector(self, size, device):
        cached = self._value_vector_cache.get(size)
        if cached is None:
            values = torch.arange(size, dtype=torch.float32)
            if size > 1:
                values = values / float(size - 1)
            self._value_vector_cache[size] = values
            cached = values
        return cached.to(device)

    def _build_predicted_feature_repr(self, predicts, indices):
        reprs = []
        for idx in indices:
            logits = predicts[idx]
            probs = F.softmax(logits, dim=-1)
            values = self._value_vector(probs.shape[-1], logits.device)
            reprs.append((probs * values.view(1, 1, -1)).sum(dim=-1, keepdim=True))
        return torch.cat(reprs, dim=-1)

    def _build_groundtruth_feature_repr(self, gt, indices):
        reprs = []
        for idx in indices:
            feature_name = self.rpp_feature_selected[idx]
            scale = max(1, int(self.rpp_feature_dict.get(feature_name, 1)) - 1)
            values = gt[..., idx].float() / float(scale)
            reprs.append(values.unsqueeze(-1))
        return torch.cat(reprs, dim=-1)


    def _get_key_padding_mask(self,tgt_mask):
        '''

        return:
            key_padding_mask (N,S)
        '''
        key_padding_mask = tgt_mask == 0
        return key_padding_mask.to(tgt_mask.device)


    def _get_square_subsequent_mask(self, sz: int, device: Optional[torch.device] = None) -> Tensor:
        """
        Generate a causal (subsequent) mask for transformer decoder/encoder use.
        Returns a tensor of shape (sz, sz) where positions (i, j) with j>i are masked.
        This produces a float mask filled with -inf for masked positions which is compatible
        with PyTorch's Transformer modules (they add the mask directly to attention scores).
        """
        if device is None:
            device = torch.device('cpu')
        # upper triangular matrix with -inf above the main diagonal, 0 on and below diagonal
        mask = torch.triu(torch.ones((sz, sz), device=device), diagonal=1)
        # convert to float mask with -inf where masked
        mask = mask.float().masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
        return mask

    def _align_mask(self, mask, target_len, device):
        if mask.dim() != 2:
            mask = mask.view(mask.shape[0], -1)
        cur_len = mask.shape[1]
        if cur_len > target_len:
            mask = mask[:, :target_len]
        elif cur_len < target_len:
            pad = mask.new_zeros(mask.shape[0], target_len - cur_len)
            mask = torch.cat([mask, pad], dim=1)
        return mask.to(device)



class RppEmbedding(nn.Module):
    def __init__(self,cfg): 
        super(RppEmbedding,self).__init__()



        # d_embed
        d_embed = cfg['d_embed']
        d_model = cfg['d_model']

        features = cfg['rpp_feature_selected']

        dim_dict = cfg['rpp_feature_dict']


        self.embeddings = nn.ModuleList()
        for f in features:
            self.embeddings.append(nn.Embedding(dim_dict[f], d_embed,padding_idx=0))


        self.token_embedding = nn.Linear(len(self.embeddings) * d_embed, d_model)

    def forward(self,sample):

        # Input: (batch_size,seq,feature) -> output: (batch_size,seq,d_embed)
        embeds = [self.embeddings[i](sample[...,i]) for i in range(len(self.embeddings))]
        embeds = torch.cat(embeds, -1) 
        embeds = self.token_embedding(embeds)
        return embeds 

class NoteEmbedding(nn.Module):
    def __init__(self, cfg):  
        super(NoteEmbedding,self).__init__()



        d_embed = cfg['d_embed']
        d_model = cfg['d_model']

        
        features = cfg['note_feature_selected']

        
        dim_dict = cfg['note_feature_dim_dict']

        
        self.embeddings = nn.ModuleList()
        for f in features:
            self.embeddings.append(nn.Embedding(dim_dict[f], d_embed,padding_idx=0))



        self.token_embedding = nn.Linear(len(self.embeddings) * d_embed, d_model)

    def forward(self,sample):

        # Input: (batch_size,seq,feature) -> output: (batch_size,seq,embedding)
        embeds = [self.embeddings[i](sample[...,i]) for i in range(len(self.embeddings))]
        embeds = torch.cat(embeds, -1) 
        embeds = self.token_embedding(embeds)
        return embeds 

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with optional structural and feature-difference cues."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)
        self.register_buffer("position_index", torch.arange(max_len).float().unsqueeze(0))

    def forward(self, x, key_padding_mask=None, raw_feats=None):
        seq_len = x.shape[1]
        base = x + self.pe[:, :seq_len, :].requires_grad_(False)
        return self.dropout(base)


