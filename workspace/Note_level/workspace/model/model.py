import yaml
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import TransformerDecoder,TransformerDecoderLayer
import math
import itertools
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import pickle
import os
import ast

CONFIG_PATH = '../../config/config.yaml'

class NoteTransformer(nn.Module):
    def __init__(self,cfg):
        super(NoteTransformer, self).__init__()

        self.cfg = cfg
        d_model = cfg['d_model']
        num_layer = cfg['num_layers']
        nhead = cfg['n_head']
        dropout = cfg['dropout']
        
        # embedding
        self.rpp_pe_encoding = PositionalEncoding(
            d_model=d_model,
            dropout=dropout,
            max_len=cfg['rpp_seq_max']
        )
        self.rpp_embedding = RppEmbedding(cfg=cfg)

        self.pos_feat_idx = None
        self.bar_feat_idx = None
        
        # Identify feature indices for structural aggregation
        # Handle both casing styles for robustness
        rpp_feats = [f.lower() for f in cfg.get('rpp_feature_selected', [])]
        
        self.pos_feat_idx = None
        if 'position' in rpp_feats:
             self.pos_feat_idx = rpp_feats.index('position')
             
        self.bar_feat_idx = None
        if 'bar' in rpp_feats:
             self.bar_feat_idx = rpp_feats.index('bar')

        self.dur_feat_idx = None
        if 'duration' in rpp_feats:
             self.dur_feat_idx = rpp_feats.index('duration')
        

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=num_layer)
        self.decoder = DecoderTransformer(cfg=cfg)
        # Loss: ignore PAD (assumed index 0) when computing cross-entropy so the model
        # is not incentivized to predict padding tokens as a "safe" output.
        self.loss_fuc = nn.CrossEntropyLoss(ignore_index=0)
        self.feature_loss_weights = self._build_feature_loss_weights()

    def _expand_memory_to_ticks(self, memory, rpp_feat):
        """
        Expand [batch, num_nodes, d_model] memory to [batch, max_ticks, d_model]
        based on bar/position/duration features in rpp_feat.
        """
        # Feature indices
        bar_idx = self.bar_feat_idx
        pos_idx = self.pos_feat_idx
        dur_idx = self.dur_feat_idx 

        if bar_idx is None or pos_idx is None or dur_idx is None:
             # Fallback if features missing
             return None, None

        # Configs
        bar_ticks = 1920 
        max_bars = 50 
        total_ticks = max_bars * bar_ticks
        resolution = 120 

        B, N, D = memory.shape
        grid_len = total_ticks // resolution 
        
        # New memory: [B, grid_len, D]
        # This will be sparse (only filled where RPP nodes exist/cover)
        expanded_mem = torch.zeros(B, grid_len, D, device=memory.device)
        
        # Expanded Mask: True means PADDING (ignore in Attention)
        # Default all True (all ignored) until filled
        key_padding_mask = torch.ones(B, grid_len, device=memory.device, dtype=torch.bool)
        
        
        rpp_np = rpp_feat.cpu().numpy()
        

        
        for b in range(B):

            valid_nodes = (rpp_np[b, :, bar_idx] > 0)
            

            bars = np.maximum(0, rpp_np[b, valid_nodes, bar_idx] - 1)
            poss = np.maximum(0, rpp_np[b, valid_nodes, pos_idx] - 1)
            durs = np.maximum(0, rpp_np[b, valid_nodes, dur_idx] - 1) 
            if len(bars) == 0: continue


            start_ticks = bars * bar_ticks + poss * resolution

            dur_ticks = (durs + 1) * resolution 
            
            end_ticks = start_ticks + dur_ticks
            
            start_grids = (start_ticks // resolution).astype(int)
            end_grids = (end_ticks // resolution).astype(int)
            
            # Clip
            start_grids = np.clip(start_grids, 0, grid_len)
            end_grids = np.clip(end_grids, 0, grid_len)
            
            # Fill
            # For each valid node n, fill range [start, end)

            valid_indices = np.where(valid_nodes)[0]
            for i, n_idx in enumerate(valid_indices):
                s, e = start_grids[i], end_grids[i]
                if e > s:

                    expanded_mem[b, s:e, :] = memory[b, n_idx, :]
                    key_padding_mask[b, s:e] = False
                    
        return expanded_mem, key_padding_mask

    def forward(self,V,tgt,tgt_mask=None,memory_mask=None,adj=None):
        # Input:  V [batch,seq,feature] | E [batch,2,index] | target [batch,token_seq,feature] | tgt_mask [batch,token_seq]
        # Output: [batch,token_seq,sum(dim_feature)]

        if torch.isnan(V).any():
             print("[Model] NaN detected in input V (rpp_feat)")
        if torch.isnan(tgt).any():
             print("[Model] NaN detected in input tgt (note_feat)")


        
        rpp = self.rpp_embedding(V, return_all=False)

        token_mask = (V != 0).any(dim=-1)
        if token_mask.dtype != torch.long and token_mask.dtype != torch.int64:
            token_mask = token_mask.long()
        pad_mask = None
        if token_mask is not None:
            pad_mask = token_mask == 0
            if pad_mask.any():
                all_masked = pad_mask.all(dim=-1)
                if all_masked.any():
                    pad_mask[all_masked, 0] = False
        key_padding = token_mask.float() if token_mask is not None else None
        rpp_embed = self.rpp_pe_encoding(rpp, key_padding_mask=key_padding)
        if torch.isnan(rpp_embed).any():
             print("[Model] NaN detected in PositionalEncoding output")

        memory = self.encoder(rpp_embed, src_key_padding_mask=pad_mask)
        if torch.isnan(memory).any():
             print("[Model] NaN detected in Encoder output")

        # [Frame-based Alignment] Expand Memory from RPP Nodes to Absolute Ticks
        expanded_memory, expanded_mask = self._expand_memory_to_ticks(memory, V)
        
       
        use_expanded = False
        if expanded_memory is not None:

             if (~expanded_mask).any():
                  use_expanded = True
                  memory = expanded_memory
                  memory_mask_input = expanded_mask
             else:

                  pass
        
        if not use_expanded:
             memory_mask_input = memory_mask if memory_mask is not None else token_mask
        
       
        
        output = self.decoder(tgt=tgt,memory=memory,tgt_mask=tgt_mask,memory_mask=memory_mask_input)
        if torch.isnan(output).any():
             print("[Model] NaN detected in Decoder output")

        return output


    def predict_transform(self,X):
        # Input:  X [batch,1,d_model]
        # Output: X [batch,d_feat]
        feature_size = [self.cfg['note_feature_dim_dict'][f] for f in self.cfg['note_feature_selected']]
        div_index =[0] + [sum(feature_size[:i+1]) for i in range(len(feature_size))]

        
        Xs = [X[..., i:j] for i, j in zip(div_index[:-1], div_index[1:])]
        Xs = [torch.squeeze(x, dim=1) for x in Xs]
        temperatures = list(self.cfg.get('temperature', [1.0] * len(Xs)))
        topk = int(self.cfg.get('sampling_topk', 0) or 0)
        deterministic = bool(self.cfg.get('deterministic_inference', False))

        sampled = []
        for idx, logits in enumerate(Xs):
            temp = float(temperatures[idx]) if idx < len(temperatures) else 1.0
            logits = logits / max(temp, 1e-6)
            probs = F.softmax(logits, dim=1)

            if deterministic:
                choice = torch.argmax(probs, dim=1, keepdim=True)
            elif topk > 0 and topk < probs.shape[1]:
                vals, indices = torch.topk(probs, topk, dim=1)
                vals = vals / vals.sum(dim=1, keepdim=True)
                sampled_idx = torch.multinomial(vals, num_samples=1)
                choice = torch.gather(indices, dim=1, index=sampled_idx)
            else:
                choice = torch.multinomial(probs, num_samples=1)

            sampled.append(choice)

        Xs = torch.cat(sampled, dim=1)   # [batch,token_seq]
        return Xs

    def loss(self,predict,gt,feature_masks=None):
        # X [batch,token_seq,sum(dim_feature)]   --------    Y [batch,token_seq,note_feature]
        # X [10,512,453]                         --------    Y [10,512,5]

        # Split Feature
        feature_size = [self.cfg['note_feature_dim_dict'][f] for f in self.cfg['note_feature_selected']]
        assert len(feature_size) == gt.shape[2], 'in model.loss : Dim of Featureselected != Dim of GroundTruth[2]'
        assert sum(feature_size) == predict.shape[2], 'in model.Loss : Dim(predict[2]) != Dim(features)'
        div_index =[0] + [sum(feature_size[:i+1]) for i in range(len(feature_size))]
        predicts = [predict[...,i:j] for i,j in zip(div_index[:-1],div_index[1:])]
        gts = [gt[...,i] for i in range(gt.shape[2])]

        # Gather loss
        if feature_masks is not None and len(feature_masks) != len(predicts):
            raise ValueError("feature_masks must align with note_feature_selected")

        losses = []
        for idx, (pre, y) in enumerate(zip(predicts, gts)):
            mask = None
            if feature_masks is not None:
                mask = feature_masks[idx]
            losses.append(self._loss_each_featrue(pre=pre, gt=y, mask=mask))
        if not losses:
            return predict.new_tensor(0.0)

        weights = self.feature_loss_weights
        if weights is None or len(weights) != len(losses):
            return sum(losses) / len(losses)

        weighted_sum = sum(w * l for w, l in zip(weights, losses))
        weight_total = sum(weights)
        if weight_total <= 0:
            return sum(losses) / len(losses)
        return weighted_sum / weight_total

    def _loss_each_featrue(self,pre,gt,mask=None):

        # pre [batch,seq,feature_dim] ---- gt [batch,seq]
        logits = pre.view(-1, pre.shape[2])
        targets = gt.view(-1)
        if mask is not None:
            mask_flat = mask.view(-1)
            if mask_flat.dtype != torch.bool:
                mask_flat = mask_flat > 0
            if not torch.any(mask_flat):
                return logits.new_tensor(0.0)
            logits = logits[mask_flat]
            targets = targets[mask_flat]
        if logits.shape[0] == 0:
            return pre.new_tensor(0.0)
        

        if (targets == 0).all():
             return pre.new_tensor(0.0)

        return self.loss_fuc(logits, targets)

    def _build_feature_loss_weights(self):
        feature_names = self.cfg.get('note_feature_selected', [])
        if not feature_names:
            return None
        cfg_weights = self.cfg.get('note_feature_loss_weights') or {}
        weights = []
        for name in feature_names:
            weight = float(cfg_weights.get(name, 1.0))
            weights.append(max(weight, 0.0))
        if not any(weights):
            return None
        return weights



class DecoderTransformer(nn.Module):
    def __init__(self,cfg):
        super(DecoderTransformer, self).__init__()

        # init
        d_model,n_head,dropout,num_layers,feature,note_dim_dict = cfg['d_model'],cfg['n_head'],cfg['dropout'],\
                                                    cfg['num_layers'],cfg['note_feature_selected'],cfg['note_feature_dim_dict']
        d_out = np.sum([note_dim_dict[f] for f in feature])

        # Embedding   Note [batch_size,seq,feature] -> Note [batch_size,seq,d_model]
        self.embedding = NoteEmbedding(cfg=cfg)
        self.pe_encoding = PositionalEncoding(d_model=d_model,dropout=dropout,max_len=cfg['token_seq_max'])

        # transformer decoder
        transformerdecoder_layer = TransformerDecoderLayer(d_model=d_model,nhead=n_head,dropout=dropout,batch_first=True, norm_first=True)
        self.transformerdecoder = TransformerDecoder(transformerdecoder_layer,num_layers=num_layers)

        # Linear
        self.linear = nn.Linear(d_model,int(d_out))

    def forward(self,tgt,memory,tgt_mask=None,memory_mask=None):
        # Input:  target [batch,token_seq,feature]  | memory [batch,rpp_seq,d_model] | tgt_mask [batch,token_seq]
        # Output: V [batch,token_seq,sum(dim_feature)]


        X = self.embedding(tgt) # [batch,seq,feature] -> [batch,seq,d_model]
        X = self.pe_encoding(X)


        # Transformer Decoder
        # tgtmask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1],device=tgt.device)
        tgtmask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(tgt.device)
        tgtmask_padding = self._get_tgt_key_padding_mask(tgt_mask=tgt_mask) if tgt_mask!=None else None
        memory_mask = self._get_memory_mask(memory_mask=memory_mask) if memory_mask!=None else None

        X = self.transformerdecoder(tgt=X,memory=memory,tgt_mask=tgtmask,
                                    tgt_key_padding_mask=tgtmask_padding,memory_key_padding_mask = memory_mask)

        # Linear
        return self.linear(F.relu(X))

    def _get_memory_mask(self,memory_mask):
        # Input: [batch,seq]
        # Output: [batch,seq] with float -inf for padding
        memory_pad_mask = torch.zeros(memory_mask.shape,dtype=torch.bool,device=memory_mask.device)
        memory_pad_mask[memory_mask == 1] = False
        memory_pad_mask[memory_mask == 0] = True
        
        # Guard against fully masked memory sequences
        if memory_pad_mask.any():
            all_masked = memory_pad_mask.all(dim=-1)
            if all_masked.any():
                memory_pad_mask[all_masked, 0] = False
        
        # Convert to float for compatibility with Float attn_mask
        float_mask = torch.zeros_like(memory_pad_mask, dtype=torch.float)
        float_mask.masked_fill_(memory_pad_mask, float('-inf'))

        return float_mask

    def _get_tgt_key_padding_mask(self,tgt_mask):
        # Input: [batch,seq]
        # Output: [batch] with float -inf for padding

        key_padding_mask = torch.zeros(tgt_mask.shape,dtype=torch.bool)
        key_padding_mask[tgt_mask == 0] = True
        key_padding_mask[tgt_mask == 1] = False
        
        # Guard against fully masked sequences
        # Ensure the first token is never masked to prevent NaN in causal attention for padding start
        if key_padding_mask.any():
            key_padding_mask[:, 0] = False
        
        # Convert to float
        float_mask = torch.zeros_like(key_padding_mask, dtype=torch.float)
        float_mask.masked_fill_(key_padding_mask, float('-inf'))

        return float_mask.to(tgt_mask.device)


class RppEmbedding(nn.Module):
    def __init__(self,cfg): 
        super(RppEmbedding,self).__init__()


        d_embed = cfg['d_embed']
        d_model = cfg['d_model']
        self.d_model = d_model

        features = cfg['rpp_feature_selected']

        dim_dict = cfg['rpp_feature_dim_dict']


        self.embeddings = nn.ModuleList()
        for f in features:
            self.embeddings.append(nn.Embedding(dim_dict[f], d_embed,padding_idx=0))


        self.token_embedding = nn.Linear(len(self.embeddings) * d_embed, d_model)

    def forward(self, sample, return_all=False):

        # Input: (batch_size,seq,feature) -> output: (batch_size,seq,d_embed)
        embeds_list = [self.embeddings[i](sample[...,i]) for i in range(len(self.embeddings))]
        embeds = torch.cat(embeds_list, -1) 
        embeds = self.token_embedding(embeds)
        embeds = embeds * math.sqrt(self.d_model) 
        
        if return_all:
            return embeds, embeds_list
        return embeds 

class NoteEmbedding(nn.Module):
    def __init__(self, cfg):  
        super(NoteEmbedding,self).__init__()

        # d_embed
        d_embed = cfg['d_embed']
        d_model = cfg['d_model']
        self.d_model = d_model

        features = cfg['note_feature_selected']

        dim_dict = cfg['note_feature_dim_dict']


        self.embeddings = nn.ModuleList()
        for f in features:
            self.embeddings.append(nn.Embedding(dim_dict[f], d_embed,padding_idx=0))



        self.token_embedding = nn.Linear(len(self.embeddings) * d_embed, d_model)

    def forward(self,sample):

        # Input: (batch_size,seq,feature) -> output: (batch_size,seq,embedding)
        embeds = [self.embeddings[i](sample[...,i]) for i in range(len(self.embeddings))]
        embeds = torch.cat(embeds, -1) # cat
        embeds = self.token_embedding(embeds)
        embeds = embeds * math.sqrt(self.d_model) 
        return embeds 

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding optionally augmented with adjacency-aware structure."""

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

    def forward(self, x, key_padding_mask=None):
        seq_len = x.size(1)
        base = x + self.pe[:, :seq_len].requires_grad_(False)
        return self.dropout(base)

# Datasets
class FirstNoteGuidanceMixin:
    def _init_first_note_guidance(self, config):
        self._first_note_enabled = False
        self._bar_feature_index = self._feature_index_or_none('bar')
        self._position_feature_index = self._feature_index_or_none('position')
        self._rhythm_feature_index = self._feature_index_or_none('rhythm_pattern')
        mapping_path = config.get('rpp_feat2idx_path') if config else None
        self._rhythm_pattern_note_counts = self._load_rhythm_pattern_note_counts(mapping_path)
        if (self._bar_feature_index is None or
                self._position_feature_index is None or
                self._rhythm_feature_index is None or
                self._rhythm_pattern_note_counts is None):
            return
        self._first_note_enabled = True

    def _feature_index_or_none(self, key):
        if not hasattr(self, 'rpp_feature_selected'):
            return None
        try:
            return self.rpp_feature_selected.index(key)
        except ValueError:
            return None

    def _load_rhythm_pattern_note_counts(self, mapping_path):
        if not mapping_path or not os.path.exists(mapping_path):
            print(f"FirstNoteGuidanceMixin: rhythm pattern mapping missing at '{mapping_path}', guidance disabled.")
            return None
        try:
            with open(mapping_path, 'rb') as f:
                mapping = pickle.load(f).get('rhythm_pattern')
        except Exception as exc:
            print(f"FirstNoteGuidanceMixin: failed to load rhythm_pattern mapping ({exc}); guidance disabled.")
            mapping = None
        if not mapping:
            return None
        size = max(mapping.values()) + 1 if mapping else 0
        counts = [0] * size
        for key, idx in mapping.items():
            if idx >= len(counts):
                counts.extend([0] * (idx - len(counts) + 1))
            if key in ('<pad>', '<start>'):
                counts[idx] = 0
                continue
            try:
                pattern = ast.literal_eval(key)
            except Exception:
                pattern = None
            if isinstance(pattern, tuple):
                counts[idx] = len(pattern)
            elif isinstance(pattern, list):
                counts[idx] = len(pattern)
            elif isinstance(pattern, int):
                counts[idx] = 1
            else:
                counts[idx] = 0
        return counts

    def _note_count_from_rhythm(self, token_id):
        if (self._rhythm_pattern_note_counts is None or
                token_id < 0 or
                token_id >= len(self._rhythm_pattern_note_counts)):
            return 0
        return self._rhythm_pattern_note_counts[token_id]

    def _build_first_note_metadata(self, data):
        note_feat = data.get('note_feat')
        seq_len = int(note_feat.shape[0]) if note_feat is not None else 0
        mask = torch.zeros(seq_len, dtype=torch.bool)
        forced_bar = torch.zeros(seq_len, dtype=torch.long)
        forced_pos = torch.zeros(seq_len, dtype=torch.long)

        if not self._first_note_enabled or note_feat is None:
            return {
                'first_note_mask': mask,
                'first_note_bar': forced_bar,
                'first_note_position': forced_pos
            }

        rpp_feat = data.get('rpp_feat')
        if rpp_feat is None:
            return {
                'first_note_mask': mask,
                'first_note_bar': forced_bar,
                'first_note_position': forced_pos
            }

        rpp_mask = data.get('rpp_mask')
        note_mask = data.get('note_mask')

        valid_notes = int(torch.count_nonzero(note_mask).item()) if note_mask is not None else seq_len
        valid_notes = max(valid_notes, 0)
        valid_rpp = int(torch.count_nonzero(rpp_mask).item()) if rpp_mask is not None else int(rpp_feat.shape[0])
        valid_rpp = max(valid_rpp, 0)

        note_ptr = 1  # skip SOS token at index 0
        max_notes = min(valid_notes, seq_len)
        for rpp_idx in range(1, min(valid_rpp, rpp_feat.shape[0])):
            token_id = int(rpp_feat[rpp_idx, self._rhythm_feature_index].item())
            note_count = self._note_count_from_rhythm(token_id)
            if note_count <= 0:
                continue
            if note_ptr >= max_notes:
                break
            mask_idx = note_ptr
            if mask_idx >= seq_len:
                break
            mask[mask_idx] = True
            forced_bar[mask_idx] = rpp_feat[rpp_idx, self._bar_feature_index]
            forced_pos[mask_idx] = rpp_feat[rpp_idx, self._position_feature_index]
            note_ptr += note_count

        return {
            'first_note_mask': mask,
            'first_note_bar': forced_bar,
            'first_note_position': forced_pos
        }


class MidiDataset(IterableDataset, FirstNoteGuidanceMixin):
    def __init__(self, config, dataroot):
        import os
        import glob
        import random
        self.dataroot = dataroot
        self.files = []
        if os.path.isdir(dataroot):
            self.files = sorted(glob.glob(os.path.join(dataroot, '*.pkl')))
        else:
            self.files = [dataroot]
        
        self.note_feature_all = ['bar', 'position', 'duration', 'pitch', 'velocity'] 
        self.note_feature_selected = config['note_feature_selected']
        self.note_feature_selected_index = [self.note_feature_all.index(s) for s in self.note_feature_selected]
        self.rpp_feature_all = config['rpp_feature_all']
        self.rpp_feature_selected = config['rpp_feature_selected']
        self.rpp_feature_selected_index = [self.rpp_feature_all.index(s) for s in self.rpp_feature_selected]
        self._init_first_note_guidance(config)
    
    def __iter__(self):
        worker_info = get_worker_info()
        import random
        files_list = list(self.files)
        
        if worker_info is not None:
             # Split workload
             per_worker = int(np.ceil(len(files_list) / float(worker_info.num_workers)))
             iter_start = worker_info.id * per_worker
             iter_end = min(iter_start + per_worker, len(files_list))
             files_list = files_list[iter_start:iter_end]
             
        random.shuffle(files_list)
        
        for pkl_path in files_list:
             data_pool = []
             try:
                 with open(pkl_path, 'rb') as f:
                     data_pool = pickle.load(f)
             except Exception as e:
                 print(f"Error loading {pkl_path}: {e}")
                 continue
                 
             random.shuffle(data_pool)
             
             for entry in data_pool:
                 try:
                     processed = self._process_item(entry)
                     if processed is not None:
                         yield processed
                 except Exception:
                     continue

    def _process_item(self, raw_data):
        # Validation from original __init__
        if 'rpp_feat' not in raw_data or 'note_feat' not in raw_data or 'condition' not in raw_data or 'name' not in raw_data:
            return None

        # check rpp_feat dimension
        rpp = raw_data['rpp_feat']
        if hasattr(rpp, 'shape'):
             if len(rpp.shape) < 2:
                 return None
             cols = int(rpp.shape[1])
             if not (cols == len(self.rpp_feature_selected) or cols > max(self.rpp_feature_selected_index)):
                 return None
        else:
             return None


        data = {"name": None,
                "condition": None,
                "rpp_feat": None,
                "rpp_mask":None,
                "note_feat": None,
                'note_feat_gt':None,
                'note_mask':None,
                "num_nodes": None,
                "n_in_sequences": None,
                "n_in_stride": None,
                "n_in_starts": None}
        
        for k,v in raw_data.items():
            if k in data.keys():
                if k == 'name':
                    data[k] = v
                elif k in ('num_nodes', 'n_in_sequences', 'n_in_stride', 'n_in_starts'):
                    data[k] = v
                else:
                    data[k] = torch.from_numpy(v).long()
        
        # feat select
        try:
            if hasattr(data['rpp_feat'], 'shape'):
                cols = int(data['rpp_feat'].shape[1])
                if cols == len(self.rpp_feature_selected):
                    pass
                else:
                    data['rpp_feat'] = data['rpp_feat'][:, self.rpp_feature_selected_index]
            else:
                return None
        except Exception:
            return None

        try:
            if data['note_feat'] is not None:
                data['note_feat'] = data['note_feat'][:, self.note_feature_selected_index]
            if data['note_feat_gt'] is not None:
                data['note_feat_gt'] = data['note_feat_gt'][:, self.note_feature_selected_index]
        except Exception:
             return None
             
        num_nodes = data['num_nodes'] if data['num_nodes'] is not None else raw_data.get('num_nodes')
        if num_nodes is None and data['rpp_mask'] is not None:
            num_nodes = int(torch.count_nonzero(data['rpp_mask']).item())
        if num_nodes is not None:
            data['num_nodes'] = torch.tensor(int(num_nodes), dtype=torch.long)

        if data['rpp_mask'] is None and data['rpp_feat'] is not None:
            data['rpp_mask'] = (data['rpp_feat'] != 0).any(dim=1).long()
        if data['note_mask'] is None and data['note_feat'] is not None:
            data['note_mask'] = (data['note_feat'] != 0).any(dim=1).long()

        # Check if Sample is Valid
        if data['rpp_mask'] is not None and torch.sum(data['rpp_mask']) == 0:
            return None
        if data['note_mask'] is not None and torch.sum(data['note_mask']) == 0:
            return None

        if data['rpp_mask'] is None and data['rpp_feat'] is not None:
            rpp_nonzero = (data['rpp_feat'] != 0).any(dim=1).long()
            data['rpp_mask'] = rpp_nonzero
        if data['note_mask'] is None and data['note_feat'] is not None:
            note_nonzero = (data['note_feat'] != 0).any(dim=1).long()
            data['note_mask'] = note_nonzero


        sequences_source = raw_data.get('n_in_sequences')
        if sequences_source:
             sequences = [tuple(int(x) for x in seq) for seq in sequences_source if len(seq)]
        else:
            stride = int(raw_data.get('n_in_stride', 1) or 1)
            starts = raw_data.get('n_in_starts', (3, 4))
            if isinstance(starts, (list, tuple)):
                starts_iter = [int(s) for s in starts]
            else:
                starts_iter = [int(starts)]
            num_nodes_est = int(num_nodes) if num_nodes is not None else 0
            sequences = []
            for start in starts_iter:
                if start >= num_nodes_est or num_nodes_est == 0:
                    continue
                seq = list(range(start, num_nodes_est, stride))
                if seq:
                    sequences.append(tuple(seq))
        data['n_in_sequences'] = sequences

        stride = int(raw_data.get('n_in_stride', 1) or 1)
        data['n_in_stride'] = torch.tensor(stride, dtype=torch.long)

        starts = raw_data.get('n_in_starts', (3, 4))
        if isinstance(starts, (list, tuple)):
            starts_tensor = torch.tensor([int(s) for s in starts], dtype=torch.long)
        else:
            starts_tensor = torch.tensor([int(starts)], dtype=torch.long)
        data['n_in_starts'] = starts_tensor

        data.update(self._build_first_note_metadata(data))
                            
        return data



    def __len__(self):
        import os
        import glob
        import pickle
        import gc
        cache = os.path.join(self.dataroot, "length.txt")
        files = glob.glob(os.path.join(self.dataroot, '*.pkl')) if os.path.isdir(self.dataroot) else [self.dataroot]
        
        need_recompute = True
        if os.path.exists(cache):
            cache_mtime = os.path.getmtime(cache)
            if all(os.path.getmtime(f) <= cache_mtime for f in files):
                try:
                    return int(open(cache).read().strip())
                except:
                    pass


        print(f"[Dataset] Missing or outdated length cache at {self.dataroot}. Recomputing length from {len(files)} files... (this may take a minute)")
        total = 0
        for fpath in sorted(files):
            try:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                    total += len(data)
                    del data
                    gc.collect()
            except Exception as e:
                print(f"[Dataset] Error loading {fpath} for length computation: {e}")
        
        try:
            with open(cache, "w") as f:
                f.write(str(total))
        except Exception as e:
            print(f"[Dataset] Warning: Could not write length cache to {cache}: {e}")
            
        return total

    
 
class MidiDataset_Inference(Dataset, FirstNoteGuidanceMixin):
    def __init__(self, config,dataroot):
        with open(dataroot,'rb') as f:
            raw_list = pickle.load(f)

        self.note_feature_all = ['bar', 'position', 'duration', 'pitch', 'velocity'] 
        self.note_feature_selected = config['note_feature_selected']
        

        self.note_feature_selected_index = [self.note_feature_all.index(s) for s in self.note_feature_selected]

        # Validate and filter entries so DataLoader never receives incomplete samples
        self.rpp_feature_all = config['rpp_feature_all']
        self.rpp_feature_selected = config['rpp_feature_selected']
        self.rpp_feature_selected_index = [self.rpp_feature_all.index(s) for s in self.rpp_feature_selected]
        self._init_first_note_guidance(config)

        valid_list = []
        skipped = 0
        for entry in raw_list:
            try:

                if 'rpp_feat' not in entry or 'note_feat' not in entry or 'condition' not in entry or 'name' not in entry :
                    skipped += 1
                    continue

               
                rpp = entry['rpp_feat']
                if hasattr(rpp, 'shape'):
                    if len(rpp.shape) < 2:
                        skipped += 1
                        continue
                    cols = int(rpp.shape[1])
                    if not (cols == len(self.rpp_feature_selected) or cols > max(self.rpp_feature_selected_index)):
                        skipped += 1
                        continue
                else:
                    skipped += 1
                    continue

                valid_list.append(entry)
            except Exception:
                skipped += 1
                continue

        if skipped > 0:
            print(f"MidiDataset_Inference: filtered out {skipped} invalid/unsupported records from '{dataroot}'")

        self.data_list = valid_list

    def __getitem__(self, index):
        raw_data = self.data_list[index]
        data = {"name": None,
                "condition": None,
                "rpp_feat": None,
                "rpp_mask":None,
                "note_feat": None,
                "note_mask": None,
                "num_nodes": None,
                "n_in_sequences": None,
                "n_in_stride": None,
                "n_in_starts": None}


        # tensorlize
        for k,v in raw_data.items():
            if k in data.keys():
                if k == 'name':
                    data[k] = v
                elif k in ('num_nodes', 'n_in_sequences', 'n_in_stride', 'n_in_starts'):
                    data[k] = v
                else:
                    data[k] = torch.from_numpy(v).long()

        # feat select
        # If rpp_feat already contains only the selected columns, keep as-is. Otherwise select from full features.
        if data['rpp_feat'] is None:
            raise ValueError('Missing rpp_feat in data entry')
        if hasattr(data['rpp_feat'], 'shape'):
            cols = int(data['rpp_feat'].shape[1])
            if cols == len(self.rpp_feature_selected):
                pass
            else:
                data['rpp_feat'] = data['rpp_feat'][:, self.rpp_feature_selected_index]
        else:
            raise ValueError('rpp_feat has no shape')
        
        try:
            if data['note_feat'] is not None:

                data['note_feat'] = data['note_feat'][:, self.note_feature_selected_index]
            
        except Exception:
            raise ValueError(f"Error slicing note_feat for index {index}")
        
        num_nodes = data['num_nodes'] if data['num_nodes'] is not None else raw_data.get('num_nodes')
        if num_nodes is None and data['rpp_mask'] is not None:
            num_nodes = int(torch.count_nonzero(data['rpp_mask']).item())
        if num_nodes is not None:
            data['num_nodes'] = torch.tensor(int(num_nodes), dtype=torch.long)

        sequences_source = raw_data.get('n_in_sequences')
        if sequences_source:
            sequences = [tuple(int(x) for x in seq) for seq in sequences_source if len(seq)]
        else:
            stride = int(raw_data.get('n_in_stride', 1) or 1)
            starts = raw_data.get('n_in_starts', (3, 4))
            if isinstance(starts, (list, tuple)):
                starts_iter = [int(s) for s in starts]
            else:
                starts_iter = [int(starts)]
            num_nodes_est = int(num_nodes) if num_nodes is not None else 0
            sequences = []
            for start in starts_iter:
                if start >= num_nodes_est or num_nodes_est == 0:
                    continue
                seq = list(range(start, num_nodes_est, stride))
                if seq:
                    sequences.append(tuple(seq))
        data['n_in_sequences'] = sequences

        stride = int(raw_data.get('n_in_stride', 1) or 1)
        data['n_in_stride'] = torch.tensor(stride, dtype=torch.long)

        starts = raw_data.get('n_in_starts', (3, 4))
        if isinstance(starts, (list, tuple)):
            starts_tensor = torch.tensor([int(s) for s in starts], dtype=torch.long)
        else:
            starts_tensor = torch.tensor([int(starts)], dtype=torch.long)
        data['n_in_starts'] = starts_tensor

        if data['rpp_mask'] is None and data['rpp_feat'] is not None:
            data['rpp_mask'] = (data['rpp_feat'] != 0).any(dim=1).long()
        if data['note_mask'] is None and data['note_feat'] is not None:
            data['note_mask'] = (data['note_feat'] != 0).any(dim=1).long()

        data.update(self._build_first_note_metadata(data))

        return data


    def __len__(self):
        return len(self.data_list)  
