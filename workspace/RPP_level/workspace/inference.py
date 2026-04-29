import pickle
import random
import time
import json
import ast
import torch
from torch.utils.data import DataLoader
import yaml
from model import RPPTransformer
from tqdm import tqdm
import os
import argparse
import numpy as np
import torch.nn.functional as F


# Parser
parser = argparse.ArgumentParser(description="s2g inference")
parser.add_argument("-e","--expdir", type=str, default="", help="specify RecordDirectory")
parser.add_argument("-i","--input", type=str, default="auto", help="specify data path")
parser.add_argument("-o","--output", type=str, default="auto", help="specify save path")
parser.add_argument("-m","--mode",type=str, default="all", help="specify stage if stage is seperated * edge * feature")
parser.add_argument("--temperature", type=float, default=1.0, help="softmax temperature used when --sampling_mode=stochastic")

args = parser.parse_args()


def parse_thresholds_arg(raw_value, num_classes):
    """Parse user-provided thresholds string/JSON into a per-class list."""
    thresholds = [1.1] + [0.5] * max(0, num_classes - 1)
    if not raw_value:
        return thresholds

    parsed = None
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError:
        tokens = [tok.strip() for tok in raw_value.split(',') if tok.strip()]
        parsed = [float(tok) for tok in tokens]

    if isinstance(parsed, dict):
        for key, value in parsed.items():
            try:
                idx = int(key)
            except ValueError:
                continue
            if 1 <= idx < num_classes:
                thresholds[idx] = float(value)
    elif isinstance(parsed, (list, tuple)):
        for offset, value in enumerate(parsed, start=1):
            if offset >= num_classes:
                break
            thresholds[offset] = float(value)

    for cls_idx in range(1, num_classes):
        thresholds[cls_idx] = float(min(max(thresholds[cls_idx], 0.0), 1.0))

    return thresholds


def resolve_priority(num_classes):
    """Return class check order for thresholding (exclude background class 0)."""
    if num_classes >= 4:
        preferred = [2, 3, 1]
        tail = [c for c in range(1, num_classes) if c not in preferred]
        return [c for c in preferred if c < num_classes] + tail
    return list(range(num_classes - 1, 0, -1))


def sync_rpp_feature_dict(cfg, config_path):
    dict_rel_path = cfg.get('rpp_feat2idx_path')
    if not dict_rel_path:
        return cfg
    config_dir = os.path.dirname(os.path.abspath(config_path))
    dict_path = dict_rel_path if os.path.isabs(dict_rel_path) else os.path.normpath(os.path.join(config_dir, dict_rel_path))
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"rpp_feat2idx_path not found: {dict_path}")
    with open(dict_path, 'rb') as f:
        vocab = pickle.load(f)
    dims = {k: len(v) for k, v in vocab.items()}
    feature_dict = cfg.setdefault('rpp_feature_dict', {})
    selected = cfg.get('rpp_feature_selected', []) or []
    missing = [feat for feat in selected if feat not in dims and feat != 'global_pos']
    if missing:
        raise KeyError(f"Features {missing} missing from {dict_path}")
    for feat in selected:
        if feat in dims:
            feature_dict[feat] = dims[feat]
    if 'cadence_tag' in cfg.get('rpp_feature_all', []) and 'cadence_tag' in dims:
        feature_dict['cadence_tag'] = dims['cadence_tag']
    return cfg


def apply_threshold_decision(probs, thresholds, priority):
    """Deterministic edge class selection via per-class probability thresholds."""
    if probs.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=probs.device)
    pred = torch.zeros(probs.shape[0], dtype=torch.long, device=probs.device)
    for cls_idx in priority:
        cls_thr = thresholds[cls_idx]
        mask = probs[:, cls_idx] >= cls_thr
        if mask.any():
            pred = torch.where(mask, torch.full_like(pred, cls_idx), pred)
    return pred


def sample_with_temperature(probs, temperature):
    """Sample edge classes after applying temperature to logits."""
    if probs.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=probs.device)
        return empty, torch.zeros_like(probs)
    safe_temp = max(temperature, 1e-4)
    scaled_logits = torch.log(probs.clamp_min(1e-9)) / safe_temp
    scaled_probs = torch.softmax(scaled_logits, dim=-1)
    samples = torch.multinomial(scaled_probs, num_samples=1).squeeze(-1)
    return samples, scaled_probs


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        logits: [Batch, VocabSize]
    """
    original_device = logits.device
    logits = logits.float()
    
    # 1. Top-K constraint
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))
        if top_k < logits.size(-1):
            topk_vals, _ = torch.topk(logits, top_k, dim=-1)
            threshold_val = topk_vals[..., -1, None]
            indices_to_remove = logits < threshold_val
            logits = logits.masked_fill(indices_to_remove, filter_value)

    # 2. Top-P (Nucleus) constraint
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Construct mask in the ORIGINAL index space
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        
        
        if sorted_indices.dim() != indices_to_remove.dim():
             # Mismatch in dimensions
             raise ValueError(f"Shape mismatch: {sorted_indices.shape} vs {indices_to_remove.shape}")
        
        
        dim_to_scatter = logits.dim() - 1
        
        try:
             indices_to_remove.scatter_(dim_to_scatter, sorted_indices, sorted_indices_to_remove)
        except Exception as e:
             # Fallback or debug print
             print(f"Scatter failed with shapes: dest {indices_to_remove.shape}, src {sorted_indices_to_remove.shape}, index {sorted_indices.shape}, dim {dim_to_scatter}")
             raise e

        logits = logits.masked_fill(indices_to_remove, filter_value)
        
    return logits


class EquivalenceTracker:
    """GPU-friendly equivalence tracker implemented via label propagation."""
    def __init__(self, max_nodes, device):
        self.device = device
        self.max_nodes = max_nodes
        self.labels = torch.arange(max_nodes, device=device, dtype=torch.long)
        self.active = torch.zeros(max_nodes, dtype=torch.bool, device=device)

    def reset(self):
        self.labels = torch.arange(self.max_nodes, device=self.device, dtype=torch.long)
        self.active.zero_()

    def activate_prefix(self, count):
        if count <= 0:
            return
        self.active[:count] = True
        self.labels[:count] = torch.arange(count, device=self.device, dtype=torch.long)

    def activate_node(self, idx):
        self.active[idx] = True
        self.labels[idx] = idx

    def union(self, left_idx, right_idx):
        label_left = self.labels[left_idx]
        label_right = self.labels[right_idx]
        mask = self.labels == label_right
        if mask.any():
            self.labels = torch.where(mask, label_left, self.labels)

    def component_mask(self, anchor_idx, limit):
        if limit <= 0:
            return torch.zeros(limit, dtype=torch.bool, device=self.device)
        target = self.labels[anchor_idx]
        mask = (self.labels[:limit] == target) & self.active[:limit]
        return mask


class FeatureInferenceDataset(torch.utils.data.Dataset):
    def __init__(self, config, dataroot):
        with open(dataroot, 'rb') as f:
            self.data_list = pickle.load(f)
        self.config = config

    def __getitem__(self, index):
        data = self.data_list[index]
        # Tensorize numpy arrays if present
        
        # rpp_feat is long
        if isinstance(data.get('rpp_feat'), np.ndarray):
            data['rpp_feat'] = torch.from_numpy(data['rpp_feat']).long()
        
       
        if isinstance(data.get('condition'), np.ndarray):
            data['condition'] = torch.from_numpy(data['condition'])
            if data['condition'].dtype in [torch.int32, torch.int16]:
                data['condition'] = data['condition'].long()

        if isinstance(data.get('rpp_mask'), np.ndarray):
             data['rpp_mask'] = torch.from_numpy(data['rpp_mask']).float()

        if isinstance(data.get('note_feat'), np.ndarray):
             data['note_feat'] = torch.from_numpy(data['note_feat']).long()

        return data

    def __len__(self):
        return len(self.data_list)



def apply_forced_classes(final_classes, required_rp, required_mc, forced_indices):
    if final_classes.numel() == 0:
        return final_classes
    forced = torch.full_like(final_classes, fill_value=-1)
    both_mask = required_rp & required_mc
    if forced_indices['both'] is not None:
        forced = torch.where(
            both_mask,
            final_classes.new_full(final_classes.shape, forced_indices['both']),
            forced
        )
    rp_only_mask = required_rp & ~required_mc
    rp_target = forced_indices['rp'] if forced_indices['rp'] is not None else forced_indices['both']
    if rp_target is not None:
        forced = torch.where(
            rp_only_mask,
            final_classes.new_full(final_classes.shape, rp_target),
            forced
        )
    mc_only_mask = required_mc & ~required_rp
    mc_target = forced_indices['mc'] if forced_indices['mc'] is not None else forced_indices['both']
    if mc_target is not None:
        forced = torch.where(
            mc_only_mask,
            final_classes.new_full(final_classes.shape, mc_target),
            forced
        )
    enforce_mask = forced >= 0
    return torch.where(enforce_mask, forced, final_classes)


def main():
    
    # Constrained Decoding
    ENABLE_ANTI_ROLLBACK = True 
    ENABLE_MC_RP_CONSISTENCY = True
    ENABLE_NON_OVERLAP = True 

    # Arg
    if args.expdir == "":
        print("Please specify RecordDirectory!")
        exit()
    exp_dir = args.expdir
    if args.input == "auto":
        input_pkl = os.path.join(exp_dir,'pkl','input.pkl')
    else:
        input_pkl = os.path.join(exp_dir,'pkl',args.input)
    if args.output == "auto":
        output_pkl = os.path.join(exp_dir,'pkl','output.pkl')
    else:
        output_pkl = os.path.join(exp_dir,'pkl',args.output)

    # Default Container
    output_s2g = []

    # Config
    CONFIG_PATH = '../config/config.yaml'
    CONFIG_PATH_ABS = os.path.abspath(CONFIG_PATH)
    with open(CONFIG_PATH_ABS, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config_f = sync_rpp_feature_dict(config, CONFIG_PATH_ABS)
    

    # Device
    if config['use_gpu'] and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(config['gpuID']) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")



    # Data

    inference_dataset = FeatureInferenceDataset(config_f, input_pkl)
    
    inference_dataloader = DataLoader(dataset=inference_dataset, batch_size=1, shuffle=False, drop_last=True)

    # Model

    if args.mode == 'feature' or args.mode == 'all':
        model_f = RPPTransformer(cfg=config_f).to(device)
        model_f.load_state_dict(torch.load(config['best_model_path_f'], map_location=device))
        
        # [Fix] Handle mismatch between checkpoint start_token_vector and current config
        if len(model_f.rpp_feature_selected) != model_f.start_token_vector.shape[0]:
            print(f"[Warning] Mismatch in start_token_vector size: Checkpoint {model_f.start_token_vector.shape[0]} vs Config {len(model_f.rpp_feature_selected)}. Re-initializing start_token_vector.")
            start_tokens = []
            for feat_name in model_f.rpp_feature_selected:
                if feat_name in model_f.rpp_feature_dim_dict:
                    vocab = int(model_f.rpp_feature_dim_dict[feat_name])
                else:
                    print(f"Warning: Feature {feat_name} not found in dim_dict, using default size 1.")
                    vocab = 1
                start_tokens.append(max(vocab - 1, 0))
            model_f.register_buffer('start_token_vector', torch.tensor(start_tokens, dtype=torch.long, device=device))
            
        model_f.eval()

        rp_note_counts = None
        try:
           
            dict_path = config_f.get('rpp_feat2idx_path', 'DataProcess/feat2idx_rpp.pkl')
            with open(dict_path, 'rb') as f_map:
                map_data = pickle.load(f_map)
            
            if 'rhythm_pattern' in map_data:
                rp_dict = map_data['rhythm_pattern']
                max_idx = max(rp_dict.values())

                rp_note_counts_cpu = torch.zeros(max_idx + 1, dtype=torch.long)
                for k, idx in rp_dict.items():
                    if k in ('<pad>', '<start>'):
                        continue
                    try:
                        pattern = ast.literal_eval(k)
                        rp_note_counts_cpu[idx] = len(pattern) if isinstance(pattern, (tuple, list)) else 1
                    except:
                        rp_note_counts_cpu[idx] = 1
                rp_note_counts = rp_note_counts_cpu.to(device)
                
        except Exception as e:
            print(f"[Consistency][WARN] 无法载入数量映射表: {e}")
        mc_note_counts = None
        try:
            if 'melody_contour' in map_data:
                mc_dict = map_data['melody_contour']
                max_mc_idx = max(mc_dict.values())
                mc_note_counts_cpu = torch.zeros(max_mc_idx + 1, dtype=torch.long)
                for k, idx in mc_dict.items():
                    if k in ('<pad>', '<start>'):
                        continue
                    try:
                        val = int(k)
                        if val == 12: count = 1
                        elif 0 <= val <= 2: count = 2
                        elif 3 <= val <= 11: count = 3
                        else: count = 1 # Fallback
                        mc_note_counts_cpu[idx] = count
                    except:
                        mc_note_counts_cpu[idx] = 1
                mc_note_counts = mc_note_counts_cpu.to(device)
        except Exception as e:
            print(f"[Consistency][WARN] 无法载入旋律数量映射表: {e}")
            
    
    # generate
    num_song = 0

    if args.mode == 'feature':
        for batch in tqdm(inference_dataloader):

            # num_song check
            if num_song >= config['num_song']:
                break
            
            tgt = batch['rpp_feat'].to(device)
            rpp_mask = batch['rpp_mask'].to(device)

            if tgt.shape[-1] == 6 and len(model_f.rpp_feature_selected) == 5 and 'global_pos' in model_f.rpp_feature_selected:
                bar = tgt[..., 0]
                pos = tgt[..., 1]
                

                mask = (bar > 0) & (pos > 0)
                global_pos = torch.zeros_like(bar)
                b_idx = torch.clamp(bar - 1, min=0)
                p_idx = torch.clamp(pos - 1, min=0)
                global_pos[mask] = b_idx[mask] * 16 + p_idx[mask] + 1
                
                tgt = torch.cat([global_pos.unsqueeze(-1), tgt[..., 2:]], dim=-1)

            # Inference[Batch]
            if config.get('input_lenth', 0) > 0:
                tgt = tgt[:,:config.get('input_lenth', 0),:]
            else:
                # Initialize with correct Start Token (<start> usually at index dim-1)
                B = tgt.shape[0]
                F_dim = tgt.shape[2]
                
                # Retrieve dimensions from model config
                feat_dims = [model_f.rpp_feature_dim_dict[f] for f in model_f.rpp_feature_selected]
                # Start token is typically the last index (dim-1) for each feature
                start_indices = [d - 1 for d in feat_dims]
                
                start_token_vec = torch.tensor(start_indices, dtype=tgt.dtype, device=tgt.device).view(1, 1, -1)
                tgt = start_token_vec.expand(B, -1, -1)

            # Pre-calculate feature indices for splitting
            feature_size = [model_f.rpp_feature_dim_dict[f] for f in model_f.rpp_feature_selected]
            div_index = [0] + [sum(feature_size[:i+1]) for i in range(len(feature_size))]
            feat_ranges = {f: (div_index[i], div_index[i+1]) for i, f in enumerate(model_f.rpp_feature_selected)}
            gp_range = feat_ranges.get('global_pos')
            
            # Forward
            idx_gp = model_f.feature_name_to_idx['global_pos']
            idx_rp = model_f.feature_name_to_idx.get('rhythm_pattern')
            idx_mc = model_f.feature_name_to_idx.get('melody_contour')
            idx_dur = model_f.feature_name_to_idx.get('duration')
            
            # Anti-Rollback & Look-ahead Context
            locked_next_pos = None

            # Early stop flag to allow breaking out when position goes out of embedding bounds
            early_stop = False
            
            
            end_len = config['output_lenth']
            
            for i in range(config.get('input_lenth', 0), end_len):
                
                # --- Step 1: Predict Position i (or use Locked) ---
                logits_all = model_f(tgt=tgt, tgt_key_mask=None)
                logits_last = logits_all[:, -1:, :] # [B, 1, D_model]
                hidden_i = model_f._cached_hidden[:, -1:, :] 

                if locked_next_pos is not None:
                    next_pos_idx = locked_next_pos
                else:
                    # Extract Global Pos Logits
                    if gp_range is None: gp_start, gp_end = 0, 0 
                    else: gp_start, gp_end = gp_range
                    gp_logits = logits_last[:, :, gp_start:gp_end]

                    # Anti-Rollback Masking
                    prev_pos = tgt[:, -1, idx_gp].long().unsqueeze(1).unsqueeze(2) # [B, 1, 1]
                    vocab_size = gp_logits.shape[-1]
                    pos_indices = torch.arange(vocab_size, device=device).view(1, 1, -1)
                    
                    if ENABLE_ANTI_ROLLBACK:
                        is_start_token = (prev_pos == (vocab_size - 1))
                        effective_prev_pos = torch.where(is_start_token, torch.zeros_like(prev_pos), prev_pos)

                        rollback_mask = pos_indices <= effective_prev_pos
                        # Also explicitly mask the Start Token itself in output to prevent re-generating SOS
                        rollback_mask[..., -1] = True
                        # Also mask Pad (index 0) if not already masked by <= logic (though usually 0 <= something)
                        rollback_mask[..., 0] = True
                        
                        gp_logits = gp_logits.masked_fill(rollback_mask, -1e9)
                    else:
                        basic_mask = torch.zeros_like(pos_indices, dtype=torch.bool)
                        basic_mask[..., -1] = True
                        basic_mask[..., 0] = True
                        gp_logits = gp_logits.masked_fill(basic_mask, -1e9)
                    
        
                    temp = 0.95
                    

                    scaled_logits = gp_logits / temp
                    # Using Nucleus Sampling (Top-P) = 0.9 + Top-K = 20 for position
                    filtered_logits = top_k_top_p_filtering(scaled_logits, top_k=20, top_p=0.9)
                    gp_prob = F.softmax(filtered_logits, dim=-1)
                    next_pos_idx = torch.multinomial(gp_prob.squeeze(1), 1) # [B, 1]

                
                MAX_VALID_POS = 2048

                if (next_pos_idx.view(-1) > MAX_VALID_POS).any():
                    maxpos = int(next_pos_idx.max().item())
                    try:
                        names = batch['name'] if 'name' in batch else None
                        if names is not None:
                            print(f"Song finished early at pos {maxpos} for samples: {names}")
                        else:
                            print(f"Song finished early at pos {maxpos}")
                    except Exception:
                        print(f"Song finished early at pos {maxpos}")
                    early_stop = True
                    break

                
                clip_next_pos = next_pos_idx.clamp(0, model_f.future_pos_embedding.num_embeddings - 1)
                future_pos_emb = model_f.future_pos_embedding(clip_next_pos) 
                
                cur_pos_idx_prev = tgt[:, -1, idx_gp].long().unsqueeze(1)
                delta_input = (next_pos_idx - cur_pos_idx_prev).clamp(0, model_f.delta_embedding.num_embeddings - 1)
                delta_emb = model_f.delta_embedding(delta_input)
                
                head_context = torch.cat([hidden_i, future_pos_emb, delta_emb], dim=-1)
                
                generated_details = {}
                feature_order = ['rhythm_pattern', 'melody_contour']
                

                idx_rp = model_f.feature_name_to_idx['rhythm_pattern']
                idx_mc = model_f.feature_name_to_idx['melody_contour']

                for feat in feature_order:
                    if feat not in model_f.rpp_feature_selected:
                        continue
                    
                    f_idx = model_f.feature_name_to_idx[feat]
                    batch_vals = []

                    for b in range(tgt.shape[0]):
                        inherited_val = None
                        target_note_count = None 
                        

                        if inherited_val is not None:
                            batch_vals.append(inherited_val.view(1, 1))
                        else:
                            
                            head = model_f.feature_heads[feat]
                            feat_logits = head(head_context[b:b+1])

                            block_mask = torch.zeros_like(feat_logits, dtype=torch.bool)
                            vocab_size = feat_logits.shape[-1]
                            v_indices = torch.arange(vocab_size, device=device).view(1, 1, -1)


                            if feat == 'melody_contour' and ENABLE_MC_RP_CONSISTENCY and rp_note_counts is not None and 'rhythm_pattern' in generated_details:
                                curr_rp_idx = generated_details['rhythm_pattern'][b]
                                safe_rp_idx = curr_rp_idx.long().clamp(0, len(rp_note_counts) - 1)
                                count = rp_note_counts[safe_rp_idx]
                                
                                
                                if count == 1: 
                                    # Allow only Index 13
                                    block_mask |= (v_indices != 13)
                                elif count == 2: 
                                    # Allow Indices 1, 2, 3
                                    block_mask |= (v_indices > 3) | (v_indices < 1)
                                elif count == 3: 
                                    # Allow Indices 4..12
                                    block_mask |= (v_indices < 4) | (v_indices > 12)

                           
                            elif feat == 'rhythm_pattern' and ENABLE_MC_RP_CONSISTENCY and target_note_count is not None and rp_note_counts is not None:
                                valid_indices_mask = (rp_note_counts == target_note_count).unsqueeze(0).unsqueeze(0)
                                valid_len = min(vocab_size, valid_indices_mask.shape[-1])
                                block_mask[:, :, :valid_len] = ~valid_indices_mask[:, :, :valid_len]
                                block_mask[:, :, 0] = True 


                            if block_mask.any():
                                feat_logits = feat_logits.masked_fill(block_mask, -float('inf'))
                            
                            sample_params = {
                                'rhythm_pattern': {'k': 10, 'p': 1.0, 't': 1.5},
                                'melody_contour': {'k': 10, 'p': 1.0, 't': 1.5},
                                'duration': {'k': 8, 'p': 1.0, 't': 1.0}
                            }

                            params = sample_params.get(feat, {'k': 10, 'p': 0.9, 't': 1.0}) 
                            
                            t_scaled_logits = feat_logits / params['t']
                            

                            filtered_logits = top_k_top_p_filtering(
                                t_scaled_logits.squeeze(1), 
                                top_k=params['k'], 
                                top_p=params['p']
                            ).unsqueeze(1) # Restore dim for softmax

                            
                            if block_mask.any():
                               
                                filtered_logits = filtered_logits.masked_fill(block_mask, -float('inf'))
                            
                            feat_prob = F.softmax(filtered_logits, dim=-1)
                            # filtered_logits: [1, 1, V] -> squeeze(1) -> [1, V]
                            val = torch.multinomial(feat_prob.squeeze(1), 1)
                            batch_vals.append(val)
                    
                    generated_details[feat] = torch.cat(batch_vals, dim=0)
                # --- Step 4: Full Context Look-ahead for Pos i+1 ---
                # Build Node i (with Dur=0 placeholder)
                temp_node = torch.zeros(tgt.shape[0], 1, tgt.shape[2], device=device, dtype=tgt.dtype)
                temp_node[:, 0, idx_gp] = next_pos_idx[:, 0]
                for f, v in generated_details.items():
                    f_idx = model_f.feature_name_to_idx[f]
                    temp_node[:, 0, f_idx] = v[:, 0]
                    
                tgt_lookahead = torch.cat([tgt, temp_node], dim=1)
                
                # Predict Pos i+1
                logits_look = model_f(tgt=tgt_lookahead, tgt_key_mask=None)
                if gp_range is None: gp_start, gp_end = 0, 0
                else: gp_start, gp_end = gp_range
                
                logits_plus1 = logits_look[:, -1:, gp_start:gp_end]
                
                # Anti-Rollback (Must be > pos i)
                cur_pos_expanded = next_pos_idx.unsqueeze(1)
                
                if ENABLE_ANTI_ROLLBACK:
                    p_indices = torch.arange(logits_plus1.shape[-1], device=device).view(1, 1, -1)
                    # 1. Ban Rollback
                    mask_next = p_indices <= cur_pos_expanded
                    
                    # 2. [New] Ban Large Jumps (Max 32 units = 2 bars) to prevent skipping content
                    MAX_JUMP = 48
                    mask_next |= (p_indices > (cur_pos_expanded + MAX_JUMP))

                    # Also mask SOS and PAD
                    mask_next[..., -1] = True 
                    mask_next[..., 0] = True
                    logits_plus1 = logits_plus1.masked_fill(mask_next, -1e9)
                else:
                    # Basic masking 
                    mask_next = torch.zeros(logits_plus1.shape[-1], dtype=torch.bool, device=device).view(1, 1, -1)
                    mask_next[..., -1] = True 
                    mask_next[..., 0] = True
                    logits_plus1 = logits_plus1.masked_fill(mask_next, -1e9)
                
                
                prob_plus1_scaled = logits_plus1 / 0.9
                prob_plus1_filtered = top_k_top_p_filtering(prob_plus1_scaled.squeeze(1), top_k=10, top_p=0.85).unsqueeze(1)
                
                prob_plus1 = F.softmax(prob_plus1_filtered, dim=-1)
                pos_plus1_idx = torch.multinomial(prob_plus1.squeeze(1), 1)
                locked_next_pos = pos_plus1_idx # Lock for next iter
                
                # --- Step 5: Sample Duration (Constrained) ---
                if 'duration' in model_f.rpp_feature_selected:
                    delta_constraint = (pos_plus1_idx - next_pos_idx).clamp(min=1)
                    
                    head_dur = model_f.feature_heads['duration']
                    dur_logits = head_dur(head_context)
                    
                    d_vocab = dur_logits.shape[-1]
                    d_indices = torch.arange(d_vocab, device=device).view(1, 1, -1)
                    
                    # Mask > delta
                    if ENABLE_NON_OVERLAP:
                        delta_exp = delta_constraint.unsqueeze(1)
                        mask_dur = d_indices > delta_exp
                    else:
                        mask_dur = torch.zeros(d_indices.shape, dtype=torch.bool, device=device)

                    # Also mask 0 (pad)
                    mask_dur[:, :, 0] = True


                    d_scaled = dur_logits / 0.9 # Strict duration
                    
                    d_biased = d_scaled.clone()
                    
                    # Critical Fix: Apply mask AGAIN on biased logits to ensure boosted values don't overcome mask (though -inf should hold)
                    if mask_dur.any():
                         d_biased = d_biased.masked_fill(mask_dur, -float('inf'))

                    # --- SAFETY CHECK: If all logits are -inf, unmask Duration 4 (Least bad short note) ---
                    # This happens if delta constraint forces duration to be small (e.g. 1, 2, 3) but we banned them.
                    max_val = d_biased.max(dim=-1)[0]
                    is_dead = (max_val == -float('inf'))
                    
                    if is_dead.any():
                        valid_len = delta_constraint[is_dead] # e.g. 2 means only durations 1,2 allowed.
                        

                        b_indices = torch.nonzero(is_dead).squeeze(1)
                        target_indices = valid_len.long() # Max valid duration index
                        
                        # Ensure target index is within vocab
                        target_indices = torch.clamp(target_indices, min=1, max=vocab_size-1)
                        
                        # Set logit to 10.0 to force selection
                        d_biased[b_indices, 0, target_indices] = 10.0

                    d_filtered = top_k_top_p_filtering(d_biased.squeeze(1), top_k=5, top_p=0.85).unsqueeze(1)
                    dur_prob = F.softmax(d_filtered, dim=-1)
                    dur_val = torch.multinomial(dur_prob.squeeze(1), 1)
                    generated_details['duration'] = dur_val
                
                # --- Step 6: Finalize Node i & Update Tgt -----
                # Reuse temp_node structure, fill Duration
                if 'duration' in generated_details:
                     temp_node[:, 0, idx_dur] = generated_details['duration'][:, 0]
                
                tgt = torch.cat([tgt, temp_node], dim=1)

            
            if early_stop:
                # do nothing special here; downstream saving uses actual generated length
                pass

            
            rpp_feat_normalize = normalize_generated_rpp_feat(tgt) 
            
            # global_pos is at index 0. (bar-1)*16 + (pos-1) + 1
            g_pos = rpp_feat_normalize[:, :, 0]
            others = rpp_feat_normalize[:, :, 1:]
            
            # Handle SOS (val=0) and invalid logic
            valid_mask = (g_pos > 0)
            val_shifted = torch.clamp(g_pos - 1, min=0)
            
            bars = (val_shifted // 16) + 1
            poss = (val_shifted % 16) + 1
            
    
            bars = torch.where(valid_mask, bars, torch.zeros_like(bars))
            poss = torch.where(valid_mask, poss, torch.zeros_like(poss))
            
            if bars.shape[1] > 0:
                # Override Batch's 0-th timestamp to 129, 17 as requested
                bars[:, 0] = 129
                poss[:, 0] = 17

            rpp_feat_decoded = torch.cat([bars.unsqueeze(-1), poss.unsqueeze(-1), others], dim=-1)

            rp_idx_in_tgt = model_f.feature_name_to_idx.get('rhythm_pattern')
            mc_idx_in_tgt = model_f.feature_name_to_idx.get('melody_contour')

            # save
            for b in range(tgt.shape[0]):
                cur_dict = {}
                cur_dict['name'] = batch['name'][b]
                

                arr_feat = rpp_feat_decoded[b].cpu().numpy().copy()
                

                generated_len = arr_feat.shape[0]
                
                
                real_len = max(0, generated_len - 1)
                
                
                target_real_len = min(config['output_lenth'], real_len)
                
                


                
                try:
                    # bar is the first column
                    bar_col = arr_feat[:, 0]
                    bar_col[bar_col == 130] = 129
                    arr_feat[:, 0] = bar_col
                except Exception:
                    pass
                
               
                total_len = 1 + target_real_len
                cur_dict['rpp_feat'] = arr_feat[:total_len]
                
                cur_dict['condition'] = np.array(batch['condition'][b].cpu())
                cur_dict['note_feat'] = np.array(batch['note_feat'][b].cpu())
                
                
                seq_max = config['seq_max']
                generated_steps = total_len
                cur_dict['rpp_mask'] = np.array([1]*generated_steps + [0]*(seq_max - generated_steps))

                output_s2g.append(cur_dict)

                # record
                num_song += 1


    else:
        pass



    # Save
    with open(output_pkl,'wb') as f:
        pickle.dump(output_s2g,f)


def normalize_generated_rpp_feat(rpp_feat):
    """
    Directly return the generated features without padding.
    The length should match the accumulated 'tgt' length from the autoregressive loop.
    Since rpp_feat is already [Batch, Seq, Feat], we just return it.
    """
    return rpp_feat


if __name__ == '__main__':
    main()



