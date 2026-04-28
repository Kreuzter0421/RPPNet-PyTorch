import random
import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
import itertools
import pickle
from torch.utils.data import IterableDataset, get_worker_info
import glob
import os

# Datasets
class MuGraphDataset(IterableDataset):
    def __init__(self, config, dataroot):
        self.dataroot = dataroot
        self.files = []
        if os.path.isdir(dataroot):
            self.files = sorted(glob.glob(os.path.join(dataroot, '*.pkl')))
        else:
            self.files = [dataroot]
        

        self.rps_feature_all = config['rps_feature_all']
        self.rps_feature_selected = config['rps_feature_selected']

    def __iter__(self):
        worker_info = get_worker_info()
        # Sharding logic for multiple workers
        if worker_info is None:  # Single-process data loading
            file_iter = self.files
        else:  # Multi-process data loading
            # Split files among workers
            per_worker = int(np.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))
            file_iter = self.files[iter_start:iter_end]
            
    
        file_list = list(file_iter)
        random.shuffle(file_list)
        
        for pkl_path in file_list:
            data_list = []
            with open(pkl_path, 'rb') as f:
                data_list = pickle.load(f)
            
            # Shuffle items within the chunk
            random.shuffle(data_list)
            
            for raw_data in data_list:
                # Process item (same logic as original __getitem__)
                data = self._process_item(raw_data)
                yield data
                
    def _process_item(self, raw_data):
        # Extracted logic from original __getitem__
        data = {"name": None,
                "condition": None,
                "rps_feat": None,
                "rps_feat_gt":None,
                "rps_mask":None,
                "num_nodes": None,
                "n_in_sequences": None,
                "n_in_stride": None,
                "n_in_starts": None,
                "node_positions": None}
        # tensorlize
        for k,v in raw_data.items():
            if k in data.keys():
                if k == 'name':
                    data[k] = v
                elif k in ('num_nodes', 'n_in_sequences', 'n_in_stride', 'n_in_starts'):
                    data[k] = v
                else:
                    data[k] = torch.from_numpy(v).long()
        
        # feat select logic
        if data['rps_feat'] is not None:
             data['rps_feat'] = self._process_feats(data['rps_feat'])
             
        if data['rps_feat_gt'] is not None:
             data['rps_feat_gt'] = self._process_feats(data['rps_feat_gt'])
             
        # Metadata logic (extended)
        
        seq_len = 0
        if data['rps_feat'] is not None:
            seq_len = int(data['rps_feat'].shape[0])
        if seq_len > 0:
            data['node_positions'] = torch.arange(seq_len, dtype=torch.long)

        # metadata
        num_nodes = data['num_nodes'] if data['num_nodes'] is not None else raw_data.get('num_nodes')
        if num_nodes is None and data['rps_mask'] is not None:
            num_nodes = int(torch.count_nonzero(data['rps_mask']).item())
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

        return data

    def _process_feats(self, full_feats):
        # Same helper logic as before
        if full_feats is None:
            return None
        
        try:
            bar_idx = self.rps_feature_all.index('bar')
            pos_idx = self.rps_feature_all.index('position')
        except ValueError:
            indices = [self.rps_feature_all.index(f) for f in self.rps_feature_selected if f in self.rps_feature_all]
            return full_feats[:, indices]

        seq_len = full_feats.shape[0]
        selected_cols = []
        
        for feat_name in self.rps_feature_selected:
            if feat_name == 'global_pos':
                bars = full_feats[:, bar_idx]
                poss = full_feats[:, pos_idx]
                valid_mask = (bars > 0) & (poss > 0)
                b_safe = torch.clamp(bars, min=1)
                p_safe = torch.clamp(poss, min=1)
                calc_vals = (b_safe - 1) * 16 + (p_safe - 1) + 1
                global_vals = torch.where(valid_mask, calc_vals, torch.zeros_like(calc_vals))
                selected_cols.append(global_vals)
            elif feat_name in self.rps_feature_all:
                f_idx = self.rps_feature_all.index(feat_name)
                selected_cols.append(full_feats[:, f_idx])
            else:
                selected_cols.append(torch.zeros(seq_len, dtype=torch.long))
        
        if not selected_cols:
            return None
        return torch.stack(selected_cols, dim=1)
    
    def __len__(self):
        total = 0
        for fpath in self.files:
            try:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                    total += len(data)
            except:
                pass
        return total
    
