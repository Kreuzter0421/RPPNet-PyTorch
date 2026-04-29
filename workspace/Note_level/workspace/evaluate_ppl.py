import sys
import os


import argparse
import yaml
import torch
import math
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import NoteTransformer, MidiDataset
from train import collate_noteLevel, build_feature_loss_masks

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate NoteTransformer Global Perplexity (PPL)")
    parser.add_argument("--config", type=str, default="../config/config.yaml",
                        help="Path to the model config.yaml file.")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model checkpoint (.pth).")
    parser.add_argument("--test-dir", type=str, required=True,
                        help="Path to the folder or .pkl file of the Test Set.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Eval batch size.")
    parser.add_argument("--pad-idx", type=int, default=0,
                        help="Padding index to ignore in the loss calculation.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.full_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"Loading Test Dataset from: {args.test_dir}")
    test_dataset = MidiDataset(config=cfg, dataroot=args.test_dir)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        collate_fn=collate_noteLevel, 
        num_workers=4,
        drop_last=False 
    )
    
    print(f"Loading model from: {args.model_path}")
    model = NoteTransformer(cfg=cfg).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=args.pad_idx, reduction='sum')
    
    total_nll_loss = 0.0
    total_valid_tokens = 0
    use_graph_modules = bool(cfg.get('use_graph_modules', True))

    feature_size = [cfg['note_feature_dim_dict'][f] for f in cfg['note_feature_selected']]
    div_index = [0] + [sum(feature_size[:i+1]) for i in range(len(feature_size))]
    
    print("Starting Teacher Forcing Evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating PPL"):
            V = batch['rpp_feat'].to(device)
           
            tgt = batch['note_feat'].to(device)
            
            tgt_gt = batch['note_feat_gt'].to(device)
            tgt_mask = batch['note_mask'].to(device)
            

                
            first_note_mask = batch.get('first_note_mask')
            feature_masks = None
            if first_note_mask is not None:
                feature_masks = build_feature_loss_masks(cfg['note_feature_selected'], first_note_mask.to(device))
            
            
            out = model(V=V, tgt=tgt, tgt_mask=tgt_mask)
            

            predicts = [out[..., i:j] for i, j in zip(div_index[:-1], div_index[1:])]
           
            gts = [tgt_gt[..., i] for i in range(tgt_gt.shape[2])]
            

            for idx, (pre, y) in enumerate(zip(predicts, gts)):
                # pre shape: [Batch, SeqLen, VocabSize]
                # y shape: [Batch, SeqLen]
                logits = pre.view(-1, pre.shape[2])
                targets = y.view(-1)
                

                mask = feature_masks[idx] if feature_masks is not None else None
                if mask is not None:
                    mask_flat = mask.view(-1)
                    if mask_flat.dtype != torch.bool:
                        mask_flat = mask_flat > 0
                    if not torch.any(mask_flat):
                        continue
                    logits = logits[mask_flat]
                    targets = targets[mask_flat]
                
                if logits.shape[0] == 0:
                    continue
                    

                batch_feature_loss_sum = loss_fn(logits, targets)
                
               
                valid_count = (targets != args.pad_idx).sum().item()
                
                total_nll_loss += batch_feature_loss_sum.item()
                total_valid_tokens += valid_count

    if total_valid_tokens == 0:
        print("Error: No valid tokens found during evaluation. Cannot compute PPL.")
        return
        

    avg_nll_loss = total_nll_loss / total_valid_tokens

    global_ppl = math.exp(avg_nll_loss)
    
    print("\n" + "=" * 50)
    print("                EVALUATION RESULTS                ")
    print("=" * 50)
    print(f"Total Cumulative NLL Loss : {total_nll_loss:.4f}")
    print(f"Total Valid Tokens        : {total_valid_tokens}")
    print(f"Global Average Loss/Token : {avg_nll_loss:.4f}")
    print(f"Global Perplexity (PPL)   : {global_ppl:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
