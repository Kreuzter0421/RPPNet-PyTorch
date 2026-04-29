import pickle
import pprint
import yaml
import torch
from model.model import MidiDataset_Inference as MidiDataset
from torch.utils.data import DataLoader
from model.model import NoteTransformer

import os
from tqdm import tqdm
import numpy as np
import miditoolkit
import argparse

# Parser
parser = argparse.ArgumentParser(description="Note inference")
parser.add_argument("-e","--expdir", type=str, default="", help="specify RecordDirectory")
parser.add_argument("-i","--input", type=str, default="auto", help="specify data path")
parser.add_argument("-o","--output", type=str, default="auto", help="specify save path")
args = parser.parse_args()


def write_midi(cfg,data,save_path,name):
    # Input : data (seq,feature)
    # Output : midi file

    # init
    midi_obj = miditoolkit.MidiFile()
    melody_track = miditoolkit.Instrument(program=0,name='melody')
    tick_resolution = cfg['tick_resolution']
    bar_max = cfg['bar_max']
    GRID = np.arange(0,bar_max*1920+1,tick_resolution) 


    debug_count = 0
    
    for note_ft in data:
        bar, pos, dur, pitch, vel = 0, 0, 0, 0, 0
        if note_ft[0] == 1 and torch.all(note_ft[1:] == 0):      
            continue
        elif torch.all(note_ft == 0):  # pad
            continue 
        else:                               # note
            features = [f  for f in cfg['note_feature_selected']]
            for ft,idx in zip(features,list(note_ft)):
                if ft == 'bar':
                    bar = int(idx - 1)
                elif ft == 'position':
                    pos = GRID[idx-1]
                elif ft == 'duration':
                    
                    raw_dur = GRID[idx-1]
                    dur = raw_dur if raw_dur > 0 else 120 # Fallback to 1 unit if 0
                elif ft == 'pitch':
                    pitch = int(idx-1)
                elif ft == 'velocity':
                    vel = int(idx-1)
        
        # check
        if bar<0 or pos<0 or dur<=0 or pitch<0 or vel<0 or dur>=3840:
            continue
        

        melody_track.notes.append(miditoolkit.Note(
            start=int(bar*1920 + pos),
            end=bar*1920 + pos + dur,
            pitch=pitch,
            velocity=64
        ))

    
    midi_obj.tempo_changes = [miditoolkit.TempoChange(tempo=100,time=0)]


    midi_obj.instruments.append(melody_track)
    midi_obj.dump(save_path)

def inference(cfg, model, batch, device):
    model.eval()    
    V = batch['rpp_feat'].to(device)
    
    
    is_v2 = cfg.get('model_type', 'v1') == 'v2'

    
    if not is_v2:
        tgt = batch['note_feat'].to(device) 
        if cfg['input_lenth'] > 0:
            if tgt.shape[1] > cfg['input_lenth']:
                tgt = tgt[:,:cfg['input_lenth'],:]
                
            
                seq = tgt[0]
                sort_keys = seq[:, 0] * 100000000 + seq[:, 1] * 10000 + seq[:, 3]
                sorted_indices = torch.argsort(sort_keys)
                tgt[0] = seq[sorted_indices]
                
        else:

            B = tgt.shape[0]
            FeatDim = tgt.shape[2]
            tgt = torch.zeros((B, 1, FeatDim), dtype=torch.long, device=device)
            tgt[:, 0, 0] = 1 
            
        max_len = cfg['inference_lenth']
        start_len = tgt.shape[1]
        for i in range(start_len, max_len):
            out = model(V=V, tgt=tgt, tgt_mask=None, memory_mask=None)
            next_note = out[:,-1:,:]
            
           
            rpp_force = cfg.get('constraints', {}).get('force_rpp_alignment', False)
            
            # Prepare RPP Ticks for checking
            rpp_ticks_cache = None
            if rpp_force:
                v_dat = V[0].float()
                valid_mask = (v_dat[:, 0] > 0)
                if valid_mask.any():
                    rpp_bars = v_dat[valid_mask, 0] - 1
                    rpp_poses = (v_dat[valid_mask, 1] - 1) * 120
                    rpp_ticks_cache = rpp_bars * 1920 + rpp_poses

            # Sampling Loop for RPP Alignment
            aligned_sample_found = False
            rpp_retries = 0
            
            while not aligned_sample_found and rpp_retries < 20:
                next_note = model.predict_transform(out[:,-1:,:]) # Sample
                if next_note.dim() == 2:
                     next_note = next_note.unsqueeze(1)
                
                if not rpp_force or rpp_ticks_cache is None:
                    aligned_sample_found = True
                    break

                # Check Alignment
                n_bar = next_note[0,0,0].item() - 1
                n_pos = (next_note[0,0,1].item() - 1) * 120
                
                if n_bar < 0: # Check SOS/EOS if needed
                     aligned_sample_found = True
                     break

                c_tick = n_bar * 1920 + n_pos
                
                # Check distance
                # We use 30 ticks tolerance (approx 32nd note)
                dist = torch.min(torch.abs(rpp_ticks_cache - c_tick))
                
                if dist <= 30: 
                    aligned_sample_found = True
                else:
                    rpp_retries += 1
                    
            
            pass

            # Note Check & Filter (Apply constraints during generation)
            if i > 0:
                is_valid = False
                max_retries = 20  
                current_retry = 0
                
                
                prev_tensor = tgt[0, -1].cpu().numpy()
                
                # Loop until valid note is found or max retries reached
                # The first 'next_note' is already generated above
                while not is_valid and current_retry < max_retries:
                    curr_tensor = next_note[0, 0].cpu().numpy()
                    
                    
                    if curr_tensor[0] == 0: 
                        is_valid = True
                        break
                    
                    
                    constraints = cfg.get('constraints', {})


                    # note_check standard rules
                    # Pass cfg to note_check to use dynamic constraints
                    if note_check(prev_tensor, curr_tensor, constraints):
                         out_reshaped = out[:,-1:,:]
                         next_note = model.predict_transform(out_reshaped)
                         if next_note.dim() == 2:
                             next_note = next_note.unsqueeze(1)
                         current_retry += 1
                    else:
                        is_valid = True
                        
                        # Handle overlap truncation (Monophony enforcement by cutting previous note)
                        p_bar_val = int(prev_tensor[0] - 1)
                        p_pos_val = int(prev_tensor[1] - 1) * 120
                        p_dur_val = int(prev_tensor[2] - 1) * 120
                        
                        c_bar_val = int(curr_tensor[0] - 1)
                        c_pos_val = int(curr_tensor[1] - 1) * 120
                        c_dur_val = int(curr_tensor[2] - 1) * 120

                        # [Constraint 3] Disallow Cross-Measure Ties (Force Truncation at Bar End)
                        if constraints.get('prevent_cross_measure', False):
                            if c_pos_val + c_dur_val > 1920:
                                max_allowed_dur = 1920 - c_pos_val
                                if max_allowed_dur > 0:
                                    new_dur_idx = max(1, int(max_allowed_dur / 120) + 1)
                                    next_note[0, 0, 2] = new_dur_idx
                                    c_dur_val = (new_dur_idx - 1) * 120

                        # Truncate Only if overlapping
                        if constraints.get('prevent_overlap', False):
                            p_start_tick = p_bar_val * 1920 + p_pos_val
                            p_end_tick = p_start_tick + p_dur_val
                            c_start_tick = c_bar_val * 1920 + c_pos_val

                            if c_start_tick < p_end_tick:
                                if c_start_tick > p_start_tick: 
                                    new_dur_tick = c_start_tick - p_start_tick
                                    new_dur_idx = max(1, int(new_dur_tick / 120) + 1)
                                    tgt[0, -1, 2] = new_dur_idx

            tgt = torch.cat([tgt, next_note], dim=1)
        
        # [Post-Processing: Bar Alignment & Truncation]
        constraints = cfg.get('constraints', {})
        
        # 1. Shift: Find the first real note and shift everything so it starts at Bar 0 (Index 1).
        if constraints.get('force_bar_0_start', False) and tgt.shape[1] > 1:
            start_idx = 1
            while start_idx < tgt.shape[1] and tgt[0, start_idx, 0].item() == 0:
                start_idx += 1
            
            if start_idx < tgt.shape[1]:
                 first_bar_feature = tgt[0, start_idx, 0].item()
                 shift_amt = first_bar_feature - 1
                 if shift_amt > 0:
                    #  print(f"  [Shift] Shifting piece by -{shift_amt} bars to start at 0.")
                     tgt[0, start_idx:, 0] = torch.clamp(tgt[0, start_idx:, 0] - shift_amt, min=1)
            
        # 2. Truncate: Only keep notes within first 32 bars.
        if constraints.get('truncate_to_32_bars', False):
            mask = (tgt[0, :, 0] <= 32) & (tgt[0, :, 0] > 0)
            mask[0] = True # Keep SOS
            tgt_filtered = tgt[0][mask].unsqueeze(0)
            return tgt_filtered
        
        return tgt
    # V2 Logic
    B, N, D_RPP = V.shape
    feature_dims = [cfg['note_feature_dim_dict'][f] for f in cfg['note_feature_selected']]
    F_dim = len(feature_dims)
    
    generated_chunks = [] 
    
    for i in range(N):
        # Construct history
        if i > 0:
            history_stack = torch.stack(generated_chunks, dim=1)
            full_gt = torch.zeros(B, N, 4, F_dim, device=device).long()
            full_gt[:, :i, :, :] = history_stack
        else:
            full_gt = None # or zeros

        # Init current chunk
        current_chunk = torch.zeros(B, 4, F_dim, device=device).long()
        current_chunk[:, 0, :] = 1 # SOS
        
        finished_mask = torch.zeros(B, dtype=torch.bool, device=device)

        for step in range(3): 
            full_tgt = torch.zeros(B, N, 4, F_dim, device=device).long()
            full_tgt[:, i, :, :] = current_chunk
            
            # Forward
            out = model(V=V, tgt=full_tgt, tgt_mask=None, gt_notes=full_gt)
            out = out.view(B, N, 4, -1)
            
            step_out = out[:, i, step, :] 
            next_token = model.predict_transform(step_out) 
            
            # Pad Truncation Logic
            is_pad = (next_token[:, 0] == 0)
            finished_mask = finished_mask | is_pad
            if finished_mask.any():
                next_token[finished_mask] = 0
            
            current_chunk[:, step+1, :] = next_token
            
        # History chunk: Shift left (remove SOS) -> [N1, N2, N3, PAD]
        hist = torch.zeros_like(current_chunk)
        hist[:, :-1, :] = current_chunk[:, 1:, :]
        generated_chunks.append(hist)
        
    output = torch.stack(generated_chunks, dim=1).view(B, -1, F_dim)
    return output

def main():

    # Arg (Directory)
    if args.expdir == "":
        print("Please specify RecordDirectory!")
        exit()
    exp_dir = args.expdir
    raw_pkl = os.path.join(exp_dir, 'pkl', 'output.pkl')
    
    if args.input == 'auto':
        input_pkl = os.path.join(exp_dir, 'pkl', 'output.pkl')
    else:
        input_pkl = os.path.join(exp_dir,'pkl',f'{args.input}')
    if args.output == 'auto':   
        output_pkl = os.path.join(exp_dir, 'pkl', 'output_note.pkl')
    else:
        output_pkl = os.path.join(exp_dir,'pkl',f'args.output')
        
    txt_test = os.path.join(exp_dir, 'pkl', 'output_note_test.txt')
    midi_infer_dir = os.path.join(exp_dir,'midi_inference')

    # Config
    congfig_path = '../config/config.yaml'
    cfg = yaml.full_load(open(congfig_path, 'r'))
    

    # Control the output MIDI file
    cfg['constraints'] = {
        'prevent_cross_measure': True,   
        'prevent_overlap': True,         
        'force_bar_0_start': True,       
        'truncate_to_32_bars': True     
    }


    if os.path.exists(input_pkl):
        print(f"{input_pkl} already exists!")
    
    # Device
    if cfg['usegpu']:
        device = torch.device("cuda:{}".format(cfg['gpuID']) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    # Model
    model = NoteTransformer(cfg=cfg).to(device)

        
    if cfg['usegpu'] and torch.cuda.is_available():
        state_dict = torch.load(cfg['best_model_path'],map_location=torch.device('cuda:{}'.format(cfg['gpuID'])))
    else:
        state_dict = torch.load(cfg['best_model_path'],map_location=torch.device('cpu'))

    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    if len(missing_keys) > 0:
        print(f"[Warning] Missing keys in state_dict: {missing_keys}")
        if any("attribute_graph" in k for k in missing_keys):
             print("[Info] Disabling AttributeGraphModule because weights are missing in checkpoint.")
             if hasattr(model, 'rhythm_idx'): model.rhythm_idx = None
             if hasattr(model, 'melody_idx'): model.melody_idx = None

    # Data
    inference_dataset = MidiDataset(config=cfg,dataroot=input_pkl)
    inference_dataloader = DataLoader(dataset=inference_dataset, batch_size=1, shuffle=True, drop_last=False)

    # Inference
    tgt_infer_list = []
    for batch in tqdm(inference_dataloader):

        # predict
        tgt_infer = inference(cfg=cfg, model=model, batch=batch, device=device)

        # write midi
        for idx,t in enumerate(tgt_infer):
            name = batch['name'][idx]
            save_path = os.path.join(midi_infer_dir,f"{name.split('.')[0]}.mid")
            write_midi(cfg,t,save_path,batch['name'][idx])

            # save pkl
            cur_dict = {}
            cur_dict['name'] = batch['name'][idx]
            cur_dict['condition'] = batch['condition'][idx].cpu().numpy()
            cur_dict['note_feat'] = batch['note_feat'][idx].cpu().numpy()
            cur_dict['rpp_feat'] = batch['rpp_feat'][idx].cpu().numpy()
            cur_dict['note_inference'] = tgt_infer[idx].cpu().numpy()
            tgt_infer_list.append(cur_dict)

            # log
            with open(txt_test,'a') as f:
                f.write(cur_dict['name'])
                f.write('\n')
                print(*cur_dict['note_inference'],sep='\n',file=f)

    # save pkl
    with open(output_pkl,'wb') as f:
        pickle.dump(tgt_infer_list,f)


def note_check(pre_note,now_note, constraints={}):
    
    '''
    check whether note is illegal
    '''
    
    if (now_note[0] == pre_note[0] and now_note[1]<=pre_note[1]) or now_note[0]<pre_note[0]:
        return True
    
    pre_end_tick = pre_note[0] * 1920 + (pre_note[1] + pre_note[2]-1) * 120
    now_start_tick = now_note[0] * 1920 + now_note[1] * 120
    if now_start_tick - pre_end_tick >= 3840:
        return True

    return False



if __name__ == '__main__':
    main()
