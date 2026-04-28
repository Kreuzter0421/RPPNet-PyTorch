import numpy as np
import pickle
from pprint import pprint
from sklearn.model_selection import train_test_split
import yaml
import os
from tqdm import tqdm
from utils.Split_Rps import data_cleaner_batch
from utils.Split_Rps import rps_divider
import glob
import miditoolkit
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

Bar_min,Bar_max = 4,128
Rps_min,Rps_max = 8,255
Rps_token_max = 256      
Note_min,Note_max = 16,1023
Note_token_max = 1024

RPS_EXPORT_FEATURES = ['bar','position','duration','rhythm_pattern','melody_contour','pitch_region']


def build_rps_start_vector(rps_feat2idx):
    """Return a start token vector aligned with RPS_EXPORT_FEATURES."""
    start_vals = []
    for feat in RPS_EXPORT_FEATURES:
        mapping = rps_feat2idx.get(feat, {})
        start_vals.append(mapping.get('<start>', 0))
    return np.array(start_vals)



class RPS:
    def __init__(self, bar, position, duration, rhythm_pattern, melody_contour, pitch_region, chord_quality, bass, root, tension, key_mode, tempo):
        self.bar = bar
        self.position = position
        self.duration = duration
        self.rhythm_pattern = rhythm_pattern
        self.melody_contour = melody_contour
        self.pitch_region = pitch_region
        self.chord_quality = chord_quality
        self.bass = bass
        self.root = root
        self.tension = tension
        self.key_mode = key_mode
        self.tempo = tempo


class RPSID():
    def __init__(self):
        self.bar = None
        self.position = None
        self.duration = None
        self.rhythm_pattern = None
        self.melody_contour = None
        self.pitch_region = None
        self.is_rest = None

    @property
    def node_feat(self):
        values = []
        for feat in RPS_EXPORT_FEATURES:
            val = getattr(self, feat, None)
            if val is None:
                val = 0
            values.append(val)
        return np.array(values)

    def __repr__(self):
        return  f"bar:{self.bar:<6} position:{self.position:<6} duration:{self.duration:<6} " \
        f"rhythm_pattern:{self.rhythm_pattern:<6} melody_contour:{self.melody_contour:<6} " \
        f"pitch_region:{self.pitch_region:<6}"

class NOTEID():
    def __init__(self):
        self.bar = None
        self.position = None
        self.duration = None
        self.pitch = None
        self.velocity = None

    @property
    def note_feat(self):
        return np.array([self.bar, self.position, self.duration,self.pitch,self.velocity])

# Worker global variables
_worker_rps_feat2idx = None
_worker_note_feat2idx = None

def init_worker(rps_feat2idx, note_feat2idx):
    global _worker_rps_feat2idx, _worker_note_feat2idx
    _worker_rps_feat2idx = rps_feat2idx
    _worker_note_feat2idx = note_feat2idx

def process_general_midi_task(args):
    """
    Worker function to process a single MIDI file.
    args: (midi_path, algorithm)
    Returns: (result_dict, status)
    status: None (success), 'skipped' (filtered), 'error_skip' (assertion), 'error' (exception)
    """
    midi, algorithm = args
    rps_feat2idx = _worker_rps_feat2idx
    note_feat2idx = _worker_note_feat2idx
    RESOLUTION = 120
    
    try:
        midi_obj = miditoolkit.MidiFile(midi)
        end_bar = midi_obj.instruments[0].notes[-1].start // 1920 + 1
        if end_bar < Bar_min: return None, 'skipped'
        if end_bar > Bar_max: return None, 'skipped'
        
        note_list, rps_list = rps_divider(midipath=midi, algorithm=algorithm, need_log=True)

        if note_list[-1].end // 1920 > 127: return None, 'skipped'

        # bar & rps filtering
        for i, rps in enumerate(rps_list):
            if rps.start // 1920 + 1 > Bar_max:
                rps_list = rps_list[:i]
                break
        
        num_rps = len(rps_list)
        if num_rps > Rps_max: rps_list = rps_list[:Rps_max]
        if num_rps < Rps_min: return None, 'skipped'

        note_list = []
        for rps in rps_list: note_list += rps.rps

        for i, note in enumerate(note_list):
            if note.start // 1920 + 1 > Bar_max:
                note_list = note_list[:i]
                break

        num_note = len(note_list)
        if num_note > Note_max: note_list = note_list[:Note_max]
        if num_note < Note_min: return None, 'skipped'

        # Input Dict Construction
        cur_dict = {
            "name": None, "condition": None, "rps_feat": None, "rps_feat_gt": None,
            "rps_mask": None, "note_feat": None, "note_feat_gt": None, "note_mask": None
        }

        # RPSID Construction
        RPSID_list = [RPSID() for _ in range(len(rps_list))]
        for i, rps in enumerate(rps_list):
            cur_feat = str(rps.bar)
            RPSID_list[i].bar = rps_feat2idx['bar'][cur_feat]

            pos_step = rps.position // RESOLUTION
            pos_step = max(0, min(pos_step, 16))
            cur_feat = str(pos_step)
            if cur_feat not in rps_feat2idx['position']: cur_feat = '0'
            RPSID_list[i].position = rps_feat2idx['position'][cur_feat]

            dur_step = rps.duration // RESOLUTION
            dur_step = max(0, min(dur_step, 32))
            cur_feat = str(dur_step)
            if cur_feat not in rps_feat2idx['duration']: cur_feat = '0'
            RPSID_list[i].duration = rps_feat2idx['duration'][cur_feat]

            cur_feat = str(rps.rhythm_pattern)
            RPSID_list[i].rhythm_pattern = rps_feat2idx['rhythm_pattern'][cur_feat]

            cur_feat = str(rps.melody_contour)
            RPSID_list[i].melody_contour = rps_feat2idx['melody_contour'][cur_feat]

            cur_feat = str(rps.pitch_region)
            RPSID_list[i].pitch_region = rps_feat2idx['pitch_region'][cur_feat]

        # rps_feat & Pad
        start_vector = build_rps_start_vector(rps_feat2idx)
        cur_rps_feat = [start_vector]
        for rps in RPSID_list:
            cur_rps_feat.append(rps.node_feat)
        cur_rps_feat = np.array(cur_rps_feat)
        cur_rps_feat_gt = cur_rps_feat[1:, :]

        pad_lenth = max(Rps_token_max - cur_rps_feat.shape[0], 0)
        pad_lenth_gt = max(Rps_token_max - cur_rps_feat.shape[0]+1, 0)

        cur_dict['rps_feat'] = np.pad(cur_rps_feat, ((0, pad_lenth), (0, 0)), 'constant', constant_values=0)
        cur_dict['rps_feat_gt'] = np.pad(cur_rps_feat_gt, ((0, pad_lenth_gt), (0, 0)), 'constant', constant_values=0)
        
        rps_mask_1 = np.ones(cur_rps_feat.shape[0])
        if pad_lenth > 0:
            rps_mask_0 = np.zeros(pad_lenth)
            cur_dict['rps_mask'] = np.concatenate((rps_mask_1, rps_mask_0))
        else:
            cur_dict['rps_mask'] = rps_mask_1

        cur_dict['name'] = os.path.basename(midi)
        cur_dict['condition'] = np.array(0)

        # Note Features
        note_feat = [[1,0,0,0,0]] 
        for note in note_list:
            cur_note = []
            cur_feat = note.start // 1920
            cur_note.append(note_feat2idx['bar'][str(cur_feat)])
            
            cur_feat = (note.start % 1920) // 120
            cur_note.append(note_feat2idx['position'][str(cur_feat)])
            
            cur_feat = min((note.end - note.start) // 120 , 16)
            cur_note.append(note_feat2idx['duration'][str(cur_feat)])
            
            cur_feat = note.pitch
            cur_note.append(note_feat2idx['pitch'][str(cur_feat)])
            
            cur_feat = note.velocity
            cur_note.append(100)

            if len(note_feat) < Note_token_max:
                note_feat.append(cur_note)

        note_feat = np.array(note_feat)
        note_feat_gt = note_feat[1:, :]
        note_token_lenth = note_feat.shape[0]
        note_pad_lenth = max(Note_token_max - note_token_lenth, 0)
        note_pad_lenth_gt = max(Note_token_max - note_token_lenth+1, 0)
        
        cur_dict['note_feat'] = np.pad(note_feat, ((0, note_pad_lenth), (0, 0)), 'constant', constant_values=0)
        cur_dict['note_feat_gt'] = np.pad(note_feat_gt, ((0, note_pad_lenth_gt), (0, 0)), 'constant', constant_values=0)
        
        note_mask_1 = np.ones(note_feat.shape[0])
        note_mask_0 = np.zeros(note_pad_lenth)
        cur_dict['note_mask'] = np.concatenate((note_mask_1, note_mask_0))

        return cur_dict, None

    except AssertionError as error:
        if "音符数量解析不对等" in str(error):
            return None, 'error_skip'
        else:
            # traceback.print_exc()
            return None, 'error'
    except Exception as error:
        # traceback.print_exc()
        return None, 'error'

# metadata -> standard data (General)
def get_standard_data_General_Pad(midi_root, RPS_FEAT2IDX, NOTE_FEAT2IDX, output_root='../Datasets/pretrain', split_dict_path=None,
                                  algorithm=None):
    """
    Directly splits data into train/val/test and saves them in chunks to avoid OOM.
    output: Writes pickle chunks to ../Datasets/pretrain/{train,val,test}/part_X.pkl
    """
    import random
    
    # Configuration
    CHUNK_SIZE = 30000
    OUTPUT_ROOT = output_root
    splits = ['train', 'test', 'val']
    
    # Prepare directories
    for split in splits:
        os.makedirs(os.path.join(OUTPUT_ROOT, split), exist_ok=True)
        
    # Buffers for each split
    buffers = {split: [] for split in splits}
    chunk_counters = {split: 0 for split in splits}
    
    # Counters
    valid_count = 0
    skipped_count = 0
    
    midi_file = glob.glob(midi_root + '**/**/*.mid', recursive=True)
    
    split_lookup_dict = {}
    if split_dict_path and os.path.exists(split_dict_path):
        import json
        with open(split_dict_path, 'r') as f:
            splits_json = json.load(f)
        for s_key, f_list in splits_json.items():
            for fname in f_list:
                split_lookup_dict[fname] = s_key

    split_lookup_dict = {}
    if split_dict_path and os.path.exists(split_dict_path):
        import json
        with open(split_dict_path, 'r') as f:
            splits_json = json.load(f)
        for s_key, f_list in splits_json.items():
            for fname in f_list:
                split_lookup_dict[fname] = s_key

    # Use ProcessPoolExecutor
    max_workers = 8
    if max_workers is None: max_workers = 4

    print(f"Starting multiprocessing with {max_workers} workers for {len(midi_file)} files.")
    
    # Prepare arguments
    tasks = [(midi, algorithm) for midi in midi_file]
    
    # Helper to flush a buffer to disk
    def flush_buffer(split_name, force=False):
        buf = buffers[split_name]
        if not buf:
            return
        
        if len(buf) >= CHUNK_SIZE or force:
            chunk_idx = chunk_counters[split_name]
            out_name = f"{split_name}_part_{chunk_idx}.pkl"
            out_file = os.path.join(OUTPUT_ROOT, split_name, out_name)
            
            print(f"Saving {split_name} chunk {chunk_idx} ({len(buf)} items) to {out_name}...")
            with open(out_file, 'wb') as f:
                pickle.dump(buf, f)
            
            # Reset buffer and increment counter
            buffers[split_name] = []
            chunk_counters[split_name] += 1

    # Processing loop
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker, initargs=(RPS_FEAT2IDX, NOTE_FEAT2IDX)) as executor:
        BATCH_SIZE = 5000
        for batch_start in range(0, len(tasks), BATCH_SIZE):
            batch_tasks = tasks[batch_start:batch_start+BATCH_SIZE]
            futures = {executor.submit(process_general_midi_task, task): task for task in batch_tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing Batch {batch_start//BATCH_SIZE + 1}/{(len(tasks)+BATCH_SIZE-1)//BATCH_SIZE}"):
                try:
                    result, status = future.result()
                    if status is None:
                        valid_count += 1
                        
                        task_info = futures[future]
                        if split_dict_path:
                            filename = os.path.basename(task_info[0])
                            target_split = None
                            if filename in split_lookup_dict:
                                target_split = split_lookup_dict[filename]
                            else:
                                del futures[future]
                                continue # File not in dict, skip
                        else:
                            rand_val = random.random() #The ratio of train/test/val
                            if rand_val < 0.9:
                                target_split = 'train'
                            elif rand_val < 0.95:
                                target_split = 'test'
                            else:
                                target_split = 'val'
                        
                        buffers[target_split].append(result)
                        
                        # Check if buffer needs flushing
                        if len(buffers[target_split]) >= CHUNK_SIZE:
                            flush_buffer(target_split)
                            
                    elif status == 'error_skip':
                        skipped_count += 1
                    
                    del futures[future]
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
                    if future in futures:
                        del futures[future]

    # Flush remaining items
    for split in splits:
        flush_buffer(split, force=True)

    print(f"Total skipped files due to note count mismatch: {skipped_count}")
    print(f"Total valid files: {valid_count}")
    print('Done processing and splitting.')




if __name__ == '__main__':


    # --------------------- midi -> pkl(General) ---------------------
    with open('./feat2idx_rps.pkl','rb') as f:
        RPS_FEAT2IDX = pickle.load(f)

    with open('./feat2idx_note.pkl','rb') as f:
        NOTE_FEAT2IDX = pickle.load(f)

    import sys
    import argparse


    parser = argparse.ArgumentParser(description='Run Pretrain Data Generation')
    parser.add_argument('--algorithm', type=str, default='DP', help='Algorithm for splitting RPS (DP, RANDOM)') #RANDOM is for Ablation Study
    parser.add_argument('--split_dict', type=str, default=None, help='Path to JSON specifying splits for files')
    parser.add_argument('--output_root', type=str, default='../Datasets/wikifonia', help='Output folder for dataset')
    args, unknown = parser.parse_known_args()

    get_standard_data_General_Pad(
        '../DataProcess/wikifonia', RPS_FEAT2IDX, NOTE_FEAT2IDX,
        split_dict_path=args.split_dict,
        output_root=args.output_root,
        algorithm=args.algorithm
    )
















