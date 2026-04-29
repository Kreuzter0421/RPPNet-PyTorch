import copy
import miditoolkit
import pprint
import numpy as np
import os
import collections
import re
from bisect import bisect_right
from .RPP_Detection import RPP_Detection
import shutil
from time import time
from tqdm import tqdm

BAR_TICKS = 1920
BEAT_TICKS = BAR_TICKS // 4
MIN_PHRASE_NOTES = 2
MIN_PHRASE_BAR_SPAN = 1.0
SMALL_PHRASE_NOTE_THRESHOLD = 6
STRONG_BEAT_INDICES = {0, 2}
GLOBAL_ALGORITHM = 'DP'

NOTE_NAME_TO_PC = {
    'C': 0, 'B#': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4,
    'F': 5, 'E#': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11
}

PC_TO_NOTE_NAME = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

MAJOR_KEY_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                              2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=float)
MINOR_KEY_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                              2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=float)


def compute_dynamic_rest_threshold(midi_obj):
    """Return an adaptive rest threshold in ticks based on tempo."""
    if midi_obj.tempo_changes:
        tempo = midi_obj.tempo_changes[0].tempo
    else:
        tempo = 120
    tempo = float(np.clip(tempo, 50, 200))
    ratios = [0.75, 0.6, 0.5, 0.4, 0.3]
    tempo_knots = [50, 90, 120, 150, 200]
    ratio = float(np.interp(tempo, tempo_knots, ratios))
    min_ratio = 0.25
    ticks = int(max(ratio, min_ratio) * BEAT_TICKS)
    return max(ticks, BEAT_TICKS // 4)


def is_strong_skeleton(note):
    beat_idx = ((note.start % BAR_TICKS) // BEAT_TICKS)
    dur_ok = (note.end - note.start) >= (BEAT_TICKS // 2)
    return beat_idx in STRONG_BEAT_INDICES and dur_ok


def smooth_phrase_segments(phrase, div_grid, min_notes=MIN_PHRASE_NOTES, min_bar_span=MIN_PHRASE_BAR_SPAN,
                           small_threshold=SMALL_PHRASE_NOTE_THRESHOLD):
    if not phrase:
        return phrase, div_grid
    merged_phrases = []
    merged_div_grid = [div_grid[0]]
    chunk_notes = []
    chunk_start = div_grid[0]
    for idx, ph in enumerate(phrase):
        chunk_notes.extend(ph)
        chunk_end = div_grid[idx + 1]
        bar_span = (chunk_end - chunk_start) / BAR_TICKS
        note_cnt = len(chunk_notes)
        is_last = idx == len(phrase) - 1
        if (note_cnt >= min_notes and bar_span >= min_bar_span) or is_last:
            if is_last and (note_cnt < min_notes or bar_span < min_bar_span) and merged_phrases:
                merged_phrases[-1].extend(chunk_notes)
                merged_div_grid[-1] = chunk_end
            else:
                merged_phrases.append(chunk_notes)
                merged_div_grid.append(chunk_end)
            chunk_notes = []
            chunk_start = chunk_end
    if not merged_phrases:
        merged_phrases = [chunk_notes]
        merged_div_grid = [div_grid[0], div_grid[-1]]
    merged_phrases, merged_div_grid = merge_small_phrase_pairs(merged_phrases, merged_div_grid, small_threshold)
    return merged_phrases, merged_div_grid


def merge_small_phrase_pairs(phrases, div_grid, threshold):
    if not phrases or len(phrases) == 1:
        return phrases, div_grid

    def is_adjacent(left_notes, right_notes):
        if not left_notes or not right_notes:
            return False
        return left_notes[-1].end == right_notes[0].start

    i = 0
    while i < len(phrases):
        current = phrases[i]
        if len(current) < threshold and current:
            prev_adjacent = i > 0 and is_adjacent(phrases[i - 1], current)
            next_adjacent = i + 1 < len(phrases) and is_adjacent(current, phrases[i + 1])

            if prev_adjacent or next_adjacent:
                if prev_adjacent:
                    phrases[i - 1].extend(current)
                    phrases.pop(i)
                    div_grid.pop(i)  # remove boundary between previous and current
                    i = max(i - 1, 0)
                    continue
                else:
                    phrases[i + 1] = current + phrases[i + 1]
                    phrases.pop(i)
                    div_grid.pop(i + 1)  # remove boundary between current and next
                    continue
        i += 1

    return phrases, div_grid


def merge_single_note_segments(segments, max_len=3):
    if not segments:
        return segments
    merged = []
    pending = []

    def attach_pending(target):
        nonlocal pending
        if not pending:
            return target
        capacity = max_len - len(target)
        if capacity <= 0:
            return target
        attach = pending[:capacity]
        pending = pending[capacity:]
        return attach + target

    for seg in segments:
        if len(seg) == 1:
            pending.extend(seg)
            continue
        if pending:
            if merged:
                updated = attach_pending(merged[-1])
                merged[-1] = updated
            else:
                seg = attach_pending(seg)
        merged.append(seg)

    if pending:
        for note in pending:
            merged.append([note])

    return merged


def _normalize_tonic_name(name):
    if not name:
        return None
    clean = name.strip().replace('♯', '#').replace('♭', 'b')
    if not clean:
        return None
    match = re.match(r'([A-Ga-g])([#b]?)(.*)', clean)
    if not match:
        return None
    letter = match.group(1).upper()
    accidental = match.group(2)
    remainder = match.group(3)
    tonic = letter + accidental
    mode_hint = remainder.lower()
    if mode_hint.startswith('maj'):
        mode = 'major'
    elif mode_hint.startswith('min') or mode_hint.startswith('m'):
        mode = 'minor'
    else:
        mode = 'minor' if clean.lower().endswith('m') else 'major'
    return tonic, mode


def parse_key_signature_name(key_name):
    result = _normalize_tonic_name(key_name)
    if not result:
        return None
    tonic, mode = result
    tonic_pc = NOTE_NAME_TO_PC.get(tonic)
    if tonic_pc is None:
        return None
    return {
        'tonic_pc': tonic_pc,
        'mode': mode,
        'source': 'metadata',
        'name': f"{tonic} {'minor' if mode == 'minor' else 'major'}"
    }


def estimate_key_from_notes(note_list):
    if not note_list:
        return None
    histogram = np.zeros(12, dtype=float)
    for note in note_list:
        pc = note.pitch % 12
        histogram[pc] += max(note.duration, 1)
    total = histogram.sum()
    if total == 0:
        histogram += 1.0
        total = histogram.sum()
    histogram /= total
    best = {'score': float('-inf')}
    for mode, profile in (('major', MAJOR_KEY_PROFILE), ('minor', MINOR_KEY_PROFILE)):
        for shift in range(12):
            rotated = np.roll(profile, shift)
            score = float(np.dot(histogram, rotated))
            if score > best['score']:
                best = {
                    'score': score,
                    'tonic_pc': shift,
                    'mode': mode,
                    'source': 'estimated'
                }
    tonic_name = PC_TO_NOTE_NAME[best['tonic_pc']]
    best['name'] = f"{tonic_name} {'minor' if best['mode'] == 'minor' else 'major'}"
    return best


def detect_song_key(midi_obj, note_list):
    key_sig = midi_obj.key_signature_changes
    if key_sig:
        parsed = parse_key_signature_name(key_sig[-1].key_name)
        if parsed:
            return parsed
    return estimate_key_from_notes(note_list)


def classify_cadence_for_rpp(rpp, key_info):
    if key_info is None or not rpp.rpp:
        return 'non_cadence'
    tonic_pc = key_info['tonic_pc']
    dominant_pc = (tonic_pc + 7) % 12
    leading_pc = (tonic_pc - 1) % 12
    note_pcs = [note.pitch % 12 for note in rpp.rpp]
    final_note = rpp.rpp[-1]
    final_pc = final_note.pitch % 12
    contains_dominant = dominant_pc in note_pcs
    contains_leading = leading_pc in note_pcs
    final_long = (final_note.duration >= BEAT_TICKS // 2)
    final_strong = ((final_note.start % BAR_TICKS) // BEAT_TICKS) in STRONG_BEAT_INDICES
    if final_pc == tonic_pc and (contains_dominant or contains_leading or final_long) and (final_long or final_strong):
        return 'full_cadence'
    if final_pc == dominant_pc or (contains_dominant and final_pc != tonic_pc):
        return 'half_cadence'
    return 'non_cadence'


def annotate_cadence_tags(rpp_sequence, key_info):
    if not rpp_sequence:
        return
    phrase_groups = collections.defaultdict(list)
    fallback_group = []
    for idx, rpp in enumerate(rpp_sequence):
        if not hasattr(rpp, 'cadence_tag'):
            rpp.cadence_tag = 'non_cadence'
        phrase_idx = getattr(rpp, 'phrase_index', None)
        if phrase_idx is None:
            fallback_group.append((idx, rpp))
        else:
            phrase_groups[phrase_idx].append((idx, rpp))
    if not phrase_groups and fallback_group:
        phrase_groups[-1] = fallback_group
    for group in phrase_groups.values():
        if not group:
            continue
        for _, rpp in group:
            rpp.cadence_tag = 'non_cadence'
        _, final_rpp = group[-1]
        final_rpp.cadence_tag = classify_cadence_for_rpp(final_rpp, key_info)


def assign_break_note(note_all, skeleton_note):
    """Mark the break note and return the note that should receive the marker."""
    for idx, note in enumerate(note_all):
        if note.start == skeleton_note.start and note.end == skeleton_note.end and note.pitch == skeleton_note.pitch:
            target = note
            if idx + 2 < len(note_all):
                next_note = note_all[idx + 1]
                n_next_note = note_all[idx + 2]
                if next_note.start == note.end and (next_note.end - next_note.start) >= (note.end - note.start):
                    target = next_note
                if next_note.start == note.end and next_note.end - n_next_note.start != 0:
                    target = next_note
            target.is_break_note = True
            return target
    return skeleton_note

class Note(miditoolkit.Note):
    def __init__(self,velocity,pitch,start,end,is_skeleton=False,is_syncopation=False):
        super(Note,self).__init__(velocity=velocity,pitch=pitch,start=start,end=end)
        self.is_skeleton = is_skeleton
        self.is_syncopation = is_syncopation
        self.rythm_weight = -1
        self.is_break_note = False

    @property
    def duration(self):
        return self.end - self.start

    def __repr__(self):
        return f'Start:{self.start//1920+1:>2}Bar {self.start%1920*16//1920:>2}Grid  Type:{self.is_skeleton:<2},{self.is_syncopation:<3}  ' \
               f'Weight1:{self.rythm_weight-self.duration//120:<3}  Weight2:{self.rythm_weight}\n'

class RPP():
    def __init__(self,rpp,number=0):
        self.rpp = rpp
        self.number = number
        self.cnt = 0
        self.mark = 0
        self.token_type = [0,0,0]
        self.shape_and_kind = ()   
        self.phrase_index = None
        self.cadence_tag = 'non_cadence'
        self.rhythm_pattern2id = {
            (0, 1): 0,
            (0, 0, 1): 1,
            (1, 0): 2,
            (1, 0, 0): 3,
            (0, 1, 0): 4,
            (1,): 5,
            (0,): 6
        }
        self.token_name = 'RPP_' + str(self.rhythm_pattern2id[self.rhythm_pattern]) + '_' + str(self.melody_contour)



    @property
    def bar(self):
        bar = self.rpp[0].start // 1920
        return bar

    @property
    def start(self):
        return self.rpp[0].start

    @property
    def end(self):
        return self.rpp[-1].end

    @property
    def position(self):
        return self.rpp[0].start % 1920

    @property
    def duration(self):
        return self.rpp[-1].end - self.rpp[0].start

    @property
    def cross_bar(self):
        if len(self.rpp)>1 and (self.rpp[0].start // 1920 != self.rpp[-1].start//1920):
            return 1
        else:
            return 0

    @property #节奏型
    def type(self):
        if(len(self.rpp) == 1):
            rpp = self.rpp
            return (rpp[0].end - rpp[0].start,0,0,0,0)
        elif len(self.rpp) == 2:
            rpp = self.rpp
            return (rpp[0].end-rpp[0].start,rpp[1].start-rpp[0].end,rpp[1].end-rpp[1].start,0,0)
        else:
            rpp = self.rpp
            return (rpp[0].end-rpp[0].start,rpp[1].start-rpp[0].end,rpp[1].end-rpp[1].start,rpp[2].start-rpp[1].end,rpp[2].end-rpp[2].start)

    @property 
    def melody_contour(self):
        return get_rpp_melody_contour(note_list=self.rpp)

    @property 
    def rhythm_pattern(self):
        if len(self.rpp) == 1:
            if self.rpp[0].is_break_note:       
                return tuple([1])
            else:
                return tuple([0])
        else:
            l = [0] * len(self.rpp)
            l[np.argmax([x.rythm_weight for x in self.rpp])] = 1
            return tuple(l)

    @property 
    def shape(self):
        return (self.position,self.rhythm_pattern,self.melody_contour,self.cross_bar)

    @property 
    def pitch_region(self):
        pitch_list = [int(note.pitch) for note in self.rpp]
        return int(sum(pitch_list)/len(pitch_list))


    def __repr__(self):
        
        return f'Start:{self.start//1920+1:>2}Bar {self.start%1920*16//1920:>2}Grid  Contain:{len(self.rpp):<2}  Rythm_structrue:{str(self.rhythm_pattern):<10}' \
               f' Melody:{self.melody_contour:<2}'

class RP():
    def __init__(self,rpp_list):
        self.rpp_list = rpp_list
        self.start = rpp_list[0].start
        self.end = rpp_list[-1].end
        self.mark = 0
        self.token_name = 'RP'
        self.token_type = [0,0,0]


    def __repr__(self):

        rpp_string = ''
        for rpp in self.rpp_list:
            rpp_string += f'RPP_ID:{rpp.number} \n'

        return f'Start:[Bar:{self.start // 1920:>2d},Position:{(self.start % 1920) * 16 // 1920:>2d}]      '\
               f'End:[Bar:{self.end // 1920:>2d},Position:{(self.end % 1920) * 16 // 1920:>2d}]\n' \
               f'{rpp_string}'\
               f'Contain:{len(self.rpp_list)}'

class P():
    def __init__(self,start,end):
        self.rp_list = []
        self.start = start
        self.end = end
        self.mark = 0
        self.token_name = 'P'


    def __repr__(self):

        return f'P contain:{len(self.rp_list)}'




def data_cleaner_batch(root,function_dict,tgt_root=None):

    # default container
    dict_list = []

    # confirm function
    function = {'quantify_120ticks':False,
                'clean_overlapNote': False,
                'clean_drumMidi': False,
                'split_1920ticks':False,
                'clean_shortMidi':False,
                'RPPdetect_compatibility_check':False,
                'note_amount_check': False,
                'clean_overlapNote_part':False,
                'rename':False}

    function.update(function_dict)

    print('### DATA_CLEANER START! ###\n----------------------------\nThe Function You Need:')
    for i,(func,able) in enumerate(function.items()):
        if able:
            print(f'{i:<2}| {func}')

    # path check
    if tgt_root == None or tgt_root == root: 
        tgt_root = root
    else:                                   
        if not os.path.exists(tgt_root):
            os.mkdir(tgt_root)
        os.system('rm -rf %s/*' % tgt_root)

        # copy file
        file_raw_list = [f for f in os.listdir(root) if '.mid' in f]
        file_exist_list = [f for f in os.listdir(tgt_root) if '.mid' in f]
        for f in file_raw_list:
            if f not in file_exist_list:
                shutil.copyfile(os.path.join(root,f),os.path.join(tgt_root,f))
    work_root = tgt_root

    # [1] quantify_120ticks
    if function['quantify_120ticks']:
        print('\n----------------------------\n# quantify_120ticks START #')
        file_name_list = [f for f in os.listdir(work_root) if '.mid' in f]
        
        file_path_list = [os.path.join(work_root,f) for f in file_name_list]
        for midipath in tqdm(file_path_list):
            midi_obj = miditoolkit.MidiFile(midipath)
            max_tick = midi_obj.instruments[0].notes[-1].end // 120 * 120 + 120
            grid = np.arange(0,max_tick,120)
            cnt = 0
            for note in midi_obj.instruments[0].notes[:]:
                
                if note.end == note.start:
                    midi_obj.instruments[0].notes.remove(note)

                
                if note.end - note.start <= 100:
                    note.end = note.start + 120
                note.start = min(grid,key= lambda x:abs(x-note.start))
                note.end = min(grid,key= lambda x:abs(x-note.end))

            
            for note in midi_obj.instruments[0].notes[:]:
                if note.end == note.start:
                    midi_obj.instruments[0].notes.remove(note)

            midi_obj.dump(midipath)

    
    if function['clean_overlapNote']:
        print('\n----------------------------\n# clean_overlapNote START #')
        file_name_list = [f for f in os.listdir(work_root) if '.mid' in f]
        file_path_list = [os.path.join(work_root,f) for f in file_name_list]
        for midipath in tqdm(file_path_list):
            midi_obj = miditoolkit.MidiFile(midipath)
            need_clean_note = []
            for i,note in enumerate(midi_obj.instruments[0].notes[:-1]):
                next = midi_obj.instruments[0].notes[i+1]
                if note.start == next.start and note.end == next.end:
                    if note.pitch<=next.pitch:
                        need_clean_note.append(note)
                    else:
                        need_clean_note.append(note)
            for note in need_clean_note:
                midi_obj.instruments[0].notes.remove(note)
            midi_obj.dump(midipath)

    # [3] clean_drumMidi
    if function['clean_drumMidi']:
        print('\n----------------------------\n# clean_drumMidi START #')

        drum_midi = []
        file_name_list = [f for f in os.listdir(work_root) if '.mid' in f]
        file_path_list = [os.path.join(work_root, f) for f in file_name_list]
        for midipath in tqdm(file_path_list):
            num = 0
            midi_obj = miditoolkit.MidiFile(midipath)
            for note in midi_obj.instruments[0].notes:
                if note.pitch <=50:
                    num +=1

            if num / len(midi_obj.instruments[0].notes)>=0.5:
                os.remove(midipath)
                drum_midi.append(midipath)

        print('Deleted Midi:')
        print(*drum_midi,sep='\n')

    # [4] 'split_1920ticks'
    if function['split_1920ticks']:
        print('\n----------------------------\n# split_1920ticks START #')
        file_name_list = [f for f in os.listdir(work_root) if '.mid' in f]
        file_path_list = [os.path.join(work_root, f) for f in file_name_list]
        for midipath in tqdm(file_path_list):
            midi_obj = miditoolkit.MidiFile(midipath)

            grid_line = [0]
            for i, note in enumerate(midi_obj.instruments[0].notes):
                if i == len(midi_obj.instruments[0].notes) - 1:
                    grid_line.append(note.end)
                elif midi_obj.instruments[0].notes[i + 1].start - note.end >= 1920:
                    grid_line.append(note.end)

            for i, (start, end) in enumerate(zip(grid_line[:-1], grid_line[1:])):
                new_midi = miditoolkit.MidiFile()
                new_midi.instruments.append(miditoolkit.Instrument(program=0, name='melody'))
                note_list = []

                for note in midi_obj.instruments[0].notes[:]:
                    if note.start >= start and note.end <= end:
                        note_list.append(note)

                if len(note_list) == 0:
                    continue

                note_list.sort(key=lambda x: x.start)
                offset = note_list[0].start // 1920 * 1920
                for note in note_list:
                    new_midi.instruments[0].notes.append(
                        miditoolkit.Note(start=note.start - offset, end=note.end - offset, pitch=note.pitch,
                                         velocity=note.velocity))


                new_midi.dump(midipath[:-4] + f'#{i}.mid')

            os.remove(midipath)

    # [5] clean_shortMidi
    if function['clean_shortMidi']:
        print('\n----------------------------\n# clean_shortMidi START #')

        shortmidi = []
        file_name_list = [f for f in os.listdir(work_root) if '.mid' in f]
        file_path_list = [os.path.join(work_root, f) for f in file_name_list]
        for midipath in tqdm(file_path_list):
            midi_obj = miditoolkit.MidiFile(midipath)
            if midi_obj.instruments[0].notes[-1].start //1920+1<16:
                os.remove(midipath)
                shortmidi.append(midipath)

        print('Deleted Midi:')
        print(*shortmidi, sep = '\n')

    # [6] RPPdetect_compatibility_check
    if function['RPPdetect_compatibility_check']:
        print('\n----------------------------\n# RPPdetect_compatibility_check START #')

        deletmidi = []
        file_name_list = [f for f in os.listdir(work_root) if '.mid' in f]
        file_path_list = [os.path.join(work_root, f) for f in file_name_list]
        for midipath in tqdm(file_path_list):
            m = RPP_Detection(midipath)
            t1 = time()
            try:
                m.all_steps()
            except:
                os.remove(midipath)
                deletmidi.append(midipath)
                continue

        print('Deleted Midi:')
        print(*deletmidi, sep = '\n')

    # [7] note_amount_check
    if function['note_amount_check']:
        print('\n----------------------------\n# note_amount_check START #')

        deletmidi = []
        file_name_list = [f for f in os.listdir(work_root) if '.mid' in f]
        file_path_list = [os.path.join(work_root, f) for f in file_name_list]
        for midipath in tqdm(file_path_list):
            midi_obj = miditoolkit.MidiFile(midipath)
            if len(midi_obj.instruments[0].notes) <64:
                os.remove(midipath)
                deletmidi.append(midipath)

        print('Deleted Midi:')
        print(*deletmidi, sep = '\n')

    # [8] clean_overlapNote_part
    if function['clean_overlapNote_part']:
        print('\n----------------------------\n# clean_overlapNote_part START #')
        file_name_list = [f for f in os.listdir(work_root) if '.mid' in f]
        file_path_list = [os.path.join(work_root,f) for f in file_name_list]
        for midipath in tqdm(file_path_list):
            midi_obj = miditoolkit.MidiFile(midipath)
            need_clean_bool = [False] * len(midi_obj.instruments[0].notes)
            need_clean_note = []
            for i,note in enumerate(midi_obj.instruments[0].notes):
                for j,prenote in enumerate(midi_obj.instruments[0].notes[:i]):
                    if prenote.end > note.start and need_clean_bool[j]==False:
                        if prenote.end-prenote.start < note.end-note.start:
                            need_clean_bool[j] = True
                            need_clean_note.append(prenote)
                        else:
                            need_clean_bool[i] = True
                            need_clean_note.append(note)
                            continue

            for note in need_clean_note:
                midi_obj.instruments[0].notes.remove(note)
            midi_obj.dump(midipath)
            
            if len(need_clean_note) > 0:
                print('ERROR:midipath:',midipath)

    # [9] rename
    if function['rename']:
        print('\n----------------------------\n# rename START #')
        file_name_list = [f for f in os.listdir(work_root) if '.mid' in f]
        file_path_list = [os.path.join(work_root, f) for f in file_name_list]

       
        tag = '@@'
        index= 0
        for midipath in file_path_list:
            test_name = os.path.join(os.path.dirname(midipath), f'{tag}{index}.mid')
            os.rename(midipath,test_name)
            # print(f'index:{index}  name1:{midipath} name2:{test_name}')
            index += 1

        
        index= 0
        file_name_list = [f for f in os.listdir(work_root) if '.mid' in f]
        file_path_list = [os.path.join(work_root, f) for f in file_name_list]
        for midipath in file_path_list:
            final_name = os.path.join(os.path.dirname(midipath), f'{index}.mid')
            os.rename(midipath,final_name)
            # print(f'index:{index}  name2:{midipath} name3:{final_name}')
            index += 1


# file_rename
def file_rename(work_root,extend=''):
    name_dict = {}

    print('\n----------------------------\n# rename START #')
    file_name_list = [f for f in os.listdir(work_root) if '.mid' in f]
    file_name_list.sort(key= lambda x:int(x.split('.')[0]))
    print(file_name_list)

    

    file_path_list = [os.path.join(work_root, f) for f in file_name_list]


    index = 0
    for midipath,midiname in zip(file_path_list,file_name_list):
        new_name = f'{index}.mid'
        final_name = os.path.join(os.path.dirname(midipath),new_name )
        os.rename(midipath, final_name)
        name_dict[midiname] = new_name
        index+=1

    return name_dict

# print_RP
def print_RP(RP_list=None):
    RP_list = RP_list
    for i,rp in enumerate(RP_list):
        print(f'--------------------------- RP{i} -----------------------------')
        print(f'START = {rp.start : <8d}       END = {rp.end : <8d}      CONTAIN = {len(rp.rpp_list) : <4d}')
        for rpp in rp.rpp_list:
            print(rpp)

# split_midi_file_by_1920ticks
def split_midi_1920ticks(root):
    file_list = [x for x in os.listdir(root) if x[-4:] == '.mid']

    for f in file_list:
        midipath = os.path.join(root,f)
        midi_obj = miditoolkit.MidiFile(midipath)

        grid_line = [0]
        for i,note in enumerate(midi_obj.instruments[0].notes):
            if i == len(midi_obj.instruments[0].notes)-1:
                grid_line.append(note.end)
            elif midi_obj.instruments[0].notes[i+1].start - note.end >=1920:
                grid_line.append(note.end)

        for i,(start,end) in enumerate(zip(grid_line[:-1],grid_line[1:])):
            new_midi = miditoolkit.MidiFile()
            new_midi.instruments.append(miditoolkit.Instrument(program=0,name='melody'))
            note_list = []

            for note in midi_obj.instruments[0].notes[:]:
                if note.start>=start and note.end<=end:
                    note_list.append(note)

            if len(note_list) == 0:
                continue

            note_list.sort(key=lambda x:x.start)
            offset = note_list[0].start //1920 *1920
            for note in note_list:
                new_midi.instruments[0].notes.append(miditoolkit.Note(start=note.start-offset,end=note.end-offset,pitch=note.pitch,velocity=note.velocity))

            print(f'------ {midipath} {i} -------')
            for note in new_midi.instruments[0].notes:
                print(note)
            new_midi.dump(midipath[:-4]+f'.{i}.mid')

        os.remove(midipath)

# get rpp [[note1,note2] ,[note1,note2,note3]...]
def get_rpplist_raw(midipath = None):
    if midipath == None or midipath[-4:] != '.mid':
        return

    m = RPP_Detection(midi_path=midipath)
    rpp_list,_ = m.all_steps()

    return rpp_list

# quantify single_track -> melody
def quntify_file(dir=None):
    file_list = os.listdir(dir)
    for f in file_list:
        if f[-4:] != '.mid':
            continue
        midipath = os.path.join(dir, f)
        midi_obj = miditoolkit.MidiFile(midipath)

        note_list = []
        for track in midi_obj.instruments:
            note_list.extend(track.notes)

        midi_obj.instruments.clear()
        track_new = miditoolkit.Instrument(program=0, name='melody')
        track_new.notes = note_list
        midi_obj.instruments.append(track_new)
        midi_obj.dump(midipath)

# quantify -> grid 120tick
def grid_quantify(root,tgt_root = '',extend_word=''):
    root = root
    if tgt_root == '':
        tgtroot = root
    else:
        tgtroot = tgt_root

    file = [f for f in os.listdir(root)]
    for f in file[:]:
        if f[-4:] != '.mid':
            file.remove(f)


    for f in file:
        midipath = os.path.join(root,f)
        midi_obj = miditoolkit.MidiFile(midipath)

        max_tick = midi_obj.instruments[0].notes[-1].end // 120 * 120 + 120
        grid = np.arange(0,max_tick,120)

        cnt = 0
        for note in midi_obj.instruments[0].notes[:]:
         
            if note.end == note.start:
                midi_obj.instruments[0].notes.remove(note)

          
            if note.end - note.start <= 100:
                note.end = note.start + 120
            note.start = min(grid,key= lambda x:abs(x-note.start))
            note.end = min(grid,key= lambda x:abs(x-note.end))

       
        for note in midi_obj.instruments[0].notes[:]:
            if note.end == note.start:
                midi_obj.instruments[0].notes.remove(note)

        midi_obj.dump(os.path.join(tgtroot,f))
        print(f'{f} Success!')

# Similarity
def rpp_similarity(section_list): # section_list: [  section1[[rpp1],[rpp2],[rpp3],...],  section2[[rpp1],[rpp2],[rpp3],...]  ]
    # section_all.section.rpp[theme_similarity,pre_similarity]
    similarity_info = []
    for i in range(len(section_list)):  
        cur_section = []
        section_now = section_list[i]
        for rpp, j in zip(section_now, range(len(section_now))):
            if i == 0 and j == 0:
                cur_rpp = [1, 0]
            elif i != 0 and j == 0:
                cur_rpp = similarity_of_twoVector(rpp, section_list[i - 1][0], rpp)
            else:
                cur_rpp = similarity_of_twoVector(section_now[0], section_now[j - 1], rpp)
            cur_section.append(cur_rpp)
        similarity_info.append(cur_section)
    print('**similarity_rpp**')
    pprint.pprint(similarity_info)
    return similarity_info

def similarity_of_twoVector(vector_theme, vector_pre, vector_origin):
    vector_theme = np.array(vector_theme)
    vector_pre = np.array(vector_pre)
    vector_origin = np.array(vector_origin)
    lenth_theme = np.linalg.norm(vector_theme)
    lenth_pre = np.linalg.norm(vector_pre)
    lenth_origin = np.linalg.norm(vector_origin)
    simi1 = round(vector_theme.dot(vector_origin) / (lenth_theme * lenth_origin), 2)
    simi2 = round(vector_pre.dot(vector_origin) / (lenth_pre * lenth_origin), 2)
    return [1.0 * simi1, 1.0 * simi2]

## similarity-theme-only
def similarity_theme(rpp_theme,rpp_list): # rpp_list: [[rpp1],[rpp2],[rpp3],....]
    vector_theme = np.array(rpp_theme)
    lenth_theme = np.linalg.norm(rpp_theme)
    similarity = []
    for rpp_now in rpp_list:
        vector_now = np.array(rpp_now)
        lenth_now = np.linalg.norm(rpp_now)
        simi = round(vector_theme.dot(vector_now) / (lenth_theme * lenth_now), 3)
        similarity.append(simi)
        print(rpp_theme,rpp_now,simi)

    return similarity

# rpp_cnt_dic
def rpp_cnt_dic(section_list):
    dic = {}
    for rpp in section_list:
        currpp = ()
        if len(rpp) == 2 :
            currpp = (rpp[0].end-rpp[0].start,rpp[1].start-rpp[0].end,rpp[1].end-rpp[1].start,0,0)
        elif len(rpp) == 3:
            currpp = (rpp[0].end-rpp[0].start,rpp[1].start-rpp[0].end,rpp[1].end-rpp[1].start,rpp[2].start-rpp[1].end,rpp[2].end-rpp[2].start)

        if currpp in  dic.keys():
            dic[currpp] += 1
        else:
            dic[currpp] = 1

    return dic

# normalize
def normalize(section):
    rpp_list = []
    for rpp in section:
        if len(rpp) == 2:
            rpp_list.append([rpp[0].end-rpp[0].start,rpp[1].start-rpp[0].end,rpp[1].end-rpp[1].start,0,0])
        elif len(rpp) == 3:
            rpp_list.append([rpp[0].end-rpp[0].start,rpp[1].start-rpp[0].end,rpp[1].end-rpp[1].start,rpp[2].start-rpp[1].end,rpp[2].end-rpp[2].start])
    return rpp_list

# write_midi
def writemidi(note_list,savepath):
    midi_obj = miditoolkit.MidiFile()
    midi_obj.instruments.append(miditoolkit.Instrument(program=0))
    for note in note_list:
        midi_obj.instruments[0].notes.append(note)
    midi_obj.dump(savepath)

# section_grid
def get_section_grid(markers=None,max_tick = None):
    text ,start ,end = [], [], []
    start.append(markers[0].time)
    text.append(markers[0].text)
    for i in range(1,len(markers)):
        if markers[i].text != markers[i-1].text:
            text.append(markers[i].text)
            start.append(markers[i].time)
            end.append(markers[i].time)
    end.append(max_tick)

    assert (len(text) == len(start) and len(start) == len(end))

    new_marker = []
    for i in range(len(text)):
        new_marker.append(miditoolkit.Marker(text=text[i],time=start[i]))

    grid = start + end[-1:]

    return new_marker,grid

# section_grid
def get_phrase_grid(markers=None,max_tick = None):

    return [x.time for x in markers] + [max_tick]

# Split [Wikifornia Version]
def split_rpp_Wiki(midipath=None,ratio=0.5):
    if midipath == None:
        return
    m = RPP_Detection(midi_path=midipath)
    section_list = [m.all_steps()[0]]
    skeleton_note = []
    color_note = []

    # section
    for section in section_list:
        cnt_dic = rpp_cnt_dic(section)
        rpp_list = normalize(section=section)

        # rpp_theme
        rpp_theme = []
        max_cnt = 0
        for key,v in cnt_dic.items():
            if v > max_cnt:
                rpp_theme = list(key)
                max_cnt = v
        print('主题RPP:',rpp_theme)

        # similarity
        similarity = similarity_theme(rpp_theme=rpp_theme,rpp_list=rpp_list)

        # save note
        for i,simi in enumerate(similarity):
            if simi <= ratio or i==0:
                for note in section[i]:
                    skeleton_note.append(note)
            else:
                for note in section[i]:
                    color_note.append(note)


    return skeleton_note

# Split [Zhpop Version] 
def split_rpp_Zhpop(midipath=None):
    if midipath == None or midipath[-4:] != '.mid':
        return

    print('---------------------------------------\n',midipath)
    m = RPP_Detection(midi_path=midipath)
    midi_obj = miditoolkit.MidiFile(midipath)
    rest_threshold = compute_dynamic_rest_threshold(midi_obj)
    rest_threshold = compute_dynamic_rest_threshold(midi_obj)

    # get section_rpp list
    new_marker,section_grid = get_section_grid(midi_obj.markers,max_tick=midi_obj.max_tick)
    rpp_list,_ = m.all_steps()
    section = [[] for _ in range(len(new_marker))]

    for rpp in rpp_list:
        rpp_start = rpp[0].start
        rpp_end = rpp[-1].end
        for index,start,end in zip(range(len(new_marker)),section_grid[:-1],section_grid[1:]):
            if rpp_start>=start and rpp_end <= end:
                section[index].append(rpp)
                break
            elif rpp_start < start and rpp_end > start:
                if (rpp_start + rpp_end ) / 2 > start:
                    section[index].append(rpp)
                else:
                    section[index-1].append(rpp)
                break
            elif rpp_start < end and rpp_end > end:
                if (rpp_start + rpp_end ) / 2 < end:
                    section[index].append(rpp)
                else:
                    section[index+1].append(rpp)
                break

    # for i in range(len(new_marker)):
    #     print(new_marker[i])
    #     pprint.pprint(section[i])
    print('old_marker',midi_obj.markers)
    print('new_marker',new_marker)
    print('Section_grid: ',section_grid)

    rpp_need = []
    # split
    for sec in section:
        if len(sec) == 0:
            continue
        cnt_dic = rpp_cnt_dic(sec)
        rpp_list = normalize(section=sec)
        pprint.pprint(rpp_list)

        # rpp_theme
        rpp_theme = []
        max_cnt = 0
        for key,v in cnt_dic.items():
            if v > max_cnt:
                rpp_theme = list(key)
                max_cnt = v


        # rpp_need
        for i,rpp in enumerate(rpp_list):
            if rpp == rpp_theme:
                rpp_need.append(sec[i])

    # print(rpp_need)
    midi_obj.instruments.clear()
    new_track = miditoolkit.Instrument(program=0,name='melody')
    for rpp in rpp_need:
        for note in rpp:
            new_track.notes.append(note)
    midi_obj.instruments.append(new_track)

    out_path = midipath[:-4] + '_rpp.mid'
    midi_obj.dump(out_path)

    return

# Split_RP 
def split_RP_v1(midipath = None,write_log=True):
    total_cnt = 0

    if midipath == None or midipath[-4:] != '.mid':
        return

    print('---------------------------------------\n', midipath)
    midi_obj = miditoolkit.MidiFile(midipath)

    # rpp_list_raw
    rpp_list_raw = get_rpplist_raw(midipath=midipath)
    print('rpp_list_raw:')
    print(rpp_list_raw)

    # phrase
    phrase_grid = get_phrase_grid(midi_obj.markers,max_tick=midi_obj.max_tick)
    ## 240tick rest
    for i in range(len(rpp_list_raw)-1):
        pre_end = rpp_list_raw[i][-1].end
        aft_start = rpp_list_raw[i][0].start
        if aft_start - pre_end >=240:
            phrase_grid.append(pre_end)
    sorted(phrase_grid)
    print('phrase_grid:',phrase_grid)

    # RPP_list
    RPP_list = [RPP(rpp=x) for x in rpp_list_raw]
    # print('RPP_list:',RPP_list)

    # type_number & type_cnt
    type_cnt = collections.Counter([x.type for x in RPP_list])
    print('type_cnt: ',type_cnt)
    type_number = {key:number for number,key in enumerate(type_cnt.keys())}
    # print('type_number: ')
    # pprint.pprint(type_number)


    for Rpp in RPP_list:
        Rpp.cnt = type_cnt[Rpp.type]
        Rpp.number = type_number[Rpp.type]
    # print('RPP_list: ')
    # for rpp in RPP_list:
    #     print(rpp,'\n')


    phrase_RPP_raw = [[] for _ in range(len(phrase_grid)-1)]
    for rpp in RPP_list:
        rpp_start = rpp.start
        rpp_end = rpp.end
        for index, start, end in zip(range(len(phrase_grid)-1), phrase_grid[:-1], phrase_grid[1:]):
            if rpp_start>=start and rpp_end <= end:
                phrase_RPP_raw[index].append(rpp)
                break
            elif rpp_start < start and rpp_end > start:
                if (rpp_start + rpp_end ) / 2 > start:
                    phrase_RPP_raw[index].append(rpp)
                else:
                    phrase_RPP_raw[index-1].append(rpp)
                break
            elif rpp_start < end and rpp_end > end:
                if (rpp_start + rpp_end ) / 2 < end:
                    phrase_RPP_raw[index].append(rpp)
                else:
                    phrase_RPP_raw[index+1].append(rpp)
                break
    phrase_RPP = []
    for phrase in phrase_RPP_raw:
        if len(phrase) >= 1:
            phrase_RPP.append(phrase)
    # print('phrase_RPP:', phrase_RPP)

 
    phrase_number = []
    for phrase in phrase_RPP:
        phrase_number.append([Rpp.number for Rpp in phrase])
    print('phrase_number: ',phrase_number)

    # RP_cnt
    number_list = [rpp.number for rpp in RPP_list]
    print('number_list: ',number_list)

    RP_cnt = {}
    for i in range(len(number_list)-1):
        if (number_list[i],number_list[i+1]) in RP_cnt.keys():
            RP_cnt[(number_list[i],number_list[i+1])] += 1
        else:
            RP_cnt[(number_list[i],number_list[i+1])] = 1

    for i in range(len(number_list)-2):
        if (number_list[i],number_list[i+1],number_list[i+2]) in RP_cnt.keys():
            RP_cnt[(number_list[i],number_list[i+1],number_list[i+2])] += 1
        else:
            RP_cnt[(number_list[i],number_list[i+1],number_list[i+2])] = 1

    print('RP_cnt',RP_cnt)

    # each phrase -> RP
    RP_list = []
    for i,phrase in enumerate(phrase_number):
        part = []
        phrase_now = phrase[:]
        while(1):
            if len(phrase_now) == 4:
                part += [2,2]
                break
            if len(phrase_now) < 4 :
                part += [len(phrase_now)]
                break
            else:
                rp1 = (phrase_now[0],phrase_now[1])
                rp2 = (phrase_now[0], phrase_now[1],phrase_now[2])
                if RP_cnt[rp1] > RP_cnt[rp2]:
                    part.append(2)
                    phrase_now = phrase_now[2:]
                else :
                    part.append(3)
                    phrase_now = phrase_now[3:]

        # select RP
        RPP_now = phrase_RPP[i]
        index = 0
        for p in part:
            RP_list.append(RP(rpp_list=RPP_now[index:index+p]))

            # total_cnt
            if p == 2:
                total_cnt += RP_cnt[(RPP_now[index].number,RPP_now[index+1].number)]
            else:
                total_cnt += RP_cnt[(RPP_now[index].number, RPP_now[index+1].number,RPP_now[index+2].number)]

            index += p

    # print_RP(RP_list=RP_list)
    print('TOTAL_CNT=',total_cnt)

    # write_log
    if write_log :
        RPP_path = midipath[:-4] + '_RPP_V1.txt'
        RP_path = midipath[:-4] + '_RP_V1.txt'

        with open (RPP_path,'w') as f:
            for i,rpp in enumerate(RPP_list):
                print(f'--------------------------------- RPP {i} ---------------------------------',file=f)
                print(rpp,file=f)

        with open (RP_path,'w') as f:
            f.write(f'Total_cnt:{total_cnt}\n')
            for i,rp in enumerate(RP_list):
                print(f'--------------------------------- RP {i} ---------------------------------',file=f)
                print(rp, file=f)


    return RPP_list,RP_list

# Split_RP
def split_RP_v2(midipath = None,write_log=True):
    total_cnt = 0

    if midipath == None or midipath[-4:] != '.mid':
        return

    print('---------------------------------------\n', midipath)
    midi_obj = miditoolkit.MidiFile(midipath)

    # rpp_list_raw
    rpp_list_raw = get_rpplist_raw(midipath=midipath)


    # phrase
    phrase_grid = get_phrase_grid(midi_obj.markers,max_tick=midi_obj.max_tick)

    for i in range(len(rpp_list_raw)-1):
        pre_end = rpp_list_raw[i][-1].end
        aft_start = rpp_list_raw[i][0].start
        if aft_start - pre_end >=240:
            phrase_grid.append(pre_end)
    sorted(phrase_grid)
    print('phrase_grid:',phrase_grid)

    # RPP_list
    RPP_list = [RPP(rpp=x) for x in rpp_list_raw]
    # print('RPP_list:', RPP_list)


    type_cnt = collections.Counter([x.type for x in RPP_list])
    print('type_cnt: ', type_cnt)
    type_number = {key: number for number, key in enumerate(type_cnt.keys())}
    # print('type_number: ')
    # pprint.pprint(type_number)


    for Rpp in RPP_list:
        Rpp.cnt = type_cnt[Rpp.type]
        Rpp.number = type_number[Rpp.type]
    # print('RPP_list: ', RPP_list)


    phrase_RPP_raw = [[] for _ in range(len(phrase_grid) - 1)]
    for rpp in RPP_list:
        rpp_start = rpp.start
        rpp_end = rpp.end
        for index, start, end in zip(range(len(phrase_grid) - 1), phrase_grid[:-1], phrase_grid[1:]):
            if rpp_start >= start and rpp_end <= end:
                phrase_RPP_raw[index].append(rpp)
                break
            elif rpp_start < start and rpp_end > start:
                if (rpp_start + rpp_end) / 2 > start:
                    phrase_RPP_raw[index].append(rpp)
                else:
                    phrase_RPP_raw[index - 1].append(rpp)
                break
            elif rpp_start < end and rpp_end > end:
                if (rpp_start + rpp_end) / 2 < end:
                    phrase_RPP_raw[index].append(rpp)
                else:
                    phrase_RPP_raw[index + 1].append(rpp)
                break
    phrase_RPP = []
    for phrase in phrase_RPP_raw:
        if len(phrase) >= 1:
            phrase_RPP.append(phrase)



    phrase_number = []
    for phrase in phrase_RPP:
        phrase_number.append([Rpp.number for Rpp in phrase])
    print('phrase_number: ', phrase_number)

    # RP_cnt
    number_list = [rpp.number for rpp in RPP_list]
    print('number_list: ', number_list)

    RP_cnt = {}
    for i in range(len(number_list) - 1):
        if (number_list[i], number_list[i + 1]) in RP_cnt.keys():
            RP_cnt[(number_list[i], number_list[i + 1])] += 1
        else:
            RP_cnt[(number_list[i], number_list[i + 1])] = 1

    for i in range(len(number_list) - 2):
        if (number_list[i], number_list[i + 1], number_list[i + 2]) in RP_cnt.keys():
            RP_cnt[(number_list[i], number_list[i + 1], number_list[i + 2])] += 1
        else:
            RP_cnt[(number_list[i], number_list[i + 1], number_list[i + 2])] = 1

    print('RP_cnt', RP_cnt)

    # each phrase -> RP
    RP_list = []

    for i, phrase in enumerate(phrase_number):
        # initial
        part = [[] for _ in range(len(phrase))]
        dp = [0] * len(phrase) # dp = [0,2,2,3,4...]

        dp[1] = RP_cnt[(phrase[0],phrase[1])]
        part[1].append(2)
        if len(phrase) >= 3:
            dp[2] = RP_cnt[(phrase[0],phrase[1],phrase[2])]
            part[2].append(3)
        if len(phrase) >= 4:
            dp[3] = dp[1] + RP_cnt[(phrase[2],phrase[3])]
            part[3].extend([2,2])

        # loop
        for j in range(4,len(phrase)):
            cnt_contain2 = dp[j-2] + RP_cnt[(phrase[j-1],phrase[j])]
            cnt_contain3 = dp[j-3] + RP_cnt[(phrase[j-2],phrase[j-1],phrase[j])]

            if cnt_contain2 > cnt_contain3:
                dp[j] = cnt_contain2
                part[j].extend(part[j-2])
                part[j].extend([2])
            else:
                dp[j] = cnt_contain3
                part[j].extend(part[j - 3])
                part[j].extend([3])

        part = part[-1]
        print('---------------------------- DP ------------------------------')
        print('dp:', dp)
        print('part:', part)

        # select RP
        RPP_now = phrase_RPP[i]
        index = 0
        for p in part:
            RP_list.append(RP(rpp_list=RPP_now[index:index + p]))
            index += p

        # total_cnt
        total_cnt += dp[-1]

    # print_RP(RP_list=RP_list)
    print('TOTAL_CNT=',total_cnt)

    # write_log
    if write_log :
        RPP_path = midipath[:-4] + '_RPP_V2.txt'
        RP_path = midipath[:-4] + '_RP_V2.txt'

        with open (RPP_path,'w') as f:
            for i,rpp in enumerate(RPP_list):
                print(f'--------------------------------- RPP {i} ---------------------------------',file=f)
                print(rpp,file=f)

        with open (RP_path,'w') as f:
            f.write(f'Total_cnt:{total_cnt}\n')
            for i,rp in enumerate(RP_list):
                print(f'--------------------------------- RP {i} ---------------------------------',file=f)
                print(rp, file=f)


    return RPP_list, RP_list


# RPP_ryhthm_structrue
def get_rpp_rhythm_structure(token_name,rpp_ryhthm_list,i):

    rpp_ryhthm = rpp_ryhthm_list[i]

    type = '0'
    if rpp_ryhthm == [0,1]:
        type = '0'
    elif rpp_ryhthm == [0,0,1]:
        type = '1'
    elif rpp_ryhthm == [1,0]:
        type = '2'
    elif rpp_ryhthm == [1,0,0]:
        type = '3'
    elif rpp_ryhthm == [0,1,0]:
        type = '4'

    parts = token_name.split('_', 2)
    if len(parts) < 3:
        return token_name
    parts[1] = type
    return '_'.join(parts)

# RPP_melody_contour
def get_rpp_melody_contour(note_list=None):

    type = '0'
    dict = {(-1):'0',(1):'1',(0):'2',(-1,-1):'3',(1,1):'4',(0,0):'5',(-1,1):'6',(1,-1):'7',(0,1):'8',(0,-1):'9',(-1,0):'10',(1,0):'11'}

    if len(note_list) == 1:
        return '12'
    if len(note_list) == 2:
        diff = note_list[1].pitch - note_list[0].pitch
        diff = 1 if diff>0 else ( -1 if diff<0 else 0)
        type = dict[(diff)]
    elif len(note_list) == 3:
        diff1 = note_list[1].pitch-note_list[0].pitch
        diff2 = note_list[2].pitch-note_list[1].pitch
        diff1 = 1 if diff1>0 else ( -1 if diff1<0 else 0)
        diff2 = 1 if diff2>0 else ( -1 if diff2<0 else 0)
        type = dict[(diff1,diff2)]

    return int(type)

# RPP_relation
def get_rpp_relation(rpp_1,rpp_2):

    dict = {(-1,-1):2,(1,1):3,(1,-1):4,(1,0):4,(0,-1):4,(-1,1):5,(-1,0):5,(0,1):5,(0,0):6}

    high_1 = max([x.pitch for x in rpp_1])
    low_1 = min([x.pitch for x in rpp_1])
    high_2 = max([x.pitch for x in rpp_2])
    low_2 = min([x.pitch for x in rpp_2])

    diff1 = 1 if high_1>high_2 else (-1 if high_1<high_2 else 0)
    diff2 = 1 if low_1>low_2 else (-1 if low_1<low_2 else 0)

    return dict[(diff1,diff2)]

def graph_token_RPP_only(midipath = None,rpp_outdir=None,rpp_list=None):    # return -> rpp_seq , vertex , edge

    # default contaiiner
    rpp_sequence = [] 
    edge = [] 

    

    if rpp_outdir == None:
        outpath = None
    else:
        outpath = os.path.join(rpp_outdir,os.path.basename(midipath))

    if midipath != None:
        note_list,rpp_list = rpp_divider(midipath=midipath,outpath=outpath,algorithm='DP')
    else:
        rpp_list = rpp_list
    node_list = ['SOS']

    for i,rpp in enumerate(rpp_list):
        if i == 0 and rpp.start >= 240:     
            node_list.append('REST')
        node_list.append(rpp)
        if i+1 <= len(rpp_list)-1 :
            if rpp_list[i+1].start - rpp.end >=240:
                node_list.append('REST')
    node_list.append('EOS')

    # [2] loop -> info2edge
    for i,node in enumerate(node_list):
        #  EOS SOS
        if isinstance(node,str):
            if node == 'SOS':
                continue
            elif node=='EOS':
                edge.append((i,i-1,1))
            elif node=='REST':
                edge.append((i,i-1,1))

        # RPP
        if isinstance(node,RPP):
            # 1 note
            if len(node.rpp) == 1:
                edge.append((i, i - 1, 1))

            # 2/3 note
            else:
                
                position_edge = (i,i-1,1)
                
                similarity_edge = None
                for j in range(i-1,0,-1):
                    if isinstance(node_list[j],RPP) and node.rhythm_pattern == node_list[j].rhythm_pattern:
                        
                        offset = node.start - node_list[j].start
                        n = offset // 120
                        if n & (n - 1) == 0:
                            relation = get_rpp_relation(node.rpp, node_list[j].rpp)
                            if node.melody_contour!=node_list[j].melody_contour:
                                relation += 5
                            similarity_edge = (i,j,relation)
                            break

                
                if similarity_edge == None:
                    edge.append(position_edge)
                else:
                    if similarity_edge[1] == i-1:    
                        edge.append(similarity_edge)
                    else:
                        edge.append(position_edge)
                        edge.append(similarity_edge)

    vertex = []
    for node in node_list:
        if isinstance(node,str):
            vertex.append(node)
        else:
            vertex.append(node.token_name)

    return node_list,vertex,edge

def graph_token(midipath = None):
    midi_obj = miditoolkit.MidiFile(midipath)
    sequence = [] 
    token = [] #[(1,1,0), (1,2,3),....]

    # P grid
    P_list = []
    phrase_grid = get_phrase_grid(midi_obj.markers,max_tick=midi_obj.max_tick)
    for start,end in zip(phrase_grid[:-1],phrase_grid[1:]):
        P_list.append(P(start = start,end=end))

    # add RP to P
    RPP_list,RP_list = split_RP_v1(midipath,write_log=False)
    print('RPP_list',RPP_list)
    print('RP_list',RP_list)

    for rp in RP_list:
        rp_start = rp.start
        rp_end = rp.end
        for p in P_list:
            if rp_start >= p.start and rp_start <= p.end:
                p.rp_list.append(rp)
                break

    print('P_list',P_list)
    
    for p in P_list:
        sequence.append(p)
        for rp in p.rp_list:
            sequence.append(rp)
            for rpp in rp.rpp_list:
                sequence.append(rpp)

    for i,each in enumerate(sequence):
        each.mark = i

    # fill token
    token.append([0,0,0])
    for each in sequence[1:]:
        if each.sign == 'P':
            token.append([each.mark,each.mark,2])
        elif each.sign == 'RP':
            token.append([each.mark,0,1]) 
            token.append([each.mark,each.mark+1,3])
        else:
            token.append([each.mark,0,1]) 
            token.append([each.mark,each.mark+1,4])




    return token

def print_graph(vertex,edge,file_index,outpath):
    with open(outpath, 'a') as f:
        f.write(f'#{file_index}\n')
        f.write(f'{len(vertex)}\n')
        for v in vertex:
            f.write(f'{v}\n')
        f.write(f'{len(edge)}\n')
        for e in edge:
            f.write(f'{e[0]} {e[1]} {e[2]}\n')
        f.write('\n')

def print_graph_check(seq,vertex,edge,filepath,file_index,input_file):

    edge_new = [[x[1],x[0],x[2]] for x in edge]
    edge_new.sort()

    with open(filepath,'a') as f:
        f.write(f'#{file_index} {input_file}\n')

        f.write(f'节点数量:{len(vertex)}\n')

        bar_vertex = 1
        for i,v in enumerate(vertex):
            if v == 'SOS':
                f.write(f'----------- Bar{bar_vertex:<2} -----------\n')
                f.write(f'0  :SOS\n')
            elif v == 'EOS':
                f.write(f'{i:<3}:EOS\n')
            elif v == 'REST':
                f.write(f'{i:<3}:REST\n')
            else:
                rpp = seq[i]
                cur_bar = rpp.start // 1920 + 1
                if cur_bar > bar_vertex :
                    bar_vertex = cur_bar
                    f.write(f'----------- Bar{bar_vertex:<2} -----------\n')
                f.write(f'{i:<3}:{v:<8}({len(rpp.rpp)})\n')

        f.write(f'边数量:{len(edge)}\n')
        bar_edge = 0

        for i,e in enumerate(edge_new):
            if not isinstance(seq[e[0]],str) and not isinstance(seq[e[1]],str):
                rpp = seq[e[0]]
                cur_bar = rpp.start // 1920 +1
                if cur_bar > bar_edge:
                    bar_edge = cur_bar
                    f.write(f'----------- Bar{bar_edge:2} -----------\n')
                f.write(f'{e[0]:<3}:{seq[e[0]].token_name:<8}  {e[1]:<3}:{seq[e[1]].token_name:<8}  {e[2]}\n')

        print('\n')

# RPP_rythm_type
def note_ryhthm_weight(note_list):
    weight_dict = {0:5,1:1,2:2,3:1,4:3,5:1,6:2,7:1,8:4,9:1,10:2,11:1,12:3,13:1,14:2,15:1}

    
    def process_synpogation(p,q):
        if p==q:
            
            note_list[p].rythm_weight = 5 if note_list[p].is_skeleton else 4
        else:
            
            if len(set([x.duration for x in note_list[p:q+1]])) == 1:        # 如果duration都相同
                cur_note_list = note_list[p:q+1]
                cur_start = cur_note_list[0].start
                for note in cur_note_list:
                    note.rythm_weight = weight_dict[(note.start-cur_start)%1920*16//1920]
            else:                                                       # 如果duration不都相同
                index_list = list(range(p,q+1))
                index_list.sort(key=lambda x:(-(note_list[x].end-note_list[x].start),note_list[x].start))
                
                score = 6
                dur = -1
                for i,index in enumerate(index_list):
                    if note_list[index].duration != dur:
                        score = max(score-1,1)
                        dur = note_list[index].duration
                        note_list[index].rythm_weight = score
                    else:
                        if note_list[index].duration == note_list[index-1].duration:
                            note_list[index].rythm_weight = max(1,note_list[index-1].rythm_weight-1)
                        else:
                            note_list[index].rythm_weight = score

            
            if p==0 or note_list[p-1].rythm_weight!=-1:
                return
            else:
                offset = 1920 - (note_list[p].start - note_list[p-1].start)
                note_list[p-1].rythm_weight = weight_dict[offset%1920*16//1920]


    active,start,end = False,-1,-1
    for i,note in enumerate(note_list):

        if note.is_syncopation :        
            if active:                       
                if note.start - note_list[i-1].end>=240: 
                    process_synpogation(p=start,q=end)
                    start,end = i,i
                else:                                   
                    end = i
            else:                            
                active = True
                start,end = i,i
        else:                           
            if active:                      
                process_synpogation(p=start, q=end)
                active,start,end = False,-1,-1
            else:                           
                continue


    for note in note_list:
        if note.rythm_weight == -1:

            note.rythm_weight = weight_dict[note.start%1920*16//1920]


    for note in note_list:
        note.rythm_weight += note.duration //120


def rpp_split_accuracy(rawroot,tgtroot,logpath=None):
    log = ''
    file_raw = [f for f in os.listdir(rawroot) if f[-4:] == '.mid']
    file_tgt = [f for f in os.listdir(tgtroot) if f[-4:] == '.mid']

    def get_divline(note_list):  
        divline = []

        for i,note in enumerate(note_list):
            if i == 0:
                divline.append(note.start)
            else:
                if note.velocity != note_list[i-1].velocity:
                    divline.append(note.start)
        return divline

    for f in file_raw:
        if f not in file_tgt:
            continue
        else:
            midi_obj1 = miditoolkit.MidiFile(os.path.join(rawroot,f))
            midi_obj2 = miditoolkit.MidiFile(os.path.join(tgtroot,f))

            divline1 = get_divline(note_list=midi_obj1.instruments[0].notes)
            divline2 = get_divline(note_list=midi_obj2.instruments[0].notes)

            samediv = [x for x in divline1 if x in divline2]

            accuracy = round(len(samediv)/len(divline1),2)

            log += f'\n --------------------- {f} --------------------- \n'
            log += f'Accuracy: {accuracy} \n'

    if logpath != None:
        with open(logpath,'a') as f:
            f.write(log)


def rpp_divider(midipath,outpath=None,need_log = False,algorithm=None):
    if algorithm is None:
        algorithm = GLOBAL_ALGORITHM

    # Default Container
    rpp_all = []
    note_all = []
    rpp2area = collections.defaultdict(int)    # {(rpp.start,rpp.end) : area}
    rpp_log = f'\n-------------------- {midipath} ---------------------- \n'    #记录分割结果
    weight_log = f'\n-------------------- {midipath} ---------------------- \n' #记录rpp覆盖面积
    note_log = f'\n-------------------- {midipath} ---------------------- \n'    #记录音符赋值


    midi_obj = miditoolkit.MidiFile(midipath)
    rest_threshold = compute_dynamic_rest_threshold(midi_obj)
    m = RPP_Detection(midi_path=midipath)  
    skeleton_bool, syncopation_bool = m.get_note_typeof_skeleton_syncopation()

    assert len(midi_obj.instruments[0].notes) == len(skeleton_bool) and len(skeleton_bool) == len(syncopation_bool),\
        f'{midipath}音符数量解析不对等'


    for i in range(len(skeleton_bool)):
        true_note = midi_obj.instruments[0].notes[i]
        info = (true_note.velocity,true_note.pitch,true_note.start,true_note.end,skeleton_bool[i],syncopation_bool[i])
        note_all.append(Note(*info))


    note_ryhthm_weight(note_list=note_all)
    note_all.sort(key= lambda x:x.start)

    
    rpp4detect = []
    for i in range(len(note_all)):
        if i+1<=len(note_all)-1:
            if note_all[i+1].start <= note_all[i].end:
                rpp4detect.append(RPP(rpp=note_all[i:i+2]))
        if i+2<=len(note_all)-1:
            if note_all[i+1].start <= note_all[i].end and note_all[i+2].start <= note_all[i+1].end:
                rpp4detect.append(RPP(rpp=note_all[i:i+3]))

    shape2kind = collections.defaultdict(int)       # { ((0,0,1),1) : 5 }
    
    for i,rpp in enumerate(rpp4detect):
        find = False
        for j in range(i-1,-1,-1):
            if rpp4detect[j].shape == rpp4detect[i].shape:
                offset = rpp4detect[i].start - rpp4detect[j].start
                n = offset // 120
                if n & (n - 1) == 0 and rpp4detect[i].start>=rpp4detect[j].end:       
                    find = True
                    rpp4detect[i].shape_and_kind = rpp4detect[j].shape_and_kind
                    break
                else:                       
                    continue
        if find == False:
            shape2kind[rpp.shape] += 1
            rpp.shape_and_kind = (rpp.shape,shape2kind[rpp.shape])
    
    shape2area = collections.defaultdict(int)      #{ (((0, 1, 0), 5), 1) : 12}
    for rpp in rpp4detect:
        if len(rpp.rpp) == 3:
            shape2area[rpp.shape_and_kind] += len(rpp.rpp) * 1.4
        else:
            shape2area[rpp.shape_and_kind] += len(rpp.rpp)

   
    



    for rpp in rpp4detect:
        rpp2area[(rpp.start,rpp.end)] = shape2area[rpp.shape_and_kind]


    # [2]div_grid
    div_grid = []
    for i,note in enumerate(midi_obj.instruments[0].notes):
        if i == 0:
            div_grid.append(note.start)
        elif i == len(midi_obj.instruments[0].notes) - 1:
            div_grid.append(note.end)
            break
        else:
            if note.start - midi_obj.instruments[0].notes[i-1].end >= rest_threshold:
                div_grid.append(note.start)
    div_skeleton = m.get_div_skeleton()


    for note in div_skeleton[:]:
        dur = note.end - note.start
        bar_start = note.start//BAR_TICKS*BAR_TICKS
        bar_end = bar_start+BAR_TICKS
        dur_list = [(x.end-x.start) for x in note_all if (x.start>=bar_start and x.start<bar_end and x.start!=note.start)]

        if len(dur_list) != 0 and dur <= max(dur_list):
            div_skeleton.remove(note)
            continue
        if not is_strong_skeleton(note):
            div_skeleton.remove(note)


    break_boundaries = []
    for note in div_skeleton:
        break_note = assign_break_note(note_all, note)
        break_boundaries.append(break_note.end)

    for end_time in break_boundaries:
        div_grid.append(end_time)
    div_grid.sort()
    need_check = True
    while(need_check):
        need_check = False
        for i, (start, end) in enumerate(zip(div_grid[:-1], div_grid[1:])):
            note_find = False
            for note in note_all:
                if note.start>=start and note.end<=end:
                    note_find = True
                    break
            if note_find == False:
                div_grid.remove(start)
                need_check = True
                break
            else:
                continue


    # [3]sort note -> phrase
    segment_bounds = list(zip(div_grid[:-1], div_grid[1:]))
    phrase = [[] for _ in segment_bounds]
    for note in note_all:
        idx = bisect_right(div_grid, note.start) - 1
        if idx < 0 or idx >= len(segment_bounds):
            continue
        start, end = segment_bounds[idx]
        if note.start >= start and note.end <= end:
            phrase[idx].append(note)
    phrase, div_grid = smooth_phrase_segments(phrase, div_grid)
    
    # marker aligned with smoothed phrases
    midi_obj.markers.clear()
    for i,div in enumerate(div_grid):
        midi_obj.markers.append(miditoolkit.Marker(text=f'Div{i}',time=div))

    # [4]proccess each phrase 
    if algorithm == 'DP':
        for i,ph in enumerate(phrase):
            rpp_log += f'\n### Div{i} BEGIN ###\n'

            
            if len(ph) <= 3:
                r = RPP(rpp=ph)
                r.phrase_index = i
                rpp_all.append(r)
                rpp_log+=f'{r} Area:{rpp2area[(r.start,r.end)]}\n'
                continue

            
            ## 0
            dp = []
            part = []
            dp.append(1)
            part.append([1])
            ## 1
            initial_rpp_0_1 = RPP(rpp=[ph[0],ph[1]])
            area2 = rpp2area[(initial_rpp_0_1.start,initial_rpp_0_1.end)]
            area1 = dp[0] + 1
            if area2 >= area1:
                dp.append(area2)
                part.append([2])
            else:
                dp.append(area1)
                part.append([1,1])
            ## 2
            initial_rpp_1_2 = RPP(rpp=[ph[1],ph[2]])
            initial_rpp_0_1_2 = RPP(rpp = [ph[0],ph[1],ph[2]])
            area1 = dp[1] + 1
            area2 = dp[0] + rpp2area[(initial_rpp_1_2.start,initial_rpp_1_2.end)]
            area3 = rpp2area[(initial_rpp_0_1_2.start,initial_rpp_0_1_2.end)]
            if area3>= area2 and area3 >= area1:
                dp.append(area3)
                part.append([3])
            elif area2>=area3 and area2>=area1:
                dp.append(area2)
                part.append([1,2])
            else:
                dp.append(area1)
                part.append(part[1]+[1])

            for i in range(3,len(ph)):
                cur_rpp2 = RPP([ph[i-1],ph[i]])
                cur_rpp3 = RPP([ph[i-2],ph[i-1],ph[i]])
                area1 = dp[i-1] + 1
                area2 = dp[i-2] + rpp2area[(cur_rpp2.start,cur_rpp2.end)]
                area3 = dp[i-3] + rpp2area[(cur_rpp3.start,cur_rpp3.end)]
                if area3 >= max(area1,area2):
                    dp.append(area3)
                    part.append(part[i-3]+[3])
                elif area2 >= max(area1,area3):
                    dp.append(area2)
                    part.append(part[i-2]+[2])
                else:
                    dp.append(area1)
                    part.append(part[i-1]+[1])

            
            Part= part[-1]
            index = 0
            segments = []
            for p in Part:
                segments.append(ph[index:index+p])
                index += p
            segments = merge_single_note_segments(segments)
            for segment in segments:
                r = RPP(segment)
                r.phrase_index = i
                rpp_all.append(r)
                key = (segment[0].start, segment[-1].end)
                area = 1 if len(segment) == 1 else rpp2area.get(key, len(segment))
                rpp_log+=f'{r} Area:{area}\n'

    elif algorithm == 'RANDOM':
        for i, ph in enumerate(phrase):
            rpp_log += f'\n### Div{i} BEGIN ###\n'
            
            
            if len(ph) <= 3:
                r = RPP(rpp=ph)
                r.phrase_index = i
                rpp_all.append(r)
                key = (ph[0].start, ph[-1].end)
                area = 1 if len(ph) == 1 else rpp2area.get(key, len(ph))
                rpp_log += f'{r} Area:{area}\n'
                continue
            
            part = []
            remaining = len(ph)
            while remaining > 0:
                choice = int(np.random.choice([1, 2, 3], p=[0.12, 0.36, 0.52]))#the rate of 1,2,3 note RPP . According to the statistics on your dataset
                choice = min(choice, remaining)
                part.append(choice)
                remaining -= choice
            
           
            index = 0
            segments = []
            for p in part:
                segments.append(ph[index:index+p])
                index += p
            
            segments = merge_single_note_segments(segments)
            for segment in segments:
                r = RPP(segment)
                r.phrase_index = i
                rpp_all.append(r)
                key = (segment[0].start, segment[-1].end)
                area = 1 if len(segment) == 1 else rpp2area.get(key, len(segment))
                rpp_log += f'{r} Area:{area}\n'

    key_info = detect_song_key(midi_obj, note_all)
    annotate_cadence_tags(rpp_all, key_info)

    for i,each in enumerate(rpp_all):
        if i == 0:
            for note in each.rpp:
                note.velocity = 60
        else:
            velo = 60
            if rpp_all[i-1].rpp[0].velocity == 60:
                velo = 120
            for note in each.rpp:
                note.velocity = velo

    
    midi_obj.instruments[0].notes.clear()
    for each in rpp_all:
        for note in each.rpp:
            midi_obj.instruments[0].notes.append(note)

    # [8]log
    if need_log == True:
        path = outpath if outpath != None else midipath
        rpp_log_path = os.path.join(os.path.dirname(path),'rpp_log.txt')
        note_log_path = os.path.join(os.path.dirname(path),'note_log.txt')
        weight_log_path = os.path.join(os.path.dirname(path),'weight_log.txt')

        # rpp_log
        with open(rpp_log_path,'a') as f:
            f.write(rpp_log)

        # note_log
        for i,ph in enumerate(phrase):
            note_log +=f'\n### Div{i} ###\n'
            for note in ph:
                note_log += f'{note}'
        with open(note_log_path,'a') as f:
            f.write(note_log)

        # weight_log
        for i, ph in enumerate(phrase):
            weight_log+= f'\n### Div{i} ###\n'

            for rpp in rpp4detect:
                if rpp.start>=ph[0].start and rpp.end <=ph[-1].end:
                    weight_log += f'{rpp} Area:{rpp2area[(rpp.start, rpp.end)]:<2} Shape:{str(rpp.shape)}\n'
        with open(weight_log_path,'a') as f:
            f.write(weight_log)


    if outpath != None:
        midi_obj.dump(outpath)

    return note_all,rpp_all

def note_weight(midipath):

    # Default Container
    note_all = []

    
    midi_obj = miditoolkit.MidiFile(midipath)
    rest_threshold = compute_dynamic_rest_threshold(midi_obj)
    m = RPP_Detection(midi_path=midipath)
    skeleton_bool, syncopation_bool = m.get_note_typeof_skeleton_syncopation()

    assert len(midi_obj.instruments[0].notes) == len(skeleton_bool) and len(skeleton_bool) == len(syncopation_bool),\
        f'{midipath}音符数量解析不对等'

    for i in range(len(skeleton_bool)):
        true_note = midi_obj.instruments[0].notes[i]
        info = (true_note.velocity,true_note.pitch,true_note.start,true_note.end,skeleton_bool[i],syncopation_bool[i])
        note_all.append(Note(*info))


    note_ryhthm_weight(note_list=note_all)
    note_all.sort(key= lambda x:x.start)
    weight_list = [note.rythm_weight for note in note_all]

    return note_all,weight_list


def rpp_divider_contain_2_3_notes(midipath,outpath=None,need_log = False,div_need=False,algorithm=None):
    if algorithm is None:
        algorithm = GLOBAL_ALGORITHM

    # Default Container
    rpp_all = []
    note_all = []
    rpp2area = collections.defaultdict(int)    # {(rpp.start,rpp.end) : area}
    rpp_log = f'\n-------------------- {midipath} ---------------------- \n'    #记录分割结果
    weight_log = f'\n-------------------- {midipath} ---------------------- \n' #记录rpp覆盖面积
    note_log = f'\n-------------------- {midipath} ---------------------- \n'    #记录音符赋值


    midi_obj = miditoolkit.MidiFile(midipath)
    rest_threshold = compute_dynamic_rest_threshold(midi_obj)
    m = RPP_Detection(midi_path=midipath)
    skeleton_bool, syncopation_bool = m.get_note_typeof_skeleton_syncopation()

    assert len(midi_obj.instruments[0].notes) == len(skeleton_bool) and len(skeleton_bool) == len(syncopation_bool),\
        f'{midipath}音符数量解析不对等'


    for i in range(len(skeleton_bool)):
        true_note = midi_obj.instruments[0].notes[i]
        info = (true_note.velocity,true_note.pitch,true_note.start,true_note.end,skeleton_bool[i],syncopation_bool[i])
        note_all.append(Note(*info))


    note_ryhthm_weight(note_list=note_all)
    note_all.sort(key= lambda x:x.start)

   
    rpp4detect = []
    for i in range(len(note_all)):
        if i+1<=len(note_all)-1:
            if note_all[i+1].start <= note_all[i].end:
                rpp4detect.append(RPP(rpp=note_all[i:i+2]))
        if i+2<=len(note_all)-1:
            if note_all[i+1].start <= note_all[i].end and note_all[i+2].start <= note_all[i+1].end:
                rpp4detect.append(RPP(rpp=note_all[i:i+3]))

    shape2kind = collections.defaultdict(int)       # { ((0,0,1),1) : 5 }
    for i,rpp in enumerate(rpp4detect):
        find = False
        for j in range(i-1,-1,-1):
            if rpp4detect[j].shape == rpp4detect[i].shape:
                offset = rpp4detect[i].start - rpp4detect[j].start
                n = offset // 120
                if n & (n - 1) == 0 and rpp4detect[i].start>=rpp4detect[j].end:        
                    find = True
                    rpp4detect[i].shape_and_kind = rpp4detect[j].shape_and_kind
                    break
                else:                       
                    continue
        if find == False:
            shape2kind[rpp.shape] += 1
            rpp.shape_and_kind = (rpp.shape,shape2kind[rpp.shape])

    shape_kind2area = collections.defaultdict(int)      #{ (((0, 1, 0), 5), 1) : 12}
    for rpp in rpp4detect:
        shape_kind2area[rpp.shape_and_kind] += len(rpp.rpp)

    for rpp in rpp4detect:
        rpp2area[(rpp.start,rpp.end)] = shape_kind2area[rpp.shape_and_kind]

    print(shape_kind2area)

    # [2]div_grid
    if div_need:
        div_grid = []
        for i,note in enumerate(midi_obj.instruments[0].notes):
            if i == 0:
                div_grid.append(note.start)
            elif i == len(midi_obj.instruments[0].notes) - 1:
                div_grid.append(note.end)
                break
            else:
                if note.start - midi_obj.instruments[0].notes[i-1].end >= rest_threshold:
                    div_grid.append(note.start)
        div_skeleton = m.get_div_skeleton()

        
        for note in div_skeleton[:]:
            dur = note.end - note.start
            bar_start = note.start//BAR_TICKS*BAR_TICKS
            bar_end = bar_start+BAR_TICKS
            dur_list = [(x.end-x.start) for x in note_all if (x.start>=bar_start and x.start<bar_end and x.start!=note.start)]

            if len(dur_list) != 0 and dur <= max(dur_list):
                div_skeleton.remove(note)
                continue
            if not is_strong_skeleton(note):
                div_skeleton.remove(note)

        
        break_boundaries = []
        for note in div_skeleton:
            break_note = assign_break_note(note_all, note)
            break_boundaries.append(break_note.end)

        for end_time in break_boundaries:
            div_grid.append(end_time)
        div_grid.sort()
        need_check = True
        while(need_check):
            need_check = False
            for i, (start, end) in enumerate(zip(div_grid[:-1], div_grid[1:])):
                note_find = False
                for note in note_all:
                    if note.start>=start and note.end<=end:
                        note_find = True
                        break
                if note_find == False:
                    div_grid.remove(start)
                    need_check = True
                    break
                else:
                    continue

        # [3]sort note -> phrase
        segment_bounds = list(zip(div_grid[:-1], div_grid[1:]))
        phrase = [[] for _ in segment_bounds]
        for note in note_all:
            idx = bisect_right(div_grid, note.start) - 1
            if idx < 0 or idx >= len(segment_bounds):
                continue
            start, end = segment_bounds[idx]
            if note.start >= start and note.end <= end:
                phrase[idx].append(note)
        phrase, div_grid = smooth_phrase_segments(phrase, div_grid)
        

        midi_obj.markers.clear()
        for i,div in enumerate(div_grid):
            midi_obj.markers.append(miditoolkit.Marker(text=f'Div{i}',time=div))

    else:
        phrase = [note_all]

    # [5]proccess each phrase 
    if algorithm == 'DP':
        for i,ph in enumerate(phrase):
            rpp_log += f'\n### Div{i} BEGIN ###\n'

            #
            if len(ph) <= 3:
                r = RPP(rpp=ph)
                rpp_all.append(r)
                rpp_log+=f'{r} Area:{rpp2area[(r.start,r.end)]}\n'
                continue


            ## 0
            dp = []
            part = []
            dp.append(-1000)
            part.append([1])
            ## 1
            initial_rpp_0_1 = RPP(rpp=[ph[0],ph[1]])
            area2 = rpp2area[(initial_rpp_0_1.start,initial_rpp_0_1.end)]
            dp.append(area2)
            part.append([2])
            ## 2
            initial_rpp_0_1_2 = RPP(rpp = [ph[0],ph[1],ph[2]])
            area3 = rpp2area[(initial_rpp_0_1_2.start,initial_rpp_0_1_2.end)]
            dp.append(area3)
            part.append([3])

            for i in range(3,len(ph)):
                cur_rpp2 = RPP([ph[i-1],ph[i]])
                cur_rpp3 = RPP([ph[i-2],ph[i-1],ph[i]])
                area2 = dp[i-2] + rpp2area[(cur_rpp2.start,cur_rpp2.end)]
                area3 = dp[i-3] + rpp2area[(cur_rpp3.start,cur_rpp3.end)]
                if area3 >= area2:
                    dp.append(area3)
                    part.append(part[i-3]+[3])
                else:
                    dp.append(area2)
                    part.append(part[i-2]+[2])


            Part= part[-1]
            index = 0
            segments = []
            for p in Part:
                segments.append(ph[index:index+p])
                index += p
            segments = merge_single_note_segments(segments)
            for segment in segments:
                r = RPP(segment)
                r.phrase_index = i
                rpp_all.append(r)

                key = (segment[0].start, segment[-1].end)
                area = 1 if len(segment) == 1 else rpp2area.get(key, len(segment))
                rpp_log+=f'{r} Area:{area}\n'

    elif algorithm == 'RANDOM':
        for i, ph in enumerate(phrase):
            rpp_log += f'\n### Div{i} BEGIN ###\n'
            
            
            if len(ph) <= 3:
                r = RPP(rpp=ph)
                r.phrase_index = i
                rpp_all.append(r)
                key = (ph[0].start, ph[-1].end)
                area = 1 if len(ph) == 1 else rpp2area.get(key, len(ph))
                rpp_log += f'{r} Area:{area}\n'
                continue
            
            part = []
            remaining = len(ph)
            while remaining > 0:
                choice = int(np.random.choice([1, 2, 3], p=[0.12, 0.36, 0.52]))
                choice = min(choice, remaining)
                part.append(choice)
                remaining -= choice
            
           
            index = 0
            segments = []
            for p in part:
                segments.append(ph[index:index+p])
                index += p
            
            segments = merge_single_note_segments(segments)
            for segment in segments:
                r = RPP(segment)
                r.phrase_index = i
                rpp_all.append(r)
                key = (segment[0].start, segment[-1].end)
                area = 1 if len(segment) == 1 else rpp2area.get(key, len(segment))
                rpp_log += f'{r} Area:{area}\n'

    key_info = detect_song_key(midi_obj, note_all)
    annotate_cadence_tags(rpp_all, key_info)


    for i,each in enumerate(rpp_all):
        if i == 0:
            for note in each.rpp:
                note.velocity = 60
        else:
            velo = 60
            if rpp_all[i-1].rpp[0].velocity == 60:
                velo = 120
            for note in each.rpp:
                note.velocity = velo

   
    midi_obj.instruments[0].notes.clear()
    for each in rpp_all:
        for note in each.rpp:
            midi_obj.instruments[0].notes.append(note)

    # [8]log
    if need_log == True:
        path = outpath if outpath != None else midipath
        rpp_log_path = os.path.join(os.path.dirname(path),'rpp_log.txt')
        note_log_path = os.path.join(os.path.dirname(path),'note_log.txt')
        weight_log_path = os.path.join(os.path.dirname(path),'weight_log.txt')

        # rpp_log
        with open(rpp_log_path,'a') as f:
            f.write(rpp_log)

        # note_log
        for i,ph in enumerate(phrase):
            note_log +=f'\n### Div{i} ###\n'
            for note in ph:
                note_log += f'{note}'
        with open(note_log_path,'a') as f:
            f.write(note_log)

        # weight_log
        for i, ph in enumerate(phrase):
            weight_log+= f'\n### Div{i} ###\n'

            for rpp in rpp4detect:
                if rpp.start>=ph[0].start and rpp.end <=ph[-1].end:
                    weight_log += f'{rpp} Area:{rpp2area[(rpp.start, rpp.end)]:<2} Kind:{str(rpp.shape_and_kind[-1])}\n'
        with open(weight_log_path,'a') as f:
            f.write(weight_log)


    if outpath == None:
        pass

    else:
        midi_obj.dump(outpath)

    return note_all,rpp_all

def midi_graph_evaluation(vertex,edge,midipath):

    # Default Container
    rpp_raw = []
    rpp_need_check = []
    note_all = []
    rpp2area = collections.defaultdict(int)  # {(rpp.start,rpp.end) : area}


    midi_obj = miditoolkit.MidiFile(midipath)
    m = RPP_Detection(midi_path=midipath)
    skeleton_bool, syncopation_bool = m.get_note_typeof_skeleton_syncopation()

    assert len(midi_obj.instruments[0].notes) == len(skeleton_bool) and len(skeleton_bool) == len(syncopation_bool), \
        f'{midipath}音符数量解析不对等'

    for i in range(len(skeleton_bool)):
        true_note = midi_obj.instruments[0].notes[i]
        info = (true_note.velocity, true_note.pitch, true_note.start, true_note.end, skeleton_bool[i], syncopation_bool[i])
        note_all.append(Note(*info))


    note_ryhthm_weight(note_list=note_all)
    note_all.sort(key=lambda x: x.start)


    div_grid = []
    for i,note in enumerate(midi_obj.instruments[0].notes):
        if i == 0:
            div_grid.append(note.start)
        elif i == len(midi_obj.instruments[0].notes) - 1:
            div_grid.append(note.end)
            break
        else:
            if note.start - midi_obj.instruments[0].notes[i-1].end >= 240:
                div_grid.append(note.start)
    div_skeleton = m.get_div_skeleton()


    for note in div_skeleton[:]:
        dur = note.end - note.start
        bar_start = note.start//1920*1920
        bar_end = bar_start+1920
        dur_list = [(x.end-x.start) for x in note_all if (x.start>=bar_start and x.start<bar_end and x.start!=note.start)]

        if len(dur_list) == 0:
            continue
        elif dur <= max(dur_list):
            div_skeleton.remove(note)

    for note in div_skeleton:
        for each in note_all:
            if each.start == note.start and each.end == note.end and each.pitch == note.pitch:
                each.is_break_note = True
                break

    # [1] rpp
    index = 0
    ryhthm_structure2num = {0:2,1:3,2:2,3:3,4:3,5:1,6:1}
    for v in vertex:
        if v[0:3] != 'RPP':
            continue
        rpp_raw.append(v)
        rpp_lenth = ryhthm_structure2num[int(v.split('_')[1])]
        rpp_need_check.append(RPP(note_all[index:index+rpp_lenth]))
        index += rpp_lenth

    # [2] rpp_score
    total_num = len(rpp_raw)
    matching_rpp_num = 0
    print('**',len(rpp_raw),len(rpp_need_check))
    for i in range(len(rpp_raw)):
        
            matching_rpp_num+=1
    rpp_matching_score = round(matching_rpp_num / total_num,2)
    print('rpp_matching_score',rpp_matching_score)

    # [3] edge_score
    print('vertex',vertex)
    print('edge',edge)
    new_seq,new_vertex,new_edge = graph_token_RPP_only(rpp_list=rpp_need_check)
    print('new_vertex',new_vertex)
    print('new_edge',new_edge)

    total_edge_num = 0
    matching_edge_num = 0


    old_rpp_index = [i for i in range(0,len(vertex)) if 'RPP' in vertex[i]]
    new_rpp_index = [i for i in range(0,len(new_vertex)) if 'RPP' in new_vertex[i]]

    for idx1,idx2 in zip(old_rpp_index,new_rpp_index):
        # old
        edge_old = [e for e in edge if e[0]==idx1]
        edge_old_str = [(vertex[e[0]],vertex[e[1]],e[2]) for e in edge_old]
        total_edge_num += len(edge_old)

        # new
        edge_new = [e for e in new_edge if e[0]==idx2]
        edge_new_str = [(new_vertex[e[0]],new_vertex[e[1]],e[2]) for e in edge_new]

        matching_edge = set(edge_old_str).intersection(edge_new_str)
        matching_edge_num+=len(matching_edge)


    edge_score = round(matching_edge_num/ total_edge_num,2)
    print('edge_score',edge_score)


if __name__ == '__main__':

    # # ----------------------- Split Rpp ---------------------------
    # file_list = os.listdir('./ZhPop 2')
    # for f in file_list:
    #     midipath = os.path.join('./ZhPop 2',f)
    #     split_rpp_Zhpop(midipath)

    # # ----------------------- Split RP v1 -----------------------------
    # midi_path = './melody/0.1.mid'
    # split_RP_v1(midipath=midi_path,write_log=True)


    # # ----------------------- Split RP v2 -----------------------------
    # midi_path = './melody/0.mid'
    # split_RP_v2(midipath=midi_path,write_log=True)

    # # ----------------------- Graph Token -----------------------------------
    # # graph
    # data_type = 'WuYun3Data_wikifornia'
    # graph_path = f'{data_type}/graph/graph.txt'
    # graph_check_path = f'{data_type}/check_file/graph_check.txt'
    # file_index_path = f'{data_type}/check_file/file_index.txt'
    # for path in [graph_path,graph_check_path,file_index_path]:
    #     if os.path.exists(path):
    #         os.remove(path)
    #
    # # data
    # dataroot = f'{data_type}/data_processed/data_final'
    # file = [f for f in os.listdir(dataroot) if f[-4:]=='.mid']
    # file = sorted(file,key= lambda x:int(x[:-4]))
    #
    # for i,path in enumerate(file):
    #     # log index
    #     with open(file_index_path,'a') as f:
    #         f.write(f'#{i:<3} {path}\n')
    #
    #     print(f"-----------------------\n#{path} Begin!")
    #     midipath = os.path.join(dataroot,path)
    #     seq,vertex,edge = graph_token_RPP_only(midipath=midipath,rpp_outdir=f'{data_type}/check_file/rpp')
    #     print_graph(vertex,edge,file_index=i,outpath=graph_path)
    #     print_graph_check(seq=seq,vertex=vertex,edge=edge,filepath=graph_check_path,file_index=i,input_file=path)
    #     print(f"#{path} Success!")


    # # ----------------------- rpp_split -----------------------------------

    # input_root = './data/ZhPop/quantify'
    # output_root = './data/ZhPop/rpp_dp'
    #
    # for f in [os.path.join(output_root,x) for x in ['note_log.txt','rpp_log.txt','weight_log.txt']]:
    #     if os.path.exists(f):
    #         os.remove(f)
    #
    # for f in os.listdir(input_root):
    #     if f[-4:] != '.mid':
    #         continue
    #
    #     print('\n-----------------------------------------------\n')
    #     print(f'{f}  Start')
    #
    #     midipath = os.path.join(input_root,f)
    #     outpath = os.path.join(output_root,f)
    #     rpp_divider(midipath,outpath,algorithm='DP')

    # # ----------------------- rpp_overlap_ratio -----------------------------------
    # rawroot = './data/ZhPop/with_rpp_marker'
    # tgtroot = './data/ZhPop/rpp_greedy'
    # rpp_split_accuracy(rawroot=rawroot,tgtroot=tgtroot,logpath=tgtroot+'/accuracy.log')

    # # ----------------------- midi_graph_evaluation -----------------------------------
    midipath = '0.mid'
    seq,vertex,edge = graph_token_RPP_only(midipath=midipath)
    midi_graph_evaluation(vertex=vertex,edge=edge,midipath=midipath)
