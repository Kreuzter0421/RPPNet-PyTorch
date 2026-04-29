import pprint
from typing import List
import miditoolkit
from miditoolkit.midi import parser as mid_parser
from miditoolkit.midi import containers as ct
from itertools import chain
import os
from tqdm import tqdm
from func_timeout import func_set_timeout, FunctionTimedOut

interrupt_interval = 240

default_resolution = 480
beats_per_bar = 4
ticks_per_beat = 480  # default resolution = 480 ticks per quarter note
grid_per_bar = 16
cell = ticks_per_beat * 4 / grid_per_bar
grids_triple = 32
grids_normal = 64
file_name = ''
dst_path = ''

# The intensity corresponding to the grid
grid_intensity_16 = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]
grid_intensity_8 = [5, 1, 3, 1, 4, 1, 3, 1]
grid_intensity_4 = [5, 1, 2, 1]
grid_intensity_2 = [5, 1]

def print_formated(name, value):
    print("====================" + name + "====================")
    print(value)
    print("\n")

def print_note(notes):
    for note in notes:
        print("start: {}, end: {}, pitch: {}, velocity: {}, priority: {}, index: {}".format(note.start, note.end, note.pitch, note.velocity, note.priority, note.index))
    print("note_length: {}".format(len(notes)))

def print_rs(rhythm_seg):
    for rs in rhythm_seg:
        for note in rs:
            if note.priority < 5:
                print("start: {}, end: {}, pitch: {}, velocity: {}, priority: {}, index: {}".format(note.start, note.end, note.pitch, note.velocity, note.priority, note.index))
        if note.priority == 5:
            print("------------------------Phrase------------------------")
        if rs[0].priority != 5:
            print("------------------------RS----------------------------")

def print_cell(rhythm_cells):
    for rs in rhythm_cells:
        for cell in rs:
            if len(cell) == 1:
                continue
                print("------------------------Phrase------------------------")
            else:
                print(cell_intensity_tags(cell))
                for note in cell:
                    print("start: {}, end: {}, pitch: {}, velocity: {}, priority: {}, index: {}".format(note.start, note.end, note.pitch, note.velocity, note.priority, note.index))
                print("------------------------Cell--------------------------")
        if rs[0][0].priority != 5:
            print("------------------------RS----------------------------")

def get_rpp_rhythm_type(rhythm_cells):
    rpp_rhythm_type = []
    for rs in rhythm_cells:
        for cell in rs:
            if len(cell) >=1 :
                rpp_rhythm_type.append(cell_intensity_tags(cell)[0])

    return rpp_rhythm_type

def cell_intensity_tags(cell):
    tags = []

    for note in cell:
        if note.priority == 2:
            tags.append(6)
            continue
        
        grid_ticks = beats_per_bar * ticks_per_beat / grid_per_bar      # grid_ticks = 120
        note_duration = note.end - note.start
        if note_duration > grid_ticks * 8:
            tags.append(5)
        elif note_duration > grid_ticks * 4:
            grid_intensity_idx = int(note.start % (beats_per_bar * ticks_per_beat) // (grid_ticks * 8))
            tags.append(grid_intensity_2[grid_intensity_idx])
        elif note_duration > grid_ticks * 2:
            grid_intensity_idx = int(note.start % (beats_per_bar * ticks_per_beat) // (grid_ticks * 4))
            tags.append(grid_intensity_4[grid_intensity_idx])
        elif note_duration > grid_ticks:
            grid_intensity_idx = int(note.start % (beats_per_bar * ticks_per_beat) // (grid_ticks * 2))
            tags.append(grid_intensity_8[grid_intensity_idx])
        else:
            grid_intensity_idx = int(note.start % (beats_per_bar * ticks_per_beat) // (grid_ticks))
            tags.append(grid_intensity_16[grid_intensity_idx])
    
    simplified_tags = []

    for t in tags:
        if t == max(tags):
            simplified_tags.append(1)
        else:
            simplified_tags.append(0)
            
    return simplified_tags, tags

class Note:
    def __init__(self, start, end, pitch, velocity, index, priority = 4):
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.index = index
        self.priority = priority



class RPP_Detection:

    def __init__(self, midi_path, resolution=480, grids=16):
        self.midi_path = midi_path
        self.file_name = os.path.basename(midi_path)
        self.resolution = resolution  
        self.grids = grids  
        self.step = resolution * 4 / grids  
        self.bar_ticks = resolution * 4
        self.subsections = self._divide_subsections()

    def _divide_subsections(self):
        midi = miditoolkit.MidiFile(self.midi_path)
        notes = midi.instruments[0].notes
        res_dict = dict()
        for note in notes:
            start = note.start
            end = note.end
            duration = end - start
            if duration >= self.step:
                key = int(start // self.bar_ticks) 
                if key not in res_dict:
                    res_dict[key] = []
                res_dict[key].append(note)
        return res_dict

    # ---------------------
    # Segmented Speech Recognition
    # ---------------------
    def _get_split(self):
        split_dict = dict()  
        split_dict_4 = dict()  
        split_dict_8 = dict()
        split_dict_16 = dict()

        step16 = self.step
        for bar_id, bar_notes in self.subsections.items():
            if bar_id not in split_dict:
                split_dict[bar_id] = []
                split_dict_4[f'{bar_id}'] = []
                split_dict_8[f'{bar_id}'] = []
                split_dict_16[f'{bar_id}'] = []

            start = self.bar_ticks * bar_id
            note_start_4 = [4 * step16 + start, 12 * step16 + start]
            note_start_8 = [i * step16 + start for i in range(2, 16, 4)]
            note_start_16 = [i * step16 + start for i in range(1, 16, 2)]

            for note in bar_notes:
                note_duration = note.end - note.start
                if note_duration >= step16:

                    if (note.start == note_start_4[0]) and (note.end > (8 * step16 + start)):
                        split_dict[bar_id].append(note)
                        split_dict_4[f'{bar_id}'].append(note)
                    elif (note.start == note_start_4[1]) and (note.end > (16 * step16 + start)):
                        split_dict[bar_id].append(note)
                        split_dict_4[f'{bar_id}'].append(note)

                    elif (note.start == note_start_8[0]) and (note.end > 4 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_8[f'{bar_id}'].append(note)
                    elif (note.start == note_start_8[1]) and (note.end > 8 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_8[f'{bar_id}'].append(note)
                    elif (note.start == note_start_8[2]) and (note.end > 12 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_8[f'{bar_id}'].append(note)
                    elif (note.start == note_start_8[3]) and (note.end > 16 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_8[f'{bar_id}'].append(note)

                    elif (note.start == note_start_16[0]) and (note.end > 2 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[1]) and (note.end > 4 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[2]) and (note.end > 6 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[3]) and (note.end > 8 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[4]) and (note.end > 10 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[5]) and (note.end > 12 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[6]) and (note.end > 14 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                    elif (note.start == note_start_16[7]) and (note.end > 16 * step16 + start):
                        split_dict[bar_id].append(note)
                        split_dict_16[f'{bar_id}'].append(note)
                else:
                    continue
        return split_dict, split_dict_4, split_dict_8, split_dict_16

    # --------------------------------------------------------------------------------------------------------------
    # Beat Accent Recognition
    # --------------------------------------------------------------------------------------------------------------
    def _get_stress(self):
        heavy_dict = dict()
        heavy_dict_clean = dict()
        split_dict, _, _, _ = self._get_split()


        for bar_id, bar_notes in self.subsections.items(): 
            start = self.bar_ticks * (bar_id)  
            first_beat_position = start
            third_beat_postion = start + 8 * self.step
            if bar_id not in heavy_dict:
                heavy_dict[bar_id] = []
            for note in bar_notes:
                if (note.start == first_beat_position) or (note.start == third_beat_postion):
                    heavy_dict[bar_id].append(note)


        for heavy_bar_id, heavy_bar_notes in heavy_dict.items():
            if heavy_bar_id not in heavy_dict_clean:
                heavy_dict_clean[heavy_bar_id] = []

            for heavy_note in heavy_bar_notes:
                heavy_note_flag = True
                heavy_note_length = heavy_note.end - heavy_note.start

                for split_bar_id, split_bar_notes in split_dict.items():
                    for split_note in split_bar_notes:
                        split_note_length = split_note.end - split_note.start
                        start_delta = split_note.start - heavy_note.start
                        if 0 <= start_delta < 3 * self.step:
                            heavy_note_flag = False
                            break

                    if heavy_note_flag == False:
                        break

                if heavy_note_flag:
                    heavy_dict_clean[heavy_bar_id].append(heavy_note)
        return heavy_dict_clean

    # ------------------------------------------------------------
    # Long note Recognition
    # ------------------------------------------------------------
    def _get_long(self):
        long_dict = dict()
        for bar_id, bar_notes in self.subsections.items():

            if bar_id not in long_dict:
                long_dict[bar_id] = []

          
            duration_list = [x.end - x.start for x in bar_notes]
            max_duration = max(duration_list)
            tup = [(i, duration_list[i]) for i in range(len(duration_list))]
            idx_list = [i for i, n in tup if n == max_duration]  

            for idx in idx_list:
                long_dict[bar_id].append(bar_notes[idx])
        return long_dict

    # -------------------------------
    # Fundamental Tone Extraction
    # -----------

    def prepare_dict(self):
        split_dict, _, _, _ = self._get_split()  
        heavy_dict = self._get_stress()  
        long_dict = self._get_long()  

        heavy_list = list(chain(*heavy_dict.values()))
        long_list = list(chain(*long_dict.values()))
        split_list = list(chain(*split_dict.values()))

        return split_list, heavy_list, long_list

    def extract_skeleton(self, heavy_list, long_list, split_list):
        skeleton_dict = dict()
        skeleton_note_list = []
        prolongation_note_list = []

        note_index = 0

        for k, v in self.subsections.items():  
            if k not in skeleton_dict:
                skeleton_dict[k] = []
            for note in v:
                
                note_object = Note(start=note.start, end=note.end, pitch=note.pitch, velocity=note.velocity,
                                   index=note_index)
                note_index += 1
                
                if ((note in heavy_list) and (note not in long_list) and (note not in split_list)):
                    skeleton_dict[k].append(note)
                    note_object.priority = 3
                    skeleton_note_list.append(note_object)
                
                elif ((note in heavy_list) and (note in long_list) and (note not in split_list)):
                    skeleton_dict[k].append(note)
                    note_object.priority = 1
                    skeleton_note_list.append(note_object)
                
                elif ((note not in heavy_list) and (note in long_list) and (note in split_list)):
                    skeleton_dict[k].append(note)
                    note_object.priority = 2
                    skeleton_note_list.append(note_object)
                else:
                    note_object.priority = 4
                    prolongation_note_list.append(note_object)
        
        return skeleton_note_list, prolongation_note_list
    
    def filter_continuous_skeleton(self, need_filter, skeleton_note_list, prolongation_note_list):
        
        continuous_note_list = []
        last_note_index = 0
        for idx, note in enumerate(skeleton_note_list):
            if idx == 0:
                continuous_note_list.append([note])
                last_note_index = note.index
            else:
                if note.index == last_note_index + 1:
                    continuous_note_list[-1].append(note)
                    last_note_index = note.index
                else:
                    continuous_note_list.append([note])
                    last_note_index = note.index
        
        if not need_filter:
            return skeleton_note_list, prolongation_note_list

        final_skeleton_note_list = []
        for group_idx, note_group in enumerate(continuous_note_list):
            
            if len(note_group) <= 2:
                final_skeleton_note_list.append(note_group)
            
            else:
                priority_list = []

                for note in note_group:
                    priority_list.append(note.priority)
                priority_set = set(priority_list)
                priority_set_length = len(priority_set)
                max_priority = min(priority_set)  

                if priority_set_length == 1:
                    if max_priority == 1:
                        temp_group = []
                        for note in note_group:
                            if note.start % 1920 == 0:  
                                temp_group.append(note)
                        final_skeleton_note_list.append(temp_group)
                    elif max_priority == 2:
                        temp_group = []
                        bar_group = dict()
                        for note in note_group:
                            note_bar = int(note.start / 1920)
                            if note_bar not in bar_group:
                                bar_group[note_bar] = []
                            bar_group[note_bar].append(note)
                        for k, v in bar_group.items():
                            if len(v) == 1: 
                                temp_group.append(v[0])
                            else:           
                                notes_length = [note.end - note.start for note in v]
                                max_length_note_index = notes_length.index(max(notes_length))
                                temp_group.append(v[max_length_note_index])
                        final_skeleton_note_list.append(temp_group)
                    elif max_priority == 3:
                        temp_group = []
                        for note in note_group:
                            if note.start % 1920 == 0:  
                                temp_group.append(note)
                        final_skeleton_note_list.append(temp_group)

                elif priority_set_length == 2 or priority_set_length == 3:
                    
                    if 1 in priority_set:
                        
                        tempo_note_group_1 = []
                        for note in note_group:
                            if note.priority == 1:
                                tempo_note_group_1.append(note)
                        
                        temp_group = []
                        if len(tempo_note_group_1) == 1:
                            temp_group.append(tempo_note_group_1[0])
                            final_skeleton_note_list.append(temp_group)
                        else:
                            for idx, note in enumerate(tempo_note_group_1):
                                if idx == len(tempo_note_group_1) - 1:
                                    if note.index - 1 == tempo_note_group_1[idx - 1].index:  
                                        if note.start % 1920 == 0:
                                            temp_group.append(note)
                                    else:
                                        temp_group.append(note)
                                elif idx == 0:
                                    if note.index + 1 == tempo_note_group_1[idx + 1].index: 
                                        if note.start % 1920 == 0:
                                            temp_group.append(note)
                                    else:
                                        temp_group.append(note)
                                else:
                                    
                                    if note.index + 1 != tempo_note_group_1[idx + 1].index and note.index - 1 != \
                                            tempo_note_group_1[idx - 1].index:
                                        temp_group.append(note)
                                    
                                    else:
                                        if note.start % 1920 == 0:
                                            temp_group.append(note)
                            final_skeleton_note_list.append(temp_group)
                   
                    else:
                        
                        tempo_note_group_2 = []
                        for note in note_group:
                            if note.priority == 2:
                                tempo_note_group_2.append(note)

                        temp_group = []
                        if len(tempo_note_group_2) == 1:
                            temp_group.append(tempo_note_group_2[0])
                            final_skeleton_note_list.append(temp_group)
                        else:
                            tempo_split_note_dict = dict()
                            for idx, note in enumerate(tempo_note_group_2):
                                note_bar = int(note.start / 1920)
                                if note_bar not in tempo_split_note_dict:
                                    tempo_split_note_dict[note_bar] = []
                                tempo_split_note_dict[note_bar].append(note)
                            for k, v in tempo_split_note_dict.items():
                                if len(v) == 1:
                                    temp_group.append(v[0])
                                else:
                                    notes_length = [note.end - note.start for note in v]  # the common spit
                                    max_length_note_index = notes_length.index(max(notes_length))
                                    temp_group.append(v[max_length_note_index])
                            final_skeleton_note_list.append(temp_group)

        unfold_skeleton_note_list = []
        for notes in final_skeleton_note_list:
            for note in notes:
                unfold_skeleton_note_list.append(note)

        for note in skeleton_note_list:
            if note not in unfold_skeleton_note_list:
                n_note = Note(start=note.start, end=note.end, pitch=note.pitch, velocity=note.velocity, index=note.index, priority=4)
                prolongation_note_list.append(n_note)

        return unfold_skeleton_note_list, prolongation_note_list

    def generate_subsection_notes_list(self, all_notes_list):
        subsection_notes_list = []
        for key in self.subsections.keys():
            curr_section = self.subsections[key]
            notes_section = []
            for old_note in curr_section:
                for note in all_notes_list:
                    if note.start == old_note.start and note.end == old_note.end and note.pitch == old_note.pitch and note.velocity == old_note.velocity:
                        notes_section.append(note)
            notes_section.sort(key=lambda x: x.index, reverse=False)
            subsection_notes_list.append(notes_section)
        
        return subsection_notes_list

    def add_interrupt_notes(self, notes_list):
        new_notes_list = []
        previous_note = None
        
        for note in notes_list:
            if previous_note == None:
                previous_note = note
                new_notes_list.append(note)
                continue
            
            else:
                duration = note.start - previous_note.end
                if duration >= interrupt_interval:
                    new_notes_list.append(Note(start=previous_note.end, end=note.start, pitch=0, velocity=0, index=0, priority=5))
                    new_notes_list.append(note)
                else:
                    new_notes_list.append(note)
                previous_note = note
        
        
        note_index = 0
        for note in new_notes_list:
            note.index = note_index
            note_index += 1
        
        return new_notes_list

    def rhythm_segmentation(self, notes_list):

        def has_skeleton_note(rhythem_seg):
            for note in rhythem_seg:
                if note.priority < 4:
                    return True
            return False

        rhythm_seg_notes_list = []
        single_rhythm_seg = []
        notes_list_idx = 0
        for notes_list_idx in range(len(notes_list)):
            note = notes_list[notes_list_idx]
            
            if note.priority < 4:
                if len(single_rhythm_seg) == 0:
                    single_rhythm_seg.append(note)
                else:
                    
                    if single_rhythm_seg[0].priority == 4:
                        single_rhythm_seg.append(note)
                        rhythm_seg_notes_list.append(single_rhythm_seg)
                        single_rhythm_seg = []
                    elif single_rhythm_seg[0].priority < 4:
                        if len(single_rhythm_seg) == 1:
                            note.priority = 4
                            single_rhythm_seg.append(note)
                           
                        else:
                            rhythm_seg_notes_list.append(single_rhythm_seg)
                            single_rhythm_seg = []
                            single_rhythm_seg.append(note)
          
            elif note.priority == 4:
                single_rhythm_seg.append(note)
            
            elif note.priority == 5:
                
                if len(single_rhythm_seg) == 1:
                    
                    if len(rhythm_seg_notes_list) != 0 and abs(single_rhythm_seg[0].start - rhythm_seg_notes_list[-1][-1].end) < interrupt_interval:
                        rhythm_seg_notes_list[-1].append(single_rhythm_seg[0])
                        single_rhythm_seg = []
                        continue
                    
                    if len(rhythm_seg_notes_list) != 0 and notes_list_idx == len(notes_list) - 1:
                        rhythm_seg_notes_list[-1].append(single_rhythm_seg[0])
                        single_rhythm_seg = []
                        continue
                    if len(rhythm_seg_notes_list) != 0:
                        if notes_list_idx != len(notes_list) - 1:
                            previous_note_interval = abs(single_rhythm_seg[0].start - rhythm_seg_notes_list[-1][-1].end)
                            next_note_interval = abs(single_rhythm_seg[0].end - notes_list[notes_list_idx + 1].start)
                            if previous_note_interval < next_note_interval:
                                rhythm_seg_notes_list[-1].append(single_rhythm_seg[0])
                                single_rhythm_seg = []
                                continue
                            else:
                                continue
                        else:
                            rhythm_seg_notes_list[-1].append(single_rhythm_seg[0])
                            single_rhythm_seg = []
                            continue
                    else:
                        continue
                else:
                    if len(single_rhythm_seg) != 0:
                        if has_skeleton_note(single_rhythm_seg):
                            rhythm_seg_notes_list.append(single_rhythm_seg)
                        else:
                            if len(rhythm_seg_notes_list) != 0:
                                if abs(single_rhythm_seg[0].start - rhythm_seg_notes_list[-1][-1].end) < interrupt_interval:
                                    for n in single_rhythm_seg:
                                        rhythm_seg_notes_list[-1].append(n)
                                else:
                                    rhythm_seg_notes_list.append(single_rhythm_seg)
                            else:
                                rhythm_seg_notes_list.append(single_rhythm_seg)
                single_rhythm_seg = []
        
        if single_rhythm_seg != []:
            if len(single_rhythm_seg) > 1:
                rhythm_seg_notes_list.append(single_rhythm_seg)
            else:
                rhythm_seg_notes_list[-1].append(single_rhythm_seg[0])

        return rhythm_seg_notes_list

    def rhythm_cell_segmentation(self, rhythm_seg_notes_list):
        
        def cell_normalization(cell):
            normalized_cell = []
            start_coefficient = cell[0].start
            pitch_coefficient = cell[0].pitch
            for note in cell:
                normalized_note = Note(start=note.start - start_coefficient, end=note.end - start_coefficient, pitch=note.pitch - pitch_coefficient, velocity=note.velocity, index=note.index, priority=note.priority)
                normalized_cell.append(normalized_note)
            return normalized_cell
        
        def normalized_cell_compare(l: List[Note], r: List[Note]):
            if len(l) != len(r):
                return False
            
            for idx in range(len(l)):
                l_note = l[idx]
                r_note = r[idx]
                if l_note.start == r_note.start and l_note.end == r_note.end:
                    continue
                else: 
                    return False
            
            return True

        def cell_intensity_tags(cell):
            tags = []

            for note in cell:
                if note.priority == 2:
                    tags.append(6)
                    continue
                
                grid_ticks = beats_per_bar * ticks_per_beat / grid_per_bar      # grid_ticks = 120
                note_duration = note.end - note.start
                if note_duration > grid_ticks * 8:
                    # print('note length > 960 ticks')
                    tags.append(5)
                elif note_duration > grid_ticks * 4:
                    grid_intensity_idx = int(note.start % (beats_per_bar * ticks_per_beat) // (grid_ticks * 8))
                    tags.append(grid_intensity_2[grid_intensity_idx])
                elif note_duration > grid_ticks * 2:
                    grid_intensity_idx = int(note.start % (beats_per_bar * ticks_per_beat) // (grid_ticks * 4))
                    tags.append(grid_intensity_4[grid_intensity_idx])
                elif note_duration > grid_ticks:
                    grid_intensity_idx = int(note.start % (beats_per_bar * ticks_per_beat) // (grid_ticks * 2))
                    tags.append(grid_intensity_8[grid_intensity_idx])
                else:
                    grid_intensity_idx = int(note.start % (beats_per_bar * ticks_per_beat) // (grid_ticks))
                    tags.append(grid_intensity_16[grid_intensity_idx])
            
            simplified_tags = []

            for t in tags:
                if t == max(tags):
                    simplified_tags.append(1)
                else:
                    simplified_tags.append(0)
                    
            return simplified_tags, tags    
            
        def cal_repetition(cell_group, rhythm_seg_notes_list):
            key_cell = cell_group[0]
            normalized_key_cell = cell_normalization(key_cell)

            
            all_notes_divisions = []
            normalized_all_notes_divisions = []
            for rs in rhythm_seg_notes_list:
                rs_notes_divisions = []
                normalized_rs_notes_divisions = []
                for idx in range(len(rs) - len(key_cell) + 1):
                    curr_division = []
                    for i in range(len(key_cell)):
                        curr_division.append(rs[idx + i])
                    normalized_curr_division = cell_normalization(curr_division)
                    
                    rs_notes_divisions.append(curr_division)
                    normalized_rs_notes_divisions.append(normalized_curr_division)

                all_notes_divisions.append(rs_notes_divisions)
                normalized_all_notes_divisions.append(normalized_rs_notes_divisions)
            
            res = 0
            for rs_idx in range(len(normalized_all_notes_divisions)):
                p_rs = normalized_all_notes_divisions[rs_idx]
                o_rs = all_notes_divisions[rs_idx]

                for division_idx in range(len(p_rs)):
                    p_division = p_rs[division_idx]
                    o_division = o_rs[division_idx]
                    
                    if normalized_cell_compare(normalized_key_cell, p_division):
                        o_simplified_tags, o_tags = cell_intensity_tags(o_division)
                        key_simplified_tags, key_tags = cell_intensity_tags(key_cell)
                        
                        if o_simplified_tags == key_simplified_tags:
                            res += 1
            return res
            
        def single_rhythm_seg_cells(rhythm_seg, rhythm_seg_notes_list):
            def enforce_max_cell_size(cells, max_size=3):
                enforced_cells = []

                for cell in cells:
                    if len(cell) <= max_size:
                        enforced_cells.append(cell)
                        continue

                    start_idx = 0
                    while start_idx < len(cell):
                        end_idx = min(start_idx + max_size, len(cell))
                        enforced_cells.append(cell[start_idx:end_idx])
                        start_idx = end_idx

                return enforced_cells

            curr_rhythm_cell_seg = []
            if len(rhythm_seg) == 1:
                curr_rhythm_cell_seg.append(rhythm_seg)
            elif len(rhythm_seg) == 2:
                curr_rhythm_cell_seg.append(rhythm_seg)
            elif len(rhythm_seg) == 3:
                curr_rhythm_cell_seg.append(rhythm_seg)
            elif len(rhythm_seg) == 4:
                curr_rhythm_cell_seg = [rhythm_seg[0:2], rhythm_seg[2:4]]
            elif len(rhythm_seg) == 5:
                rhythm_cell_choices = [
                    [rhythm_seg[0:3], rhythm_seg[3:5]],
                    [rhythm_seg[0:2], rhythm_seg[2:5]],
                ]
                highest_res = -1
                best_choice = []
                for choice in rhythm_cell_choices:
                    res = cal_repetition(choice, rhythm_seg_notes_list)
                    if res > highest_res:
                        highest_res = res
                        best_choice = choice
                curr_rhythm_cell_seg = best_choice
            elif len(rhythm_seg) == 6:
                rhythm_cell_choices = [
                    [rhythm_seg[0:3], rhythm_seg[3:6]],
                    [rhythm_seg[0:2]] + single_rhythm_seg_cells(rhythm_seg[2:len(rhythm_seg)], rhythm_seg_notes_list),
                ]
                highest_res = -1
                best_choice = []
                for choice in rhythm_cell_choices:
                    res = cal_repetition(choice, rhythm_seg_notes_list)
                    if res > highest_res:
                        highest_res = res
                        best_choice = choice
                curr_rhythm_cell_seg = best_choice
            elif len(rhythm_seg) >= 7:
                rhythm_cell_choices = [
                    [rhythm_seg[0:3]] + single_rhythm_seg_cells(rhythm_seg[3:len(rhythm_seg)], rhythm_seg_notes_list),
                    [rhythm_seg[0:2]] + single_rhythm_seg_cells(rhythm_seg[2:len(rhythm_seg)], rhythm_seg_notes_list),
                ]
                highest_res = -1
                best_choice = []
                for choice in rhythm_cell_choices:
                    res = cal_repetition(choice, rhythm_seg_notes_list)
                    if res > highest_res:
                        highest_res = res
                        best_choice = choice
                curr_rhythm_cell_seg = best_choice
            
            return enforce_max_cell_size(curr_rhythm_cell_seg)

        rhythm_cell_seg_notes_list = []
        for rhythm_seg in rhythm_seg_notes_list:
            rhythm_cell_seg_notes_list.append(single_rhythm_seg_cells(rhythm_seg, rhythm_seg_notes_list))
        
        return rhythm_cell_seg_notes_list

    def formatted_rhythm_cell_output(self, rhythm_cell_seg_notes_list, output_file_dir):
        def cell_normalization(cell):
            normalized_cell = []
            start_coefficient = cell[0].start
            pitch_coefficient = cell[0].pitch
            for note in cell:
                normalized_note = Note(start=note.start - start_coefficient, end=note.end - start_coefficient, pitch=note.pitch - pitch_coefficient, velocity=note.velocity, index=note.index, priority=note.priority)
                normalized_cell.append(normalized_note)
            return normalized_cell
        
        def normalized_cell_compare(l: List[Note], r: List[Note]):
            if len(l) != len(r):
                return False
            
            for idx in range(len(l)):
                l_note = l[idx]
                r_note = r[idx]
                if l_note.start == r_note.start and l_note.end == r_note.end:
                    continue
                else: 
                    return False
            
            return True
        
        with open(output_file_dir, "w") as f:
            f.write(file_name + "\n")

            rpp_dict = {}
            rpp_idx = 1
            for rs_idx in range(len(rhythm_cell_seg_notes_list)):
                rs = rhythm_cell_seg_notes_list[rs_idx]
                first_cell = rs[0]
                first_note = first_cell[0]
                rs_bar = first_note.start // (beats_per_bar * ticks_per_beat)
                f.write("RS{}_bar{}".format(rs_idx + 1, rs_bar + 1) + "\n")
                
                for cell in rs:
                    normalized_cell = cell_normalization(cell)
                    in_rpp_dict = False
                    for rpp_name in rpp_dict.keys():
                        if normalized_cell_compare(normalized_cell, rpp_dict[rpp_name]):
                            curr_line = "{}: {}[".format(rpp_name, len(cell))
                            for idx in range(len(cell)):
                                note = cell[idx]
                                if idx != 0:
                                    curr_line += ", "
                                curr_line += "note{}(start = {}, end = {})".format(idx + 1, note.start, note.end)
                            curr_line += "]"
                            in_rpp_dict = True
                            break
                    if not in_rpp_dict:
                        rpp_dict["RPP{}".format(rpp_idx)] = normalized_cell
                        rpp_name = "RPP{}".format(rpp_idx)
                        rpp_idx += 1
                        curr_line = "{}: {}[".format(rpp_name, len(cell))
                        for idx in range(len(cell)):
                            note = cell[idx]
                            if idx != 0:
                                curr_line += ", "
                            curr_line += "note{}(start = {}, end = {})".format(idx + 1, note.start, note.end)
                        curr_line += "]"
                    
                    f.write("\t" + curr_line + "\n")
            f.close()

    def export_midi_file(self, rhythm_cell_seg_notes_list):
        mido_obj = mid_parser.MidiFile()
        beat_resol = mido_obj.ticks_per_beat
        track = ct.Instrument(program=0, is_drum=False, name='track1')
        mido_obj.instruments = [track]
        
        color_idx = 0

        for rs in rhythm_cell_seg_notes_list:
            for cell in rs:
                for n in cell:
                    if color_idx % 2 == 0:
                        note = ct.Note(start=n.start, end=n.end, pitch=n.pitch, velocity=60)
                    else:
                        note = ct.Note(start=n.start, end=n.end, pitch=n.pitch, velocity=127)
                    mido_obj.instruments[0].notes.append(note)
                color_idx += 1
        
        mido_obj.dump('result.mid')

    def get_RPP_List(self, rhythm_cell_seg_notes_list):
        RPP_list = []
        for bar in rhythm_cell_seg_notes_list:
            for RP in bar:
                RPP_group = []
                for item in RP:
                    RPP_group.append(miditoolkit.Note(start=item.start, end=item.end, pitch=item.pitch, velocity=item.velocity))
                RPP_list.append(RPP_group)
        return RPP_list
    
    def get_skeleton_list(self, skeleton_note_list):
        notes = []
        for note in skeleton_note_list:
            if note.priority == 1 or note.priority == 2:
                notes.append(miditoolkit.Note(start=note.start, end=note.end, pitch=note.pitch, velocity=note.velocity))

        return notes

    def skeleton_notes_boolean_list(self, notes_list):
        boolean_list = []
        for note in notes_list:
            
            if note.priority < 4:
                boolean_list.append(True)
            else:
                boolean_list.append(False)

        return boolean_list

    def split_notes_boolean_list(self, notes_list, split_list):
        boolean_list = []

        def equal(lhs, rhs):
            return (
                        lhs.start == rhs.start and lhs.end == rhs.end and lhs.pitch == rhs.pitch and lhs.velocity == rhs.velocity)

        def is_split(note, split_list):
            for n in split_list:
                if equal(n, note):
                    return True
            return False

        for note in notes_list:
            if is_split(note, split_list):
                boolean_list.append(True)
            else:
                boolean_list.append(False)

        return boolean_list

    def get_div_skeleton(self):
        split_list, heavy_list, long_list = self.prepare_dict()

        skeleton_note_list, prolongation_note_list = self.extract_skeleton(heavy_list, long_list, split_list)

        filtered_skeleton_note_list, prolongation_note_list = self.filter_continuous_skeleton(need_filter=True, skeleton_note_list=skeleton_note_list, prolongation_note_list=prolongation_note_list)

        

        miditoolkit_skeleton_list = self.get_skeleton_list(filtered_skeleton_note_list)
        
        return miditoolkit_skeleton_list

    def get_note_typeof_skeleton_syncopation(self):
        split_list, heavy_list, long_list = self.prepare_dict()

        skeleton_note_list, prolongation_note_list = self.extract_skeleton(heavy_list, long_list, split_list)

        filtered_skeleton_note_list, prolongation_note_list = self.filter_continuous_skeleton(need_filter=True,
                                                                                              skeleton_note_list=skeleton_note_list,
                                                                                              prolongation_note_list=prolongation_note_list)

        all_notes_list = filtered_skeleton_note_list + prolongation_note_list
        all_notes_list.sort(key=lambda x: x.index, reverse=False)

       
        skeleton_notes_boolean_list = self.skeleton_notes_boolean_list(all_notes_list)
       
        split_note_boolean_list = self.split_notes_boolean_list(all_notes_list, split_list)

        return skeleton_notes_boolean_list,split_note_boolean_list

    @func_set_timeout(40)
    def all_steps(self):
        
        split_list, heavy_list, long_list = self.prepare_dict()


        skeleton_note_list, prolongation_note_list = self.extract_skeleton(heavy_list, long_list, split_list)

        
        filtered_skeleton_note_list, prolongation_note_list = self.filter_continuous_skeleton(need_filter=True, skeleton_note_list=skeleton_note_list, prolongation_note_list=prolongation_note_list)


       
        all_notes_list = filtered_skeleton_note_list + prolongation_note_list
        all_notes_list.sort(key=lambda x: x.index, reverse=False)
        
        
        refined_notes_list = self.add_interrupt_notes(all_notes_list)

        
        rhythm_seg_notes_list = self.rhythm_segmentation(refined_notes_list)
       
        rhythm_cell_seg_notes_list = self.rhythm_cell_segmentation(rhythm_seg_notes_list)

        rpp_list = self.get_RPP_List(rhythm_cell_seg_notes_list=rhythm_cell_seg_notes_list)
        rpp_rhythm_type = get_rpp_rhythm_type(rhythm_cell_seg_notes_list)

        

        return rpp_list,rpp_rhythm_type

if __name__ == '__main__':
    rootdir = './melody/0.mid'
    
    midi_path = rootdir
    file_name = os.path.basename(midi_path)
    dst_path = '.'
    
    m = RPP_Detection(midi_path)
    rpp_list , rpp_ryh = m.all_steps()

    print(len(rpp_list),len(rpp_ryh))
    print(rpp_ryh)


    