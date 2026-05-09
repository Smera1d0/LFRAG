import json
import os
import cv2
import numpy as np
from copy import deepcopy
from tqdm import tqdm 

# ================= Configuration =================

# 1. Path Configuration
# Input jsonl file path
INPUT_FILE = ''
# Output jsonl file path
OUTPUT_FILE = ''
# Visualization save directory (first 10 samples)
VIS_DIR = ''

# 2. Image root directory
IMG_ROOT_DIR = '' 

# 3. Merge parameters (thresholds)
Y_THRESH = 40        # Vertical merge threshold (pixels)
X_ALIGN_THRESH = 0.7 # Horizontal alignment threshold (0-1)
OVERLAP_THRESH = 0.9 # Overlap merge threshold (0-1)

# 4. Semantic merge groups
MERGE_GROUPS = {
    'figure_group': {'figure', 'figure_caption'},
    'table_group':  {'table', 'table_caption', 'table_footnote'},
    'text_group':   {'plain text', 'isolate_formula', 'formula_caption'},
    'title_group':  {'title'},
    'abandon_group': {'abandon'}
}

# 5. Group priority for class name assignment after merging (higher in the list means higher priority)
GROUP_PRIORITY = {
    'figure_group': 'figure',       
    'table_group': 'table',         
    'text_group': 'plain text',     
    'title_group': 'title',
    'abandon_group': 'abandon'
}

# 6. Visualization colors (B, G, R)
VIS_COLORS = {
    'plain text': (200, 200, 200), # grey
    'title': (0, 0, 255),          # red
    'figure': (0, 255, 0),         # green
    'table': (255, 0, 0),          # blue
    'isolate_formula': (0, 255, 255), # yellow
    'abandon': (160, 160, 160),
    'default': (100, 100, 100)
}

# ================= Core Algorithm Functions =================

def get_group_name(class_name):
    """Get group name by class name"""
    for group_name, members in MERGE_GROUPS.items():
        if class_name in members:
            return group_name
    return None

def calculate_iou_x(box1, box2):
    """Calculate the overlap rate on the X-axis projection"""
    x1 = max(box1['x1'], box2['x1'])
    x2 = min(box1['x2'], box2['x2'])
    overlap = max(0, x2 - x1)
    
    width1 = box1['x2'] - box1['x1']
    width2 = box2['x2'] - box2['x1']
    min_width = min(width1, width2)
    
    if min_width <= 0: return 0
    return overlap / min_width

def is_overlapping(box1, box2, threshold=0.9):
    """Determine if two boxes are overlapping based on the intersection over minimum area ratio"""
    x_overlap = max(0, min(box1['x2'], box2['x2']) - max(box1['x1'], box2['x1']))
    y_overlap = max(0, min(box1['y2'], box2['y2']) - max(box1['y1'], box2['y1']))
    intersection = x_overlap * y_overlap
    
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    min_area = min(area1, area2)
    
    if min_area <= 0: return False
    return (intersection / min_area) > threshold

def merge_two_boxes(base_box, merge_box, target_group_name):
    """Merge two boxes"""
    b1 = base_box['box']
    b2 = merge_box['box']
    
    # 1. Expand coordinates
    base_box['box']['x1'] = min(b1['x1'], b2['x1'])
    base_box['box']['y1'] = min(b1['y1'], b2['y1'])
    base_box['box']['x2'] = max(b1['x2'], b2['x2'])
    base_box['box']['y2'] = max(b1['y2'], b2['y2'])
    
    # 2. Update confidence (max)
    base_box['confidence'] = max(base_box['confidence'], merge_box['confidence'])
    
    # 3. Update class name
    if target_group_name and target_group_name in GROUP_PRIORITY:
        base_box['class_name'] = GROUP_PRIORITY[target_group_name]
    
    # 4. Merge ID tracking list
    if 'original_ids' in merge_box:
        base_box['original_ids'].extend(merge_box['original_ids'])
    else:
        base_box['original_ids'].append(merge_box['bbox_id'])
        
    return base_box

def process_single_page(bboxes):
    """Process the bbox list for a single page"""
    # Preprocessing: Initialize tracking IDs
    valid_bboxes = []
    for b in bboxes:
        b_copy = deepcopy(b)
        if 'original_ids' not in b_copy:
            b_copy['original_ids'] = [b['bbox_id']]
        valid_bboxes.append(b_copy)
            
    valid_bboxes.sort(key=lambda x: (x['box']['y1'], x['box']['x1']))
    
    if not valid_bboxes:
        return []

    merged_indices = set()
    final_results = []

    for i in range(len(valid_bboxes)):
        if i in merged_indices:
            continue
            
        current_box = deepcopy(valid_bboxes[i])
        current_group = get_group_name(current_box['class_name'])
        
        while True:
            merged_in_this_round = False
            
            for j in range(i + 1, len(valid_bboxes)):
                if j in merged_indices:
                    continue
                
                next_box = valid_bboxes[j]
                next_group = get_group_name(next_box['class_name'])
                
                b_curr = current_box['box']
                b_next = next_box['box']
                
                should_merge = False
                merge_target_group = current_group 

                # Merge logic based on group and spatial relationships
                is_same_group = (current_group is not None and current_group == next_group)
                is_title_plus_text = (current_group == 'title_group' and next_group == 'text_group')
                
                if is_same_group or is_title_plus_text:
                    if calculate_iou_x(b_curr, b_next) > X_ALIGN_THRESH:
                        gap = b_next['y1'] - b_curr['y2'] 
                        if -10 < gap < Y_THRESH:
                            should_merge = True
                            if is_title_plus_text:
                                merge_target_group = 'text_group'
                
                if not should_merge and is_overlapping(b_curr, b_next, OVERLAP_THRESH):
                    should_merge = True
                
                if should_merge:
                    current_box = merge_two_boxes(current_box, next_box, merge_target_group)
                    current_group = get_group_name(current_box['class_name'])
                    merged_indices.add(j)
                    merged_in_this_round = True
                    break 
            
            if not merged_in_this_round:
                break
        
        final_results.append(current_box)
    
    return final_results


# ================= Visualization Functions =================

def visualize_page(full_image_path, bboxes, output_path):
    if not os.path.exists(full_image_path):
        print(f"[Warning] Image not found, skip drawing: {full_image_path}")
        return

    img = cv2.imread(full_image_path)
    if img is None:
        print(f"[Error] Failed to read image: {full_image_path}")
        return

    for bbox in bboxes:
        box = bbox['box']
        cls_name = bbox['class_name']
        box_id = bbox.get('bbox_id', '?')
        
        x1, y1 = int(box['x1']), int(box['y1'])
        x2, y2 = int(box['x2']), int(box['y2'])
        
        color = VIS_COLORS.get(cls_name, VIS_COLORS['default'])
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"{cls_name} ({box_id})"
        font_scale = 0.5
        thickness = 1
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        text_y = y1 - 5 if y1 - 20 > 0 else y1 + h + 5
        cv2.rectangle(img, (x1, text_y - h - 5), (x1 + w, text_y + 5), color, -1)
        cv2.putText(img, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    cv2.imwrite(output_path, img)

# ================= Main Program =================

def main():
    # if not os.path.exists(VIS_DIR):
    #     os.makedirs(VIS_DIR)
    
    total_processed = 0
    saved_count = 0
    vis_count = 0
    
    print("Starting processing...")
    
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
            with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
                lines = f_in.readlines()
                iterator = lines
                if 'tqdm' in globals():
                    iterator = tqdm(lines, desc="Processing Pages")
                
                for line in iterator:
                    line = line.strip()
                    if not line: continue
                    
                    data = json.loads(line)
                    total_processed += 1
                    
                    # 1. merge
                    original_bboxes = data.get('bboxes', [])
                    merged_bboxes = process_single_page(original_bboxes)

                    # 2. Reset bbox_id and build ID mapping
                    id_map = {} # old_id -> new_id
                    
                    for new_idx, box in enumerate(merged_bboxes):
                        old_ids = box.pop('original_ids', [])
                        for old_id in old_ids:
                            id_map[old_id] = new_idx
                        
                        box['bbox_id'] = new_idx
                    
                    # 3. Update relevant_bbox_ids in QAs
                    qas = data.get('QAs', [])
                    for qa in qas:
                        old_relevant_ids = qa.get('relevant_bbox_ids', [])
                        new_relevant_ids = set()
                        
                        for old_id in old_relevant_ids:
                            if old_id in id_map:
                                new_relevant_ids.add(id_map[old_id])
                        
                        qa['relevant_bbox_ids'] = sorted(list(new_relevant_ids))
                    
                    old_relevant_ids = data.get('relevant_bbox_ids', [])
                    new_relevant_ids = set()
                    for old_id in old_relevant_ids:
                        if old_id in id_map:
                            new_relevant_ids.add(id_map[old_id])
                    if 'relevant_bbox_ids' in data:
                        data['relevant_bbox_ids'] = sorted(list(new_relevant_ids))
                    
                    # 4. Save data
                    data['bboxes'] = merged_bboxes
                    data['QAs'] = qas
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                    saved_count += 1
                    
                    # 5. Visualization (first 20 valid pages)
                    # if vis_count < 20:
                    #     raw_img_path = data.get('image_path', '')
                        
                    #     if IMG_ROOT_DIR:
                    #         if raw_img_path.startswith('/'):
                    #             rel_path = raw_img_path.lstrip('/')
                    #             full_img_path = os.path.join(IMG_ROOT_DIR, rel_path)
                    #         else:
                    #             full_img_path = os.path.join(IMG_ROOT_DIR, raw_img_path)
                    #     else:
                    #         full_img_path = raw_img_path

                    #     save_name = f"vis_{data.get('doc_id', 'unk')}_{data.get('page_id', saved_count)}.jpg"
                    #     save_path = os.path.join(VIS_DIR, save_name)
                        
                    #     visualize_page(full_img_path, merged_bboxes, save_path)
                    #     vis_count += 1
                    
    except FileNotFoundError:
        print(f"Error: Input file not found {INPUT_FILE}")
        return

    print(f"Processing complete!")
    print(f"Total pages read: {total_processed}")
    print(f"Pages saved: {saved_count}")
    print(f"Pages discarded: {total_processed - saved_count}")
    print(f"Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
