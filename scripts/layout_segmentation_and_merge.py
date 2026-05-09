import os
import cv2
import json
import torch
import argparse
from tqdm import tqdm
from copy import deepcopy
from doclayout_yolo import YOLOv10

# ================= Merge Parameters =================

Y_THRESH = 40
X_ALIGN_THRESH = 0.7
OVERLAP_THRESH = 0.9

# ================= Semantic Merge Groups =================

MERGE_GROUPS = {
    'figure_group': {'figure', 'figure_caption'},
    'table_group':  {'table', 'table_caption', 'table_footnote'},
    'text_group':   {'plain text', 'isolate_formula', 'formula_caption'},
    'title_group':  {'title'}
}

GROUP_PRIORITY = {
    'figure_group': 'figure',
    'table_group': 'table',
    'text_group': 'plain text',
    'title_group': 'title'
}

# ================= Visualization Colors =================

VIS_COLORS = {
    'plain text': (200, 200, 200),
    'title': (0, 0, 255),
    'figure': (0, 255, 0),
    'table': (255, 0, 0),
    'isolate_formula': (0, 255, 255),
    'abandon': (160, 160, 160),
    'default': (100, 100, 100)
}

# ================= Core Functions =================

def get_group_name(class_name):
    for group_name, members in MERGE_GROUPS.items():
        if class_name in members:
            return group_name
    return None


def calculate_iou_x(box1, box2):
    x1 = max(box1['x1'], box2['x1'])
    x2 = min(box1['x2'], box2['x2'])

    overlap = max(0, x2 - x1)

    width1 = box1['x2'] - box1['x1']
    width2 = box2['x2'] - box2['x1']

    min_width = min(width1, width2)

    if min_width <= 0:
        return 0

    return overlap / min_width


def is_overlapping(box1, box2, threshold=0.9):

    x_overlap = max(
        0,
        min(box1['x2'], box2['x2']) - max(box1['x1'], box2['x1'])
    )

    y_overlap = max(
        0,
        min(box1['y2'], box2['y2']) - max(box1['y1'], box2['y1'])
    )

    intersection = x_overlap * y_overlap

    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])

    min_area = min(area1, area2)

    if min_area <= 0:
        return False

    return (intersection / min_area) > threshold


def merge_two_boxes(base_box, merge_box, target_group_name):

    b1 = base_box['box']
    b2 = merge_box['box']

    base_box['box']['x1'] = min(b1['x1'], b2['x1'])
    base_box['box']['y1'] = min(b1['y1'], b2['y1'])
    base_box['box']['x2'] = max(b1['x2'], b2['x2'])
    base_box['box']['y2'] = max(b1['y2'], b2['y2'])

    base_box['confidence'] = max(
        base_box['confidence'],
        merge_box['confidence']
    )

    if target_group_name and target_group_name in GROUP_PRIORITY:
        base_box['class_name'] = GROUP_PRIORITY[target_group_name]

    if 'original_ids' in merge_box:
        base_box['original_ids'].extend(merge_box['original_ids'])
    else:
        base_box['original_ids'].append(merge_box['bbox_id'])

    return base_box


def process_single_page(bboxes):

    valid_bboxes = []

    for b in bboxes:

        if b['class_name'] != 'abandon':

            b_copy = deepcopy(b)

            if 'original_ids' not in b_copy:
                b_copy['original_ids'] = [b['bbox_id']]

            valid_bboxes.append(b_copy)

    valid_bboxes.sort(
        key=lambda x: (x['box']['y1'], x['box']['x1'])
    )

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

                is_same_group = (
                    current_group is not None and
                    current_group == next_group
                )

                is_title_plus_text = (
                    current_group == 'title_group' and
                    next_group == 'text_group'
                )

                if is_same_group or is_title_plus_text:

                    if calculate_iou_x(b_curr, b_next) > X_ALIGN_THRESH:

                        gap = b_next['y1'] - b_curr['y2']

                        if -10 < gap < Y_THRESH:

                            should_merge = True

                            if is_title_plus_text:
                                merge_target_group = 'text_group'

                if not should_merge and is_overlapping(
                    b_curr,
                    b_next,
                    OVERLAP_THRESH
                ):
                    should_merge = True

                if should_merge:

                    current_box = merge_two_boxes(
                        current_box,
                        next_box,
                        merge_target_group
                    )

                    current_group = get_group_name(
                        current_box['class_name']
                    )

                    merged_indices.add(j)

                    merged_in_this_round = True

                    break

            if not merged_in_this_round:
                break

        final_results.append(current_box)

    return final_results


# ================= Visualization =================

def visualize_page(image, bboxes, output_path):

    for bbox in bboxes:

        box = bbox['box']

        cls_name = bbox['class_name']
        box_id = bbox['bbox_id']

        x1 = int(box['x1'])
        y1 = int(box['y1'])
        x2 = int(box['x2'])
        y2 = int(box['y2'])

        color = VIS_COLORS.get(
            cls_name,
            VIS_COLORS['default']
        )

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        label = f"{cls_name} ({box_id})"

        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imwrite(output_path, image)


# ================= Main =================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        default='./models/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt',
        type=str
    )

    parser.add_argument(
        '--image_dir',
        required=True,
        type=str
    )

    parser.add_argument(
        '--output_dir',
        default='./outputs',
        type=str
    )

    parser.add_argument(
        '--jsonl_path',
        default='./outputs/layout_results.jsonl',
        type=str
    )

    parser.add_argument(
        '--imgsz',
        default=1024,
        type=int
    )

    parser.add_argument(
        '--conf',
        default=0.2,
        type=float
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    raw_vis_dir = os.path.join(args.output_dir, "raw_vis")
    merged_vis_dir = os.path.join(args.output_dir, "merged_vis")

    os.makedirs(raw_vis_dir, exist_ok=True)
    os.makedirs(merged_vis_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    model = YOLOv10(args.model)

    image_files = sorted([
        f for f in os.listdir(args.image_dir)
        if f.lower().endswith((
            '.jpg',
            '.jpeg',
            '.png'
        ))
    ])

    with open(args.jsonl_path, "w", encoding="utf-8") as fout:

        for image_name in tqdm(image_files):

            image_path = os.path.join(
                args.image_dir,
                image_name
            )

            det_res = model.predict(
                image_path,
                imgsz=args.imgsz,
                conf=args.conf,
                device=device
            )

            # ================= Raw Visualization =================

            annotated_frame = det_res[0].plot(
                pil=True,
                line_width=3,
                font_size=18
            )

            raw_output_path = os.path.join(
                raw_vis_dir,
                image_name
            )

            cv2.imwrite(
                raw_output_path,
                annotated_frame
            )

            # ================= Parse Raw BBoxes =================

            bboxes = []

            for i, result in enumerate(det_res[0].process()):

                class_name = result['name']
                confidence = float(result['confidence'])
                box = result['box']

                bboxes.append({
                    "bbox_id": i,
                    "class_name": class_name,
                    "box": {
                        "x1": float(box["x1"]),
                        "y1": float(box["y1"]),
                        "x2": float(box["x2"]),
                        "y2": float(box["y2"])
                    },
                    "confidence": confidence
                })

            # ================= Merge =================

            merged_bboxes = process_single_page(bboxes)

            for idx, bbox in enumerate(merged_bboxes):
                bbox["bbox_id"] = idx

                if "original_ids" in bbox:
                    del bbox["original_ids"]

            # ================= Save Visualization =================

            image = cv2.imread(image_path)

            merged_output_path = os.path.join(
                merged_vis_dir,
                image_name
            )

            visualize_page(
                image,
                merged_bboxes,
                merged_output_path
            )

            # ================= Save JSONL =================

            sample = {
                "image_path": image_name,
                "bboxes": merged_bboxes
            }

            fout.write(
                json.dumps(sample, ensure_ascii=False) + "\n"
            )

    print(f"Results saved to: {args.jsonl_path}")