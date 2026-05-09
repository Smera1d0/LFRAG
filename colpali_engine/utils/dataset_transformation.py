import os
import json
import random
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from typing import List, Dict, Iterable

DATASET_CONFIGS = {
    "BBox_DocVQA": {
        "jsonl": "BBox_DocVQA_train.jsonl",
        "dir": "BBox_DocVQA",
        "train_size": 16218,
        "eval_size": 0,
    },
    "docvqa": {
        "jsonl": "docvqa_train.jsonl",
        "dir": "docvqa",
        "train_size": 29793,
        "eval_size": 0,
    },
    "dude": {
        "jsonl": "dude_train.jsonl",
        "dir": "dude",
        "train_size": 9871,
        "eval_size": 0,
    },
    "infographicsvqa": {
        "jsonl": "infovqa_train.jsonl",
        "dir": "infographicsvqa",
        "train_size": 13331,
        "eval_size": 0,
    },
    "docmatix": {
        "jsonl": "docmatix_train.jsonl",
        "dir": "docmatix",
        "train_size": 50787,
        "eval_size": 10000,
    },
}
# 16218+29793+9871+13331=69213
# +50787=120000


class JsonlCropDataset(Dataset):
    def __init__(self, jsonl_path, dir_path, train_size, eval_size, type="train", seed=42):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.type = type.lower()
        self.seed = seed
        self.train_size = train_size
        self.eval_size = eval_size
        self.dir_path = dir_path
        
        if self.type not in ["train", "eval", "test"]:
            raise ValueError(f"Invalid type: {type}, must be one of ['train', 'val', 'test']")
        
        print(f"Loading and shuffling data from {jsonl_path}...")
        self.all_data = self._load_and_shuffle_data()
        
        self.data = self._split_dataset(self.train_size, self.eval_size)

    def _load_and_shuffle_data(self):
        all_data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f: 
            for line in f:
                line = line.strip()
                if line: 
                    all_data.append(json.loads(line))
        
        random.seed(self.seed)
        random.shuffle(all_data) 
        return all_data

    def _split_dataset(self, train_size, eval_size):
        if self.type == "train":
            return self.all_data[:train_size]
        elif self.type == "eval":
            return self.all_data[train_size:train_size+eval_size]
        elif self.type == "test":
            return self.all_data[train_size+eval_size:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        image_path = os.path.join(self.dir_path, data["image_path"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        W, H = image.size

        extra_image = image.copy()
        draw = ImageDraw.Draw(extra_image)
        crops = []

        for bbox in data["bboxes"]:
            x1 = bbox["box"]["x1"]
            y1 = bbox["box"]["y1"]
            x2 = bbox["box"]["x2"]
            y2 = bbox["box"]["y2"]
            crop = image.crop((x1, y1, x2, y2))
            crops.append({
                "bbox_id": bbox["bbox_id"],
                "class_name": bbox["class_name"],
                "confidence": bbox["confidence"],
                "crop_image": crop
            })
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))

        crops.append({
            "bbox_id": len(data["bboxes"]),
            "class_name": "extra",
            "confidence": None,
            "crop_image": extra_image
        })

        return {
            "image_path": image_path,
            "image": image,         
            "W": W,
            "H": H,
            "crops": crops,            # list [dict]
            "crop_counts": len(crops),  # include extra crop
            "question": data["question"],
            "answer": data["answer"],
            "relevant_bbox_ids": data["relevant_bbox_ids"],
        }



def load_multi_jsonl_datasets(
    dataset_root: str,
    split: str = "train",
    seed: int = 42,
    dataset_configs=DATASET_CONFIGS, 
):
    datasets = []

    for dataset_name, cfg in dataset_configs.items():
        jsonl_path = os.path.join(dataset_root, cfg["jsonl"])
        image_dir = dataset_root

        train_size = cfg["train_size"]
        eval_size = cfg["eval_size"]

        print(
            f"Loading {dataset_name} | "
            f"train_size={train_size}, eval_size={eval_size}"
        )

        ds = JsonlCropDataset(
            jsonl_path=jsonl_path,
            dir_path=image_dir,
            train_size=train_size,
            eval_size=eval_size,
            type=split,
            seed=seed,
        )
        ds.dataset_name = dataset_name
        datasets.append(ds)
    return ConcatDataset(datasets)

def load_docmatix_eval_datasets(dataset_path, dir_path, eval_size) -> JsonlCropDataset:
    train_size = DATASET_CONFIGS["docmatix"]["train_size"]
    eval_size = eval_size if eval_size != 0 else DATASET_CONFIGS["docmatix"]["eval_size"]
    return JsonlCropDataset(dataset_path, dir_path, train_size, eval_size, type = "eval")