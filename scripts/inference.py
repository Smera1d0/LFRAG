import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from retrievers.registry_utils import load_vision_retriever_from_registry


def _to_tag(class_name: Optional[str]) -> str:
    if not class_name:
        return "<abandon>"
    class_name = class_name.strip()
    if not class_name:
        return "<abandon>"
    if class_name.startswith("<") and class_name.endswith(">"):
        return class_name
    return f"<{class_name}>"


def load_corpus(jsonl_path: str, image_dir: str):
    crops: List[Image.Image] = []
    tags: List[str] = []
    metadata: List[Dict] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            image_path = item["image_path"]
            full_path = os.path.join(image_dir, image_path)

            img = None
            for bbox in item.get("bboxes", []):
                box = bbox.get("box", {})
                try:
                    if img is None:
                        img = Image.open(full_path).convert("RGB")
                    crop = img.crop((box["x1"], box["y1"], box["x2"], box["y2"]))
                except Exception:
                    continue

                crops.append(crop)
                tags.append(_to_tag(bbox.get("class_name")))
                metadata.append({
                    "image_path": image_path,
                    "bbox_id": bbox["bbox_id"],
                    "class_name": bbox.get("class_name", ""),
                    "box": box,
                    "confidence": bbox.get("confidence", 0.0),
                })

    return crops, tags, metadata


def build_index(args):
    """Pre-compute passage embeddings and save to disk."""
    importlib.import_module("retrievers.lfrag_retriever")
    retriever = load_vision_retriever_from_registry(
        model_class="lfrag_retriever",
        pretrained_model_name_or_path=args.model_path,
        device=args.device,
        base_model_name_or_path=args.base_model_path,
    )

    print(f"Loading corpus from {args.jsonl_path} ...")
    crops, tags, metadata = load_corpus(args.jsonl_path, args.image_dir)
    print(f"Loaded {len(crops)} document blocks.")

    if not crops:
        print("No document blocks found.")
        return

    print("Encoding document blocks...")
    passage_embeddings = retriever.forward_passages(crops, batch_size=args.batch_passage, tags=tags)

    index_path = Path(args.index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "embeddings": passage_embeddings,
        "metadata": metadata,
    }, str(index_path))
    print(f"Index saved to {index_path} ({len(crops)} blocks)")


def query_index(args):
    """Load cached embeddings and retrieve top-k blocks for a query."""
    importlib.import_module("retrievers.lfrag_retriever")
    retriever = load_vision_retriever_from_registry(
        model_class="lfrag_retriever",
        pretrained_model_name_or_path=args.model_path,
        device=args.device,
        base_model_name_or_path=args.base_model_path,
    )

    print(f"Loading index from {args.index_path} ...")
    index = torch.load(args.index_path, weights_only=False)
    passage_embeddings = index["embeddings"]
    metadata = index["metadata"]
    print(f"Loaded {len(metadata)} blocks from index.")

    print("Encoding query...")
    query_embeddings = retriever.forward_queries([args.query], batch_size=1)

    print("Computing scores...")
    scores = retriever.get_scores(query_embeddings, passage_embeddings, batch_size=args.batch_score)
    scores_list = scores[0].tolist()

    ranked_indices = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
    topk_indices = ranked_indices[:args.topk]

    print(f"\n{'='*80}")
    print(f"Query: {args.query}")
    print(f"Top-{args.topk} retrieved blocks:")
    print(f"{'='*80}")
    print(f"{'Rank':<6}{'Score':<10}{'Image':<30}{'BBox':<6}{'Class':<15}{'Box'}")
    print(f"{'-'*80}")

    results = []
    for rank, idx in enumerate(topk_indices, 1):
        meta = metadata[idx]
        score = scores_list[idx]
        box = meta["box"]
        box_str = f"({box['x1']:.0f}, {box['y1']:.0f}, {box['x2']:.0f}, {box['y2']:.0f})"
        print(f"{rank:<6}{score:<10.4f}{meta['image_path']:<30}{meta['bbox_id']:<6}{meta['class_name']:<15}{box_str}")
        results.append({
            "rank": rank, "score": score,
            "image_path": meta["image_path"], "bbox_id": meta["bbox_id"],
            "class_name": meta["class_name"], "box": box,
        })

    if args.save_crops_dir:
        os.makedirs(args.save_crops_dir, exist_ok=True)
        for rank, idx in enumerate(topk_indices, 1):
            meta = metadata[idx]
            box = meta["box"]
            full_path = os.path.join(args.image_dir, meta["image_path"])
            img = Image.open(full_path).convert("RGB")
            crop = img.crop((box["x1"], box["y1"], box["x2"], box["y2"]))
            filename = f"rank{rank}_{meta['image_path'].replace('/', '_')}_{meta['bbox_id']}.jpg"
            crop.save(os.path.join(args.save_crops_dir, filename))
        print(f"\nCrop images saved to {args.save_crops_dir}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {"query": args.query, "topk": args.topk, "results": results}
        output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Results saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="LFRAG Inference: retrieve top-k document blocks for a query")
    subparsers = parser.add_subparsers(dest="command")

    # --- build-index ---
    p_index = subparsers.add_parser("build-index", help="Pre-compute passage embeddings")
    p_index.add_argument("--jsonl_path", type=str, required=True)
    p_index.add_argument("--image_dir", type=str, required=True)
    p_index.add_argument("--index_path", type=str, default="./index.pt", help="Where to save the index")
    p_index.add_argument("--model_path", type=str, default="./ckpts/lfrag")
    p_index.add_argument("--base_model_path", type=str, default="./models/colqwen2.5-v0.2-merged")
    p_index.add_argument("--device", type=str, default="auto")
    p_index.add_argument("--batch_passage", type=int, default=8)

    # --- query ---
    p_query = subparsers.add_parser("query", help="Retrieve top-k blocks for a query")
    p_query.add_argument("--query", type=str, required=True)
    p_query.add_argument("--index_path", type=str, default="./index.pt")
    p_query.add_argument("--image_dir", type=str, default=None, help="Required if --save_crops_dir is set")
    p_query.add_argument("--model_path", type=str, default="./ckpts/lfrag")
    p_query.add_argument("--base_model_path", type=str, default="./models/colqwen2.5-v0.2-merged")
    p_query.add_argument("--topk", type=int, default=3)
    p_query.add_argument("--device", type=str, default="auto")
    p_query.add_argument("--batch_score", type=int, default=128)
    p_query.add_argument("--output", type=str, default=None)
    p_query.add_argument("--save_crops_dir", type=str, default=None)

    args = parser.parse_args()

    if args.command == "build-index":
        build_index(args)
    elif args.command == "query":
        if args.save_crops_dir and not args.image_dir:
            parser.error("--image_dir is required when --save_crops_dir is set")
        query_index(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
