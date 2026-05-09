import argparse
import importlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.evaluator_json import EvaluatorJSONL
from evaluation.evaluator_beir import ViDoReEvaluatorBEIR, load_beir_dataset
from retrievers.registry_utils import load_vision_retriever_from_registry

logger = logging.getLogger(__name__)

def _build_retriever(model: str, model_path: Optional[str], device: str) -> Any:
    if model_path:
        return load_vision_retriever_from_registry(
            model_class=model,
            pretrained_model_name_or_path=model_path,
            device=device,
        )
    return load_vision_retriever_from_registry(
        model_class=model,
        device=device,
    )


def _warn_if_missing_text_fields(jsonl_path: str) -> None:
    p = Path(jsonl_path)
    if not p.exists():
        return
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                has_page_text = bool(item.get("text_description"))
                has_bbox_text = any(bool(b.get("text_description")) for b in item.get("bboxes", []))
                if not has_page_text and not has_bbox_text:
                    logger.warning(
                        "The current jsonl does not appear to have the `text_description` field. "
                        "Non-vision retrieval models will evaluate with empty text. "
                        "Please first generate a jsonl with text_description using utils/ocr_jsonl.py, "
                        "then point --jsonl_path to it."
                    )
                break
    except Exception:
        return

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever",
        type=str,
        required=True,
        choices=[
            "lfrag_retriever",
            "siglip",
            "visrag-ret",
            "colpali",
            "colqwen2_5",
            "visdomrag",
            "bm25",
            "bge-m3",
            "nv-embed-v2",
        ],
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")   

    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="../datasets/eval/LF_Docmatix.jsonl",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="../datasets/LF_Docmatix",
    )

    parser.add_argument(
        "--beir_dir",
        type=str,
        default=None,
    )

    parser.add_argument("--batch_query", type=int, default=8)
    parser.add_argument("--batch_passage", type=int, default=8)
    parser.add_argument("--batch_score", type=int, default=None)
    parser.add_argument("--dataloader_prebatch_query", type=int, default=None)
    parser.add_argument("--dataloader_prebatch_passage", type=int, default=None)
    parser.add_argument("--level", type=str, default="both", choices=["page", "bbox", "both"])
    parser.add_argument("--save_results_path", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=10)

    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")

    parser.add_argument(
        "--json",
        action="store_true",
        help="Use JSONL format evaluator (EvaluatorJSONL).",
    )
    parser.add_argument(
        "--beir",
        action="store_true",
        help="Use BEIR format evaluator (ViDoReEvaluatorBEIR).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    if args.retriever == "colqwen2_5" and args.batch_score is None:
        args.batch_score = 128

    if args.json and args.beir:
        raise ValueError("Cannot specify both --json and --beir")
    if not args.json and not args.beir:
        args.json = True

    module_by_retriever = {
        "siglip": "retrievers.siglip_retriever",
        "visrag-ret": "retrievers.visrag_retriever",
        "colpali": "retrievers.colpali_retriever",
        "colqwen2_5": "retrievers.colqwen2_5_retriever",
        "bm25": "retrievers.bm25_retriever",
        "bge-m3": "retrievers.bge_m3_retriever",
        "nv-embed-v2": "retrievers.nv_embed_v2_retriever",
        "lfrag_retriever": "retrievers.lfrag_retriever"
    }
    importlib.import_module(module_by_retriever[args.retriever])

    if args.model_path is not None and not args.model_path.strip():
        args.model_path = None
    retriever = _build_retriever(model=args.retriever, model_path=args.model_path, device=args.device)

    if args.json:
        if not getattr(retriever, "use_visual_embedding", True):
            _warn_if_missing_text_fields(args.jsonl_path)

        evaluator = EvaluatorJSONL(
            vision_retriever=retriever,
            jsonl_path=args.jsonl_path,
            image_dir=args.image_dir,
        )

        metrics: Dict[str, Optional[float]] = evaluator.evaluate_dataset(
            batch_query=args.batch_query,
            batch_passage=args.batch_passage,
            batch_score=args.batch_score,
            dataloader_prebatch_query=args.dataloader_prebatch_query,
            dataloader_prebatch_passage=args.dataloader_prebatch_passage,
            level=args.level,
            save_results_path=args.save_results_path,
            top_k=args.top_k,
        )
    else:
        if not args.beir_dir:
            raise ValueError("--beir_dir must be provided in --beir mode")

        ds_beir = load_beir_dataset(
            beir_dir=args.beir_dir,
            use_visual_embedding=getattr(retriever, "use_visual_embedding", True),
            image_dir=args.image_dir,
        )
        evaluator_beir = ViDoReEvaluatorBEIR(vision_retriever=retriever)
        metrics = evaluator_beir.evaluate_dataset(
            ds_beir,
            batch_query=args.batch_query,
            batch_passage=args.batch_passage,
            batch_score=args.batch_score,
            dataloader_prebatch_query=args.dataloader_prebatch_query,
            dataloader_prebatch_passage=args.dataloader_prebatch_passage,
            save_results_path=args.save_results_path,
            top_k=args.top_k,
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
