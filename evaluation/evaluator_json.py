from __future__ import annotations
import json
import os
import logging
from typing import Dict, List, Optional, Union, Tuple
from datasets import Dataset, Image
from PIL import Image as PILImage

from evaluation.base_evaluator import BaseViDoReEvaluator
from evaluation.evaluator_beir import ViDoReEvaluatorBEIR, BEIRDataset
from retrievers.base_vision_retriever import BaseVisionRetriever

logger = logging.getLogger(__name__)

class EvaluatorJSONL(BaseViDoReEvaluator):
    """
    Evaluator for a custom JSONL dataset format containing images and bounding boxes.
    Calculates two metrics:
    1. Page-level retrieval (finding the correct image for a question)
    2. Bbox-level retrieval (finding the correct bounding box crop for a question)
    """

    def __init__(
        self, 
        vision_retriever: BaseVisionRetriever, 
        jsonl_path: str, 
        image_dir: str
    ):
        super().__init__(vision_retriever)
        self.jsonl_path = jsonl_path
        self.image_dir = image_dir
        
    def _save_results_jsonl(
        self,
        results: Dict[str, Dict[str, float]],
        data: List[Dict],
        save_path: str,
        top_k: int,
        level: str,
    ):
        logger.info(f"Saving {level}-level retrieval results to {save_path}")
        
        # Build image_bbox_map for bbox retrieval
        image_bbox_map = {}
        if level == "bbox":
            for item in data:
                img_path = item.get("image_path")
                if img_path and "bboxes" in item:
                     if img_path not in image_bbox_map:
                         image_bbox_map[img_path] = {str(b["bbox_id"]): b["box"] for b in item["bboxes"]}
        
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            for query_id, scores_dict in results.items():
                try:
                    idx = int(query_id)
                    item = data[idx]
                except (ValueError, IndexError):
                    continue
                
                query_text = item["question"]
                gold_answers = item.get("answers", [])
                if not gold_answers and "answer" in item:
                    gold_answers = [item["answer"]]
                
                sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
                
                retrieved_items = []
                for rank, (corpus_id, score) in enumerate(sorted_scores, start=1):
                    result_item = {
                        "rank": rank,
                        "score": score,
                        "corpus_id": corpus_id
                    }
                    
                    if level == "page":
                        result_item["image_path"] = corpus_id
                    elif level == "bbox":
                        # corpus_id is {image_path}_{bbox_id}
                        sep_idx = corpus_id.rfind("_")
                        if sep_idx != -1:
                            img_path = corpus_id[:sep_idx]
                            bbox_id = corpus_id[sep_idx+1:]
                            result_item["image_path"] = img_path
                            result_item["bbox_id"] = bbox_id
                            
                            if img_path in image_bbox_map and bbox_id in image_bbox_map[img_path]:
                                result_item["box"] = image_bbox_map[img_path][bbox_id]
                        else:
                            result_item["image_path"] = corpus_id
                    
                    retrieved_items.append(result_item)
                
                output_item = {
                    "query_id": query_id,
                    "query": query_text,
                    "ground_truth": gold_answers,
                    "retrieved_items": retrieved_items
                }
                f.write(json.dumps(output_item, ensure_ascii=False) + "\n")

    def evaluate_dataset(
        self,
        batch_query: int = 8,
        batch_passage: int = 8,
        batch_score: Optional[int] = None,
        save_results_path: Optional[str] = None,
        top_k: int = 10,
        **kwargs,
    ) -> Dict[str, Optional[float]]:
        """
        Evaluate the dataset specified in jsonl_path.
        
        Args:
            batch_query (int): Batch size for queries.
            batch_passage (int): Batch size for passages (images/crops).
            batch_score (Optional[int]): Batch size for scoring.
            save_results_path (Optional[str]): Path to save the retrieval results.
            top_k (int): Number of top results to save.
            
        Returns:
            Dict[str, Optional[float]]: Dictionary containing combined metrics for page and bbox retrieval.
        """
        
        # Load JSONL data
        logger.info(f"Loading data from {self.jsonl_path}")
        data = []
        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
                    
        level = kwargs.pop("level", "both")
        if level not in {"page", "bbox", "both"}:
            raise ValueError(f"Invalid level={level}. Must be one of: page, bbox, both.")

        results: Dict[str, Optional[float]] = {}

        if level in {"page", "both"}:
            logger.info("Starting Page-level evaluation...")
            if getattr(self.vision_retriever, "bbox_only", False):
                metrics_page, results_page = self._evaluate_page_via_bbox_mapping(
                    data=data,
                    batch_query=batch_query,
                    batch_passage=batch_passage,
                    batch_score=batch_score,
                    return_results=True,
                    **kwargs,
                )
            else:
                page_ds = self._prepare_page_dataset(data)
                evaluator_page = ViDoReEvaluatorBEIR(
                    vision_retriever=self.vision_retriever,
                    corpus_id_column="corpus_id",
                    query_id_column="query_id",
                    query_column="query",
                    passage_column="image" if self.vision_retriever.use_visual_embedding else "text_description",
                    score_column="score",
                )
                metrics_page, results_page = evaluator_page.evaluate_dataset(
                    page_ds,
                    batch_query=batch_query,
                    batch_passage=batch_passage,
                    batch_score=batch_score,
                    return_results=True,
                    **kwargs,
                )
            
            if save_results_path:
                path = save_results_path.replace(".jsonl", "_page.jsonl")
                self._save_results_jsonl(results_page, data, save_path=path, top_k=top_k, level="page")

            for k, v in metrics_page.items():
                results[f"page_{k}"] = v

        if level in {"bbox", "both"}:
            logger.info("Starting Bbox-level evaluation...")
            bbox_ds = self._prepare_bbox_dataset(data)
            evaluator_bbox = ViDoReEvaluatorBEIR(
                vision_retriever=self.vision_retriever,
                corpus_id_column="corpus_id",
                query_id_column="query_id",
                query_column="query",
                passage_column="image" if self.vision_retriever.use_visual_embedding else "text_description",
                score_column="score"
            )
            metrics_bbox, results_bbox = evaluator_bbox.evaluate_dataset(
                bbox_ds,
                batch_query=batch_query,
                batch_passage=batch_passage,
                batch_score=batch_score,
                return_results=True,
                **kwargs
            )
            
            if save_results_path:
                path = save_results_path.replace(".jsonl", "_bbox.jsonl")
                self._save_results_jsonl(results_bbox, data, save_path=path, top_k=top_k, level="bbox")

            for k, v in metrics_bbox.items():
                results[f"bbox_{k}"] = v

        return results

    @staticmethod
    def _to_tag(class_name: Optional[str]) -> str:
        if not class_name:
            return "<abandon>"
        class_name = class_name.strip()
        if not class_name:
            return "<abandon>"
        if class_name.startswith("<") and class_name.endswith(">"):
            return class_name
        return f"<{class_name}>"

    def _evaluate_page_via_bbox_mapping(
        self,
        data: List[Dict],
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
        return_results: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Optional[float]], Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, float]]]]:
        bbox_ds = self._prepare_bbox_dataset(data)

        evaluator_bbox = ViDoReEvaluatorBEIR(
            vision_retriever=self.vision_retriever,
            corpus_id_column="corpus_id",
            query_id_column="query_id",
            query_column="query",
            passage_column="image" if self.vision_retriever.use_visual_embedding else "text_description",
            score_column="score",
        )

        ds_corpus = bbox_ds["corpus"]
        ds_queries = bbox_ds["queries"]

        passage_ids: List[str] = [str(elt) for elt in ds_corpus["corpus_id"]]
        query_ids: List[str] = [str(elt) for elt in ds_queries["query_id"]]

        query_embeddings = evaluator_bbox._get_query_embeddings(
            ds=ds_queries,
            query_column="query",
            batch_query=batch_query,
            dataloader_prebatch_size=kwargs.get("dataloader_prebatch_query"),
        )
        passage_embeddings = evaluator_bbox._get_passage_embeddings(
            ds=ds_corpus,
            passage_column="image" if self.vision_retriever.use_visual_embedding else "text_description",
            batch_passage=batch_passage,
            dataloader_prebatch_size=kwargs.get("dataloader_prebatch_passage"),
        )

        scores = self.vision_retriever.get_scores(
            query_embeddings=query_embeddings,
            passage_embeddings=passage_embeddings,
            batch_size=batch_score,
        )
        results_bbox = evaluator_bbox._get_retrieval_results(
            query_ids=query_ids,
            passage_ids=passage_ids,
            scores=scores,
        )

        all_page_ids = {item["image_path"] for item in data if item.get("image_path")}
        results_page: Dict[str, Dict[str, float]] = {}
        for query_id, bbox_scores in results_bbox.items():
            page_scores: Dict[str, float] = {}
            for bbox_corpus_id, score in bbox_scores.items():
                page_id = bbox_corpus_id.rsplit("_", 1)[0]
                prev = page_scores.get(page_id)
                if prev is None or score > prev:
                    page_scores[page_id] = score
            for page_id in all_page_ids:
                page_scores.setdefault(page_id, 0.0)
            results_page[query_id] = page_scores

        qrels_page: Dict[str, Dict[str, int]] = {}
        for idx, item in enumerate(data):
            query_id = str(idx)
            corpus_id = item["image_path"]
            qrels_page[query_id] = {corpus_id: 1}

        metrics = self.compute_retrieval_scores(qrels=qrels_page, results=results_page, ignore_identical_ids=False)
        if return_results:
            return metrics, results_page
        return metrics

    def _prepare_page_dataset(self, data: List[Dict]) -> BEIRDataset:
        corpus_dict: Dict[str, Dict[str, str]] = {}
        queries_dict: Dict[str, str] = {}
        qrels_list: List[Dict[str, object]] = []

        for idx, item in enumerate(data):
            query_id = str(idx)
            queries_dict[query_id] = item["question"]

            image_path = item["image_path"]
            corpus_id = image_path

            if corpus_id not in corpus_dict:
                if self.vision_retriever.use_visual_embedding:
                    full_image_path = os.path.join(self.image_dir, image_path)
                    if getattr(self.vision_retriever, "pool_global_from_crops", False):
                        try:
                            img_obj = PILImage.open(full_image_path).convert("RGB")
                        except Exception:
                            continue
                        crops: List[PILImage.Image] = []
                        tags: List[str] = []
                        for bbox in item.get("bboxes", []):
                            box = bbox.get("box", {})
                            try:
                                crop = img_obj.crop((box["x1"], box["y1"], box["x2"], box["y2"]))
                            except Exception:
                                continue
                            crops.append(crop)
                            tags.append(self._to_tag(bbox.get("class_name")))
                        if not crops:
                            crops = [img_obj]
                            tags = [""]
                        if getattr(self.vision_retriever, "use_global_for_bbox", False):
                            corpus_dict[corpus_id] = {
                                "image": {"global_path": full_image_path, "crops": crops, "tags": tags}
                            }
                        else:
                            corpus_dict[corpus_id] = {"image": {"crops": crops, "tags": tags}}
                    else:
                        corpus_dict[corpus_id] = {"image": full_image_path}
                else:
                    corpus_dict[corpus_id] = {"text_description": item.get("text_description", "")}

            qrels_list.append(
                {
                    "query_id": query_id,
                    "corpus_id": corpus_id,
                    "score": 1,
                }
            )

        corpus_data = [{"corpus_id": k, **v} for k, v in corpus_dict.items()]
        queries_data = [{"query_id": k, "query": v} for k, v in queries_dict.items()]

        ds_corpus = Dataset.from_list(corpus_data)
        if self.vision_retriever.use_visual_embedding:
            if not getattr(self.vision_retriever, "pool_global_from_crops", False):
                ds_corpus = ds_corpus.cast_column("image", Image())
        ds_queries = Dataset.from_list(queries_data)
        ds_qrels = Dataset.from_list(qrels_list)

        return {"corpus": ds_corpus, "queries": ds_queries, "qrels": ds_qrels}

    def _prepare_bbox_dataset(self, data: List[Dict]) -> BEIRDataset:
        corpus_dict: Dict[str, Dict[str, object]] = {}
        queries_dict: Dict[str, str] = {}
        qrels_list: List[Dict[str, object]] = []

        for idx, item in enumerate(data):
            query_id = str(idx)
            queries_dict[query_id] = item["question"]

            image_path = item["image_path"]
            full_image_path = os.path.join(self.image_dir, image_path)

            bboxes = item.get("bboxes", [])
            relevant_ids = set(item.get("relevant_bbox_ids", []))

            img_obj = None

            for bbox in bboxes:
                bbox_id = bbox["bbox_id"]
                corpus_id = f"{image_path}_{bbox_id}"
                tag = self._to_tag(bbox.get("class_name"))

                if corpus_id not in corpus_dict:
                    if self.vision_retriever.use_visual_embedding:
                        if img_obj is None:
                            try:
                                img_obj = PILImage.open(full_image_path).convert("RGB")
                            except Exception:
                                continue
                        box = bbox.get("box", {})
                        try:
                            crop = img_obj.crop((box["x1"], box["y1"], box["x2"], box["y2"]))
                        except Exception:
                            crop = None
                        if crop is not None:
                            if getattr(self.vision_retriever, "use_global_for_bbox", False):
                                corpus_dict[corpus_id] = {
                                    "image": {"crop": crop, "global_path": full_image_path},
                                    "tags": tag,
                                }
                            else:
                                corpus_dict[corpus_id] = {"image": crop, "tags": tag}
                    else:
                        corpus_dict[corpus_id] = {"text_description": bbox.get("text_description", "")}

                if bbox_id in relevant_ids:
                    qrels_list.append(
                        {
                            "query_id": query_id,
                            "corpus_id": corpus_id,
                            "score": 1,
                        }
                    )

        corpus_data = [{"corpus_id": k, **v} for k, v in corpus_dict.items()]
        queries_data = [{"query_id": k, "query": v} for k, v in queries_dict.items()]

        ds_corpus = Dataset.from_list(corpus_data)
        ds_queries = Dataset.from_list(queries_data)
        ds_qrels = Dataset.from_list(qrels_list)

        return {"corpus": ds_corpus, "queries": ds_queries, "qrels": ds_qrels}
