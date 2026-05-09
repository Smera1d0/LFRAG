from __future__ import annotations

from collections import defaultdict
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypedDict, cast

import torch
from datasets import Dataset, Image, load_dataset

from evaluation.base_evaluator import BaseViDoReEvaluator
from retrievers.base_vision_retriever import BaseVisionRetriever
from retrievers.bm25_retriever import BM25Retriever


class BEIRDataset(TypedDict):
    """
    BEIR dataset type. A BEIR dataset must contain 3 subsets:
        corpus: The dataset containing the corpus of documents. Should contain the following columns:
            - corpus-id: The column containing the document IDs as integers.
            - image: The column containing the image data (PIL format).
        queries: The dataset containing the queries. Should contain the following columns:
            - query-id: The column containing the query IDs as integers.
            - query: The column containing the query text.
        qrels: The dataset containing the query relevance scores (TREC format). Should contain the following columns:
            - query-id: The column containing the query IDs as integers.
            - corpus-id: The column containing the document IDs as integers.
            - score: The column containing the relevance scores as integers.

    Note: In the TREC format used here, `score` is an integer indicating the relevance of the document to the query.
    For each query i, the relevance scores are integers in the range [0, N_i], where the higher the score, the more
    relevant the document is to the given query.
    """

    corpus: Dataset
    queries: Dataset
    qrels: Dataset


def _first_present(d: Mapping[str, Any], keys: Sequence[str]) -> Tuple[Optional[str], Any]:
    for k in keys:
        if k in d:
            return k, d[k]
    return None, None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(cast(Dict[str, Any], json.loads(line)))
    return records


def _resolve_path(base_dir: str, maybe_path: Any) -> Any:
    if isinstance(maybe_path, str):
        if os.path.isabs(maybe_path):
            return maybe_path
        return os.path.join(base_dir, maybe_path)
    if isinstance(maybe_path, Mapping):
        out = dict(maybe_path)
        p = out.get("path")
        if isinstance(p, str) and p and not os.path.isabs(p):
            out["path"] = os.path.join(base_dir, p)
        return out
    return maybe_path


def _dataset_keep_columns(ds: Dataset, keep: Sequence[str]) -> Dataset:
    drop = [c for c in ds.column_names if c not in set(keep)]
    if drop:
        return ds.remove_columns(drop)
    return ds


def _as_beir_dataset_from_flat(
    ds: Dataset,
    *,
    use_visual_embedding: bool,
    image_dir: str,
    corpus_id_column: str,
    query_id_column: str,
    query_column: str,
    score_column: str,
) -> BEIRDataset:
    if len(ds) == 0:
        raise ValueError("The dataset is empty, cannot construct BEIR dataset")

    query_src, _ = _first_present(ds.features, [query_column, "query", "question", "text"])
    if query_src is None:
        raise ValueError(f"Can't find query column, available columns: {ds.column_names}")

    corpus_id_src, _ = _first_present(
        ds.features,
        [corpus_id_column, "image_filename", "image_id", "image_path", "doc_id", "_id", "id"],
    )
    if corpus_id_src is None:
        corpus_id_src = "__row_index__"

    has_image = "image" in ds.column_names
    if use_visual_embedding and not has_image:
        if "image_filename" not in ds.column_names and "image_path" not in ds.column_names:
            raise ValueError(f"use_visual_embedding=True but missing image/image_filename column, available columns: {ds.column_names}")

    if corpus_id_src == "__row_index__":
        per_row_corpus_ids = [str(i) for i in range(len(ds))]
    else:
        per_row_corpus_ids = [str(x) for x in ds[corpus_id_src]]

    corpus_id_to_first_idx: Dict[str, int] = {}
    corpus_indices: List[int] = []
    for i, cid in enumerate(per_row_corpus_ids):
        if cid in corpus_id_to_first_idx:
            continue
        corpus_id_to_first_idx[cid] = i
        corpus_indices.append(i)

    ds_corpus = ds.select(corpus_indices)
    corpus_ids = [per_row_corpus_ids[i] for i in corpus_indices]
    ds_corpus = ds_corpus.add_column(corpus_id_column, corpus_ids)

    if use_visual_embedding:
        if not has_image:
            img_src, _ = _first_present(ds.features, ["image_filename", "image_path", "path"])
            if img_src is None:
                raise ValueError("Cannot construct image column for corpus")
            image_paths = [_resolve_path(image_dir, x) for x in ds_corpus[img_src]]
            ds_corpus = ds_corpus.add_column("image", image_paths).cast_column("image", Image())
        ds_corpus = _dataset_keep_columns(ds_corpus, [corpus_id_column, "image"])
    else:
        text_src, _ = _first_present(ds.features, ["markdown", "page", "text", "text_description", "contents"])
        if text_src is None:
            markdown = [""] * len(ds_corpus)
        else:
            markdown = ["" if x is None else str(x) for x in ds_corpus[text_src]]
        if "markdown" in ds_corpus.column_names:
            ds_corpus = ds_corpus.remove_columns(["markdown"])
        ds_corpus = ds_corpus.add_column("markdown", markdown)
        ds_corpus = _dataset_keep_columns(ds_corpus, [corpus_id_column, "markdown"])

    query_ids = [str(i) for i in range(len(ds))]
    queries = [str(x) for x in ds[query_src]]
    ds_queries = Dataset.from_dict({query_id_column: query_ids, query_column: queries})

    qrels_records = [
        {query_id_column: qid, corpus_id_column: cid, score_column: 1}
        for qid, cid in zip(query_ids, per_row_corpus_ids)
    ]
    ds_qrels = Dataset.from_list(qrels_records)

    return {"corpus": ds_corpus, "queries": ds_queries, "qrels": ds_qrels}


def load_beir_dataset(
    beir_dir: str,
    *,
    use_visual_embedding: bool,
    image_dir: Optional[str] = None,
    corpus_path: Optional[str] = None,
    queries_path: Optional[str] = None,
    qrels_path: Optional[str] = None,
    split: str = "test",
    corpus_id_column: str = "corpus_id",
    query_id_column: str = "query_id",
    query_column: str = "query",
    score_column: str = "score",
) -> BEIRDataset:
    if image_dir is None:
        image_dir = beir_dir

    if corpus_path is None:
        corpus_path = os.path.join(beir_dir, "corpus.jsonl")
    if queries_path is None:
        queries_path = os.path.join(beir_dir, "queries.jsonl")
    if qrels_path is None:
        candidate = os.path.join(beir_dir, "qrels", f"{split}.tsv")
        if os.path.exists(candidate):
            qrels_path = candidate
        else:
            qrels_path = os.path.join(beir_dir, "qrels.tsv")

    if not (os.path.exists(corpus_path) and os.path.exists(queries_path) and os.path.exists(qrels_path)):
        ds_flat = load_dataset(beir_dir, split=split)
        return _as_beir_dataset_from_flat(
            ds_flat,
            use_visual_embedding=use_visual_embedding,
            image_dir=image_dir,
            corpus_id_column=corpus_id_column,
            query_id_column=query_id_column,
            query_column=query_column,
            score_column=score_column,
        )

    corpus_raw = _read_jsonl(corpus_path)
    queries_raw = _read_jsonl(queries_path)

    corpus_records: List[Dict[str, Any]] = []
    for rec in corpus_raw:
        _, cid = _first_present(rec, [corpus_id_column, "_id", "doc_id", "id"])
        if cid is None:
            continue
        item: Dict[str, Any] = {corpus_id_column: str(cid)}
        if use_visual_embedding:
            _, img = _first_present(rec, ["image", "image_path", "path"])
            if img is None:
                continue
            item["image"] = _resolve_path(image_dir, img)
        else:
            _, text = _first_present(rec, ["markdown", "text", "text_description", "contents"])
            item["markdown"] = "" if text is None else str(text)
        corpus_records.append(item)

    query_records: List[Dict[str, Any]] = []
    for rec in queries_raw:
        _, qid = _first_present(rec, [query_id_column, "_id", "id", "query_id"])
        _, qtext = _first_present(rec, [query_column, "query", "text", "question"])
        if qid is None or qtext is None:
            continue
        query_records.append({query_id_column: str(qid), query_column: str(qtext)})

    qrels_records: List[Dict[str, Any]] = []
    with open(qrels_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        if not first_line:
            raise ValueError(f"qrels file is empty: {qrels_path}")
        f.seek(0)
        sniff = first_line.strip().split("\t")
        has_header = any(x in {"query-id", "corpus-id", "query_id", "corpus_id", "score"} for x in sniff)
        reader = csv.reader(f, delimiter="\t")
        if has_header:
            header = next(reader)
            col = {name: idx for idx, name in enumerate(header)}
            qid_idx = col.get("query-id", col.get(query_id_column, col.get("query_id")))
            cid_idx = col.get("corpus-id", col.get(corpus_id_column, col.get("corpus_id")))
            score_idx = col.get(score_column, col.get("score"))
            if qid_idx is None or cid_idx is None or score_idx is None:
                raise ValueError(f"qrels header is missing required columns: {header}")
            for row in reader:
                if not row:
                    continue
                qrels_records.append(
                    {
                        query_id_column: str(row[qid_idx]),
                        corpus_id_column: str(row[cid_idx]),
                        score_column: int(float(row[score_idx])),
                    }
                )
        else:
            for row in reader:
                if not row:
                    continue
                if len(row) >= 4:
                    qid, _, cid, score = row[0], row[1], row[2], row[3]
                elif len(row) == 3:
                    qid, cid, score = row[0], row[1], row[2]
                else:
                    raise ValueError(f"Cannot parse qrels row: {row}")
                qrels_records.append(
                    {
                        query_id_column: str(qid),
                        corpus_id_column: str(cid),
                        score_column: int(float(score)),
                    }
                )

    ds_corpus = Dataset.from_list(corpus_records)
    if use_visual_embedding:
        ds_corpus = ds_corpus.cast_column("image", Image())
    ds_queries = Dataset.from_list(query_records)
    ds_qrels = Dataset.from_list(qrels_records)

    return {"corpus": ds_corpus, "queries": ds_queries, "qrels": ds_qrels}


class ViDoReEvaluatorBEIR(BaseViDoReEvaluator):
    """
    Evaluator for the ViDoRe benchmark for datasets with a BEIR format, i.e. where each
    dataset contains 3 subsets:
        corpus: The dataset containing the corpus of documents.
        queries: The dataset containing the queries.
        qrels: The dataset containing the query relevance scores.

    **Important**: Do NOT use this evaluator for the ViDoRe (v1) leaderboard as the handling of duplicates
    slightly differs from the `ViDoReEvaluatorQA` evaluator.
    """

    def __init__(
        self,
        vision_retriever: BaseVisionRetriever,
        corpus_id_column: Optional[str] = None,
        query_id_column: Optional[str] = None,
        query_column: Optional[str] = None,
        passage_column: Optional[str] = None,
        score_column: Optional[str] = None,
    ):
        super().__init__(vision_retriever=vision_retriever)

        # Dataset column names
        self.corpus_id_column = corpus_id_column if corpus_id_column else "corpus_id"
        self.query_id_column = query_id_column if query_id_column else "query_id"
        self.query_column = query_column if query_column else "query"
        if passage_column:
            self.passage_column = passage_column
        else:
            # self.passage_column = "image" if self.vision_retriever.use_visual_embedding else "text_description"
            self.passage_column = "image" if self.vision_retriever.use_visual_embedding else "markdown"
        self.score_column = score_column if score_column else "score"

    def _save_retrieval_results(
        self,
        results: Dict[str, Dict[str, float]],
        ds_queries: Dataset,
        ds_qrels: Dataset,
        save_path: str,
        top_k: int,
    ):
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
        # Build lookups
        query_lookup = {str(item[self.query_id_column]): item[self.query_column] for item in ds_queries}
        
        qrels_lookup = defaultdict(list)
        for item in ds_qrels:
            qrels_lookup[str(item[self.query_id_column])].append(str(item[self.corpus_id_column]))
            
        with open(save_path, 'w', encoding='utf-8') as f:
            for query_id, scores_dict in results.items():
                query_text = query_lookup.get(query_id, "")
                gold_ids = qrels_lookup.get(query_id, [])
                
                sorted_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
                
                retrieved_items = []
                for rank, (corpus_id, score) in enumerate(sorted_scores, start=1):
                    retrieved_items.append({
                        "rank": rank,
                        "score": score,
                        "corpus_id": corpus_id
                    })
                
                output_item = {
                    "query_id": query_id,
                    "query": query_text,
                    "ground_truth": gold_ids,
                    "retrieved_items": retrieved_items
                }
                f.write(json.dumps(output_item, ensure_ascii=False) + "\n")

    def evaluate_dataset(
        self,
        ds: BEIRDataset,
        batch_query: int,
        batch_passage: int,
        batch_score: Optional[int] = None,
        dataloader_prebatch_query: Optional[int] = None,
        dataloader_prebatch_passage: Optional[int] = None,
        return_results: bool = False,
        save_results_path: Optional[str] = None,
        top_k: int = 10,
        **kwargs,
    ) -> Union[Dict[str, Optional[float]], Tuple[Dict[str, Optional[float]], Dict[str, Dict[str, float]]]]:
        """
        Evaluate the given BEIR dataset.

        Args:
            ds (BEIRDataset): The dataset to evaluate.
            batch_query (int): The batch size for processing queries.
            batch_passage (int): The batch size for processing passages.
            batch_score (Optional[int]): The batch size for computing similarity scores.
            dataloader_prebatch_query (Optional[int]): The number of queries to pre-batch before processing.
            dataloader_prebatch_passage (Optional[int]): The number of passages to pre-batch before processing.
            return_results (bool): Whether to return the retrieval results.
            save_results_path (Optional[str]): Path to save the retrieval results.
            top_k (int): Number of top results to save.
        """
        # Load datasets
        ds_corpus = ds["corpus"]
        ds_queries = ds["queries"]
        ds_qrels = ds["qrels"]

        # Cast IDs to string to ensure compatibility with MTEB
        passage_ids: List[str] = [str(elt) for elt in ds_corpus[self.corpus_id_column]]
        query_ids: List[str] = [str(elt) for elt in ds_queries[self.query_id_column]]

        qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
        for qrel in ds_qrels:
            query_id = str(qrel[self.query_id_column])
            corpus_id = str(qrel[self.corpus_id_column])
            qrels[query_id][corpus_id] = qrel[self.score_column]

        # Edge case: using the BM25Retriever
        if isinstance(self.vision_retriever, BM25Retriever):
            passages = ds_corpus[self.passage_column]
            queries: List[str] = ds_queries[self.query_column]

            scores = self.vision_retriever.get_scores_bm25(
                queries=queries,
                passages=passages,
            )
            results = self._get_retrieval_results(
                query_ids=query_ids,
                passage_ids=passage_ids,
                scores=scores,
            )
            metrics = self.compute_retrieval_scores(qrels=qrels, results=results)
            
            if save_results_path:
                 self._save_retrieval_results(
                     results=results,
                     ds_queries=ds_queries,
                     ds_qrels=ds_qrels,
                     save_path=save_results_path,
                     top_k=top_k
                 )

            if return_results:
                return metrics, results
            return metrics

        # Get the embeddings for the queries and passages
        query_embeddings = self._get_query_embeddings(
            ds=ds_queries,
            query_column=self.query_column,
            batch_query=batch_query,
            dataloader_prebatch_size=dataloader_prebatch_query,
        )
        passage_embeddings = self._get_passage_embeddings(
            ds=ds_corpus,
            passage_column=self.passage_column,
            batch_passage=batch_passage,
            dataloader_prebatch_size=dataloader_prebatch_passage,
        )

        # Get the similarity scores
        scores = self.vision_retriever.get_scores(
            query_embeddings=query_embeddings,
            passage_embeddings=passage_embeddings,
            batch_size=batch_score,
        )

        # Get the relevant passages and results
        results = self._get_retrieval_results(
            query_ids=query_ids,
            passage_ids=passage_ids,
            scores=scores,
        )

        # Compute the MTEB metrics
        metrics = self.compute_retrieval_scores(
            qrels=qrels,
            results=results,
            ignore_identical_ids=False,
        )

        if save_results_path:
             self._save_retrieval_results(
                 results=results,
                 ds_queries=ds_queries,
                 ds_qrels=ds_qrels,
                 save_path=save_results_path,
                 top_k=top_k
             )

        if return_results:
            return metrics, results

        return metrics

    def _get_retrieval_results(
        self,
        query_ids: List[str],
        passage_ids: List[str],
        scores: torch.Tensor,
    ) -> Dict[str, Dict[str, float]]:
        """
        Get the retrieval results from the model's scores, i.e. the retrieval scores for each passage for each query.

        Args:
            query_ids (List[str]): The list of query IDs.
            passage_ids (List[str]): The list of passage IDs.
            scores(torch.Tensor): The similarity scores between queries and passages (shape: n_queries, n_passages).

        Returns:
            (Dict[str, Dict[str, float]]): The retrieval results.

        Example output:
            ```python
            {
                "query_0": {"doc_i": 19.125, "doc_1": 18.75, ...},
                "query_1": {"doc_j": 17.25, "doc_1": 16.75, ...},
                ...
            }
            ```
        """
        results: Dict[str, Dict[str, float]] = {}

        for query_idx, query_id in enumerate(query_ids):
            for image_idx, score in enumerate(scores[query_idx]):
                image_id = passage_ids[image_idx]
                score_passage = float(score.item())

                if query_id in results:
                    current_score = results[query_id].get(image_id, 0)
                    results[query_id][image_id] = max(current_score, score_passage)
                else:
                    results[query_id] = {image_id: score_passage}

        return results
