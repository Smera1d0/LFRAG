"""
LLM Evaluation Script: Read target jsonl under results, score (0-5) based on ground_truth and prediction.

Example model launch command:
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path ./models/qwen3-14b --port 8004

Default processing targets:
- results/LF_Docmatix/*/topk_3_page.jsonl
- results/LF_Docmatix/*/topk_5_page.jsonl
- results/LF_Docmatix/*/topk_3_bbox.jsonl
- results/LF_Docmatix/*/topk_5_bbox.jsonl
- results/LF_PaperTab/*/topk_3_page.jsonl
- results/LF_PaperTab/*/topk_5_page.jsonl
- results/LF_PaperTab/*/topk_3_bbox.jsonl
- results/LF_PaperTab/*/topk_5_bbox.jsonl
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


def _normalize_api_base(api_base: str) -> str:
    base = api_base.strip()
    if not base.startswith("http://") and not base.startswith("https://"):
        base = f"http://{base}"
    return base.rstrip("/")


def build_openai_client(api_key: Optional[str], api_base: str, port: int) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    base = _normalize_api_base(api_base)
    return OpenAI(api_key=key, base_url=f"{base}:{port}/v1")


def get_model_name(client: OpenAI) -> str:
    response = client.models.list()
    data = getattr(response, "data", None)
    if not data:
        raise RuntimeError("Model server returned empty, no available model found.")
    model_id = getattr(data[0], "id", None)
    if not model_id and isinstance(data[0], dict):
        model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError("Model server did not return a valid model ID.")
    return model_id


def build_lenient_judge_prompt(ground_truth: str, prediction: str) -> str:
    return f"""
You are a QA evaluator. Compare the [Reference Answer] and the [Model Answer], then assign an integer score from 0 to 5.
Scoring rubric:
- 5: Semantically correct and covers the core information; different wording/format and minor irrelevant additions are acceptable.
- 4: Mostly correct with only minor omissions or minor inaccuracies that do not affect the main conclusion.
- 3: Partially correct; includes some key information but has clear missing parts or local errors.
- 2: Only limited relevant content; most key facts are missing or confused.
- 1: Mostly incorrect; only weak relevance.
- 0: Completely incorrect, off-topic, contradictory to the reference answer, or no valid answer.
Additional rules:
1) For numbers, dates, and proper nouns: if the main conclusion is correct but formatting differs slightly (e.g., 09/01 vs September 1), be lenient.
2) For long answers: if the core conclusion is correct but contains a small amount of noise, do not over-penalize.
3) If the reference answer indicates "not mentioned/cannot be determined", then fabricated specific facts should receive a low score (0-2).

Return strict JSON only. Do not output anything else:
{{"score": <integer 0-5>, "reason": "one short English reason"}}

[Reference Answer]
{ground_truth}
[Model Answer]
{prediction}

"""


def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return "\n".join(str(i) for i in x)
    return str(x)


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_judge_response(text: str) -> Tuple[int, str]:
    s = text.strip()
    try:
        data = json.loads(s)
        score = int(data.get("score", 0))
        reason = str(data.get("reason", ""))
        score = max(0, min(5, score))
        return score, reason
    except Exception:
        pass

    score_match = re.search(r'"?score"?\s*[:：]\s*([0-5])', s, flags=re.IGNORECASE)
    reason_match = re.search(
        r'"?reason"?\s*[:：]\s*"?(.+?)"?\s*$',
        s,
        flags=re.IGNORECASE | re.DOTALL,
    )
    score = int(score_match.group(1)) if score_match else 0
    reason = reason_match.group(1).strip() if reason_match else "Failed to parse model output, default score 0."
    return score, reason


def judge_one(
    client: OpenAI,
    model_name: str,
    ground_truth: str,
    prediction: str,
    max_retries: int = 2,
) -> Tuple[int, str]:
    prompt = build_lenient_judge_prompt(ground_truth=ground_truth, prediction=prediction)

    last_err = ""
    for _ in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a strict evaluation assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content or ""
            score, reason = _parse_judge_response(content)
            if not reason:
                reason = "Failed to parse model output, default score 0."
            return score, reason
        except Exception as e:
            last_err = str(e)

    return 0, f"Failed to call model, default score 0: {last_err}"


def find_target_files(results_dir: Path, datasets: List[str], target_files: List[str]) -> List[Path]:
    paths: List[Path] = []
    for dataset in datasets:
        dataset_dir = results_dir / dataset
        if not dataset_dir.exists():
            continue
        for f in target_files:
            paths.extend(sorted(dataset_dir.glob(f"*/{f}")))
    return paths


def _should_update(metrics: Dict[str, Any], overwrite: bool) -> bool:
    if overwrite:
        return True
    return metrics.get("llm_reason") in (None, "", "disabled")


def _judge_row(
    item: Dict[str, Any],
    client: OpenAI,
    model_name: str,
) -> Tuple[float, str]:
    ground_truth = _safe_text(item.get("ground_truth"))
    prediction = _safe_text(item.get("prediction"))

    if not prediction.strip():
        return 0.0, "Model answer is empty."

    score, reason = judge_one(client, model_name, ground_truth, prediction)
    return float(score), reason


def _avg(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _load_existing_summary(summary_path: Path) -> Dict[str, Any]:
    if not summary_path.exists():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            line = f.readline().strip()
            return json.loads(line) if line else {}
    except Exception:
        return {}


def write_summary(jsonl_path: Path, rows: List[Dict[str, Any]], judge_model: str) -> Path:
    summary_path = jsonl_path.with_name(jsonl_path.stem + "_summary.jsonl")
    old_summary = _load_existing_summary(summary_path)

    word_f1_values: List[float] = []
    rouge_values: List[float] = []
    bert_values: List[float] = []
    llm_values: List[float] = []
    gen_time_values: List[float] = []
    prompt_token_values: List[float] = []
    image_token_values: List[float] = []

    gen_model: Optional[str] = None

    for item in rows:
        if gen_model is None:
            gm = item.get("gen_model")
            if isinstance(gm, str) and gm.strip():
                gen_model = gm

        metrics = item.get("metrics", {})
        if isinstance(metrics, dict):
            v = _to_float(metrics.get("word_f1"))
            if v is not None:
                word_f1_values.append(v)
            v = _to_float(metrics.get("rougeL"))
            if v is not None:
                rouge_values.append(v)
            v = _to_float(metrics.get("bert_score_f1"))
            if v is not None:
                bert_values.append(v)
            v = _to_float(metrics.get("llm_score"))
            if v is not None:
                llm_values.append(v)

        v = _to_float(item.get("generation_time_seconds"))
        if v is not None:
            gen_time_values.append(v)

        v = _to_float(item.get("prompt_tokens"))
        if v is not None:
            prompt_token_values.append(v)

        v = _to_float(item.get("image_tokens"))
        if v is not None:
            image_token_values.append(v)

    if gen_model is None:
        old_gen_model = old_summary.get("gen_model") if isinstance(old_summary, dict) else None
        gen_model = old_gen_model if isinstance(old_gen_model, str) and old_gen_model.strip() else None

    avg_prompt_tokens = _avg(prompt_token_values)
    if avg_prompt_tokens is None:
        avg_prompt_tokens = _to_float(old_summary.get("avg_prompt_tokens")) if isinstance(old_summary, dict) else None

    avg_image_tokens = _avg(image_token_values)
    if avg_image_tokens is None:
        avg_image_tokens = _to_float(old_summary.get("avg_image_tokens")) if isinstance(old_summary, dict) else 0.0

    summary = {
        "gen_model": gen_model,
        "judge_model": judge_model,
        "llm_judge_enabled": True,
        "avg_word_f1": _avg(word_f1_values),
        "avg_rougeL": _avg(rouge_values),
        "avg_bert_score": _avg(bert_values),
        "avg_llm_score": _avg(llm_values),
        "avg_image_tokens": avg_image_tokens,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_generation_time_seconds": _avg(gen_time_values),
        "count": len(rows),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    return summary_path


def process_file(
    jsonl_path: Path,
    client: OpenAI,
    model_name: str,
    overwrite: bool,
    max_samples: Optional[int],
    num_workers: int,
) -> Tuple[int, int, Path]:
    rows: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    total = len(rows)
    end_idx = min(total, max_samples) if max_samples is not None else total

    target_indices: List[int] = []
    for idx in range(end_idx):
        item = rows[idx]
        metrics = item.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {}
            item["metrics"] = metrics

        if _should_update(metrics, overwrite=overwrite):
            target_indices.append(idx)

    if target_indices:
        if num_workers <= 1:
            for idx in target_indices:
                score, reason = _judge_row(rows[idx], client, model_name)
                metrics = rows[idx]["metrics"]
                metrics["llm_score"] = score
                metrics["llm_reason"] = reason
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_idx = {
                    executor.submit(_judge_row, rows[idx], client, model_name): idx
                    for idx in target_indices
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        score, reason = future.result()
                    except Exception as e:
                        score, reason = 0.0, f"Concurrent scoring failed, default score 0: {e}"
                    metrics = rows[idx]["metrics"]
                    metrics["llm_score"] = score
                    metrics["llm_reason"] = reason

    tmp_path = jsonl_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    tmp_path.replace(jsonl_path)

    summary_path = write_summary(jsonl_path=jsonl_path, rows=rows, judge_model=model_name)
    return len(target_indices), total, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Use local 8004 port model to score topk results with 0-5 and write back to metrics field")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--datasets", nargs="+", default=["LF_Docmatix"])
    parser.add_argument(
        "--target_files",
        nargs="+",
        default=[
            "topk_3_page.jsonl",
            # "topk_5_page.jsonl",
            "topk_3_bbox.jsonl",
            # "topk_5_bbox.jsonl",
            # "topk_1_page.jsonl",
            # "topk_1_bbox.jsonl",
        ],
    )
    parser.add_argument("--api_base", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8004)
    parser.add_argument("--api_key", type=str, default="EMPTY")
    parser.add_argument("--model", type=str, default=None, help="Auto fetch first model from server if not provided")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing llm_reason (only overwrite disabled/empty by default)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per file for debugging only")
    parser.add_argument("--num_workers", type=int, default=4, help="Concurrent threads for LLM scoring (one request per thread)")
    args = parser.parse_args()

    if args.num_workers < 1:
        raise ValueError("--num_workers must be >= 1")

    client = build_openai_client(api_key=args.api_key, api_base=args.api_base, port=args.port)
    model_name = args.model or get_model_name(client)
    print(f"[INFO] Using model: {model_name}")
    print(f"[INFO] Concurrent threads: {args.num_workers}")

    input_results_dir = Path(args.results_dir)
    project_root = Path(__file__).resolve().parent.parent
    if input_results_dir.is_absolute():
        results_dir = input_results_dir
    else:
        cwd_candidate = Path.cwd() / input_results_dir
        root_candidate = project_root / input_results_dir
        if cwd_candidate.exists():
            results_dir = cwd_candidate
        else:
            results_dir = root_candidate

    print(f"[INFO] results_dir: {results_dir}")
    targets = find_target_files(results_dir, args.datasets, args.target_files)
    if not targets:
        raise RuntimeError(
            f"No target files found, check --results_dir / --datasets / --target_files. Current results_dir={results_dir}"
        )

    print(f"[INFO] Found {len(targets)} target files")
    grand_updated = 0
    grand_total = 0

    for p in targets:
        print(f"[INFO] Processing: {p}")
        updated, total, summary_path = process_file(
            jsonl_path=p,
            client=client,
            model_name=model_name,
            overwrite=args.overwrite,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
        )
        grand_updated += updated
        grand_total += total
        print(f"[INFO] Completed: {p} | Updated {updated}/{total}")
        print(f"[INFO] Summary written: {summary_path}")

    print(f"[DONE] All completed, total updated {grand_updated} items (total samples {grand_total})")


if __name__ == "__main__":
    main()
