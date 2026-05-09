import argparse
import base64
import json
import logging
import os
import re
import string
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from PIL import Image, ImageFile
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True
# Limit BERTScore in-flight requests; configurable via --bert_score_workers
BERT_SCORE_SEMAPHORE: Optional[threading.BoundedSemaphore] = None
USAGE_DEBUG_LOCK = threading.Lock()
USAGE_DEBUG_PRINTED = False


def _normalize_api_base(api_base: str) -> str:
    base = api_base.strip()
    if not base.startswith("http://") and not base.startswith("https://"):
        base = f"http://{base}"
    return base.rstrip("/")


def build_openai_client(api_key: Optional[str], api_base: Optional[str], port: int) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    base = _normalize_api_base(api_base or "http://127.0.0.1")
    base_url = f"{base}:{port}/v1"
    return OpenAI(api_key=key, base_url=base_url)


def get_model_name(client: OpenAI) -> str:
    try:
        response = client.models.list()
    except Exception as e:
        base_url = getattr(client, "base_url", None)
        raise RuntimeError(f"Failed to connect to model server at {base_url}: {e}") from e

    data = getattr(response, "data", None)
    if not data:
        raise RuntimeError("No model found from vLLM server")
    model_id = getattr(data[0], "id", None)
    if not model_id and isinstance(data[0], dict):
        model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError("Model id is missing in vLLM response")
    return model_id


def _image_to_data_url(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def _extract_image_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    usage_dict = None
    if isinstance(usage, dict):
        usage_dict = usage
    else:
        if hasattr(usage, "model_dump"):
            try:
                usage_dict = usage.model_dump()
            except Exception:
                usage_dict = None
    if usage_dict:
        direct_value = usage_dict.get("image_tokens")
        if direct_value is not None:
            try:
                return int(direct_value)
            except Exception:
                pass
    details = None
    if usage_dict:
        details = usage_dict.get("prompt_tokens_details")
    else:
        details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    value = None
    if isinstance(details, dict):
        value = details.get("image_tokens")
        if value is None:
            value = details.get("images")
    else:
        value = getattr(details, "image_tokens", None)
    if value is None:
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def _extract_prompt_tokens(usage: Any) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        value = usage.get("prompt_tokens")
        if value is not None:
            try:
                return int(value)
            except Exception:
                return 0
    if hasattr(usage, "prompt_tokens"):
        try:
            return int(getattr(usage, "prompt_tokens"))
        except Exception:
            return 0
    if hasattr(usage, "model_dump"):
        try:
            usage_dict = usage.model_dump()
            value = usage_dict.get("prompt_tokens")
            if value is not None:
                return int(value)
        except Exception:
            return 0
    return 0


def _format_usage(usage: Any) -> str:
    if usage is None:
        return "None"
    if isinstance(usage, dict):
        return json.dumps(usage, ensure_ascii=False)
    if hasattr(usage, "model_dump"):
        try:
            return json.dumps(usage.model_dump(), ensure_ascii=False)
        except Exception:
            return repr(usage)
    return repr(usage)


def _load_image(full_path: str) -> Optional[Image.Image]:
    exists = os.path.exists(full_path)
    size = None
    if exists:
        try:
            size = os.path.getsize(full_path)
        except Exception:
            size = None
    try:
        with Image.open(full_path) as im:
            im.load()
            return im.convert("RGB")
    except Exception as e:
        logger.warning(
            f"Failed to load image {full_path} (exists={exists}, size={size}): {type(e).__name__}: {e!r}"
        )
        return None


def generate_answer_vlm(
    client: OpenAI,
    model: str,
    query: str,
    images: List[Image.Image],
    debug_usage: bool,
) -> Dict[str, Any]:
    content = []
    for img in images:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(img)}})
    content.append(
        {
            "type": "text",
            "text": (
                f"You are given {len(images)} image snippet(s) retrieved from a document.\n"
                f"Question: {query}\n"
                "Answer based only on the images."
            ),
        }
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=1024,
        temperature=0.0,
    )
    usage = getattr(response, "usage", None)
    if debug_usage:
        with USAGE_DEBUG_LOCK:
            logger.info(f"usage={_format_usage(usage)}")
    return {
        "text": response.choices[0].message.content.strip(),
        "image_tokens": _extract_image_tokens(usage),
        "prompt_tokens": _extract_prompt_tokens(usage),
    }


# --- Evaluation Metrics ---

def normalize_answer(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def word_tokenize(text: str) -> List[str]:
    return normalize_answer(text).split()


def _calculate_token_f1(prediction_tokens: List[str], ground_truth_tokens: List[str]) -> float:
    prediction_counter = Counter(prediction_tokens)
    ground_truth_counter = Counter(ground_truth_tokens)
    true_positives = sum((prediction_counter & ground_truth_counter).values())
    false_positives = sum(prediction_counter.values()) - true_positives
    false_negatives = sum(ground_truth_counter.values()) - true_positives
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_word_f1(prediction: str, ground_truth: List[str]) -> Dict[str, float]:
    """Calculates word-level F1 (taking max over multiple ground truths)."""
    if not prediction or not ground_truth:
        return {"word_f1": 0.0}

    prediction_tokens = word_tokenize(prediction)
    best_score = 0.0
    for gt in ground_truth:
        score = _calculate_token_f1(prediction_tokens, word_tokenize(gt))
        if score > best_score:
            best_score = score
    return {"word_f1": best_score}


def calculate_rouge(prediction: str, ground_truth: List[str]) -> Dict[str, float]:
    """Calculates ROUGE-L scores (taking max over multiple ground truths)."""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    best_score = 0.0
    for gt in ground_truth:
        scores = scorer.score(gt, prediction)
        if scores['rougeL'].fmeasure > best_score:
            best_score = scores['rougeL'].fmeasure
    return {"rougeL": best_score}


def calculate_bert_score(prediction: str, ground_truth: List[str], device: str = "cuda") -> Dict[str, float]:
    """Calculates BERTScore F1 (taking max over multiple ground truths)."""
    if not prediction or not ground_truth:
        return {"bert_score_f1": 0.0}

    semaphore = BERT_SCORE_SEMAPHORE
    try:
        if semaphore is not None:
            semaphore.acquire()
        best_f1 = 0.0
        for gt in ground_truth:
            P, R, F1 = bert_score([prediction], [gt], lang="en", verbose=False, device=device, rescale_with_baseline=False)
            f1_val = F1.mean().item()
            if f1_val > best_f1:
                best_f1 = f1_val
        return {"bert_score_f1": best_f1}
    except Exception as e:
        logger.warning(f"BERTScore failed, return 0.0: {e}")
        return {"bert_score_f1": 0.0}
    finally:
        if semaphore is not None:
            semaphore.release()


def build_lenient_judge_prompt(ground_truth: str, prediction: str) -> str:
    return f"""
You are a "lenient but consistent" QA evaluator. Compare the [Reference Answer] and the [Model Answer], then assign an integer score from 0 to 5.

Scoring rubric (lenient):
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
""".strip()

def _safe_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, list):
        return "\n".join(str(i) for i in x)
    return str(x)


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
    reason = reason_match.group(1).strip() if reason_match else "Model output parsing failed, default score is 0"
    return score, reason


def _judge_one_with_retry(
    client: OpenAI,
    model: str,
    ground_truth: str,
    prediction: str,
    max_retries: int = 2,
) -> Tuple[int, str]:
    prompt = build_lenient_judge_prompt(ground_truth=ground_truth, prediction=prediction)

    last_err = ""
    for _ in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a rigorous evaluation assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
            )
            content = resp.choices[0].message.content or ""
            score, reason = _parse_judge_response(content)
            if not reason:
                reason = "Rated according to lenient standards."
            return score, reason
        except Exception as e:
            last_err = str(e)

    return 0, f"Call failed, default score is 0: {last_err}"


def llm_judge_score(client: OpenAI, question: str, ground_truth: List[str], prediction: str, model: str) -> Dict[str, Any]:
    """
    Evaluates the prediction using an LLM Judge (0-5 scale).
    """
    _ = question
    ground_truth_text = _safe_text(ground_truth)
    prediction_text = _safe_text(prediction)

    if not prediction_text.strip():
        return {"llm_score": 0.0, "llm_reason": "Model answer is empty."}

    try:
        score, reason = _judge_one_with_retry(
            client=client,
            model=model,
            ground_truth=ground_truth_text,
            prediction=prediction_text,
            max_retries=2,
        )
        return {"llm_score": float(score), "llm_reason": reason}
    except Exception as e:
        logger.error(f"LLM Judge Error: {e}")
        return {"llm_score": 0.0, "llm_reason": f"Error: {str(e)}"}


# --- Main Pipeline ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True, help="Path to the retrieval results JSONL")
    parser.add_argument("--image_dir", type=str, required=True, help="Root directory for images")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generation & eval results")
    parser.add_argument("--gen_model_port", type=int, required=True, help="VLLM generation port")
    parser.add_argument("--judge_model_port", type=int, required=True, help="VLLM judge port")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API Key for Judge")
    parser.add_argument("--api_base", type=str, required=True, help="Server IP or base host")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--topk", type=int, default=10, help="Top-k images used for generation")
    parser.add_argument("--level", type=str, default="bbox", choices=["bbox", "page"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug_usage", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--disable_llm_judge", action="store_true", help="Disable LLM judge scoring")
    parser.add_argument("--disable_bert_score", action="store_true", help="Disable BERTScore calculation")
    parser.add_argument("--bert_score_workers", type=int, default=1, help="Max concurrent BERTScore computations")
    args = parser.parse_args()

    if args.bert_score_workers < 1:
        raise ValueError("--bert_score_workers must be >= 1")

    global BERT_SCORE_SEMAPHORE
    BERT_SCORE_SEMAPHORE = threading.BoundedSemaphore(args.bert_score_workers)
    gen_client = build_openai_client(args.api_key, args.api_base, args.gen_model_port)
    gen_model = get_model_name(gen_client)
    judge_client = None if args.disable_llm_judge else build_openai_client(args.api_key, args.api_base, args.judge_model_port)
    judge_model = None if args.disable_llm_judge else get_model_name(judge_client)

    # Process Data
    results_data = []
    with open(args.results_path, 'r') as f:
        for line in f:
            if line.strip():
                results_data.append(json.loads(line))

    if args.max_samples:
        results_data = results_data[:args.max_samples]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # Metrics Accumulators
    total_word_f1 = 0.0
    total_rouge = 0.0
    total_bert = 0.0
    total_llm = 0.0
    total_image_tokens = 0.0
    total_prompt_tokens = 0.0
    total_generation_time = 0.0
    count = 0

    def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
        query = item["query"]
        ground_truth = item["ground_truth"]
        retrieved = item["retrieved_items"][:args.topk]

        images = []
        for ret_item in retrieved:
            img_path = ret_item.get("image_path")
            if not img_path:
                continue
            full_path = os.path.join(args.image_dir, img_path)
            img = _load_image(full_path)
            if img is None:
                continue
            if args.level == "bbox":
                box = ret_item.get("box")
                if not box:
                    continue
                coords = None
                if isinstance(box, dict):
                    coords = (box.get("x1"), box.get("y1"), box.get("x2"), box.get("y2"))
                elif isinstance(box, (list, tuple)) and len(box) >= 4:
                    coords = (box[0], box[1], box[2], box[3])
                if not coords or any(v is None for v in coords):
                    continue
                img = img.crop(coords)
            images.append(img)

        if not images:
            logger.warning(f"No images found for query: {query}")
            prediction = "I cannot answer this question as no relevant images were found."
            image_tokens = 0
            prompt_tokens = 0
            generation_time_seconds = 0.0
        else:
            try:
                start_time = time.perf_counter()
                generation = generate_answer_vlm(gen_client, gen_model, query, images, args.debug_usage)
                generation_time_seconds = time.perf_counter() - start_time
                prediction = generation["text"]
                image_tokens = generation.get("image_tokens", 0)
                prompt_tokens = generation.get("prompt_tokens", 0)
            except Exception as e:
                logger.error(f"Generation failed for query {query}: {e}")
                prediction = ""
                image_tokens = 0
                prompt_tokens = 0
                generation_time_seconds = 0.0

        metrics = {}
        word_f1_res = calculate_word_f1(prediction, ground_truth)
        metrics.update(word_f1_res)
        rouge_res = calculate_rouge(prediction, ground_truth)
        metrics.update(rouge_res)
        bert_res = calculate_bert_score(prediction, ground_truth, device=args.device) if not args.disable_bert_score else {"bert_score_f1": 0.0}
        metrics.update(bert_res)
        if args.disable_llm_judge:
            judge_res = {"llm_score": 0.0, "llm_reason": "disabled"}
        else:
            judge_res = llm_judge_score(judge_client, query, ground_truth, prediction, model=judge_model)
        metrics.update(judge_res)

        record = {
            "query_id": item["query_id"],
            "query": query,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "generation_time_seconds": generation_time_seconds,
            "metrics": metrics
        }
        return {
            "record": record,
            "word_f1": word_f1_res["word_f1"],
            "rouge": rouge_res["rougeL"],
            "bert": bert_res["bert_score_f1"],
            "llm": judge_res["llm_score"],
            "image_tokens": image_tokens,
            "prompt_tokens": prompt_tokens,
            "generation_time_seconds": generation_time_seconds,
        }

    write_lock = threading.Lock()
    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_item, item) for item in results_data]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating & Evaluating"):
                result = future.result()
                with write_lock:
                    f_out.write(json.dumps(result["record"], ensure_ascii=False) + "\n")
                    f_out.flush()
                    total_word_f1 += result["word_f1"]
                    total_rouge += result["rouge"]
                    total_bert += result["bert"]
                    total_llm += result["llm"]
                    total_image_tokens += result["image_tokens"]
                    total_prompt_tokens += result["prompt_tokens"]
                    total_generation_time += result["generation_time_seconds"]
                    count += 1

    # Final Summary
    if count > 0:
        summary = {
            "gen_model": gen_model,
            "judge_model": judge_model,
            "llm_judge_enabled": not args.disable_llm_judge,
            "avg_word_f1": total_word_f1 / count,
            "avg_rougeL": total_rouge / count,
            "avg_bert_score": None if args.disable_bert_score else total_bert / count,
            "avg_llm_score": None if args.disable_llm_judge else total_llm / count,
            "avg_image_tokens": total_image_tokens / count,
            "avg_prompt_tokens": total_prompt_tokens / count,
            "avg_generation_time_seconds": total_generation_time / count,
            "count": count
        }
        print("\n" + "=" * 30)
        print("EVALUATION SUMMARY")
        print("=" * 30)
        print(json.dumps(summary, indent=2))

        # Save summary
        summary_path = args.output_path.replace(".jsonl", "_summary.jsonl")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
