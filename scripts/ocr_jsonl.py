import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from tqdm import tqdm


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _run_tesseract(pil_image: Image.Image, lang: str, psm: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        pil_image.save(tmp.name)
        cmd = ["tesseract", tmp.name, "stdout", "-l", lang, "--psm", str(psm)]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "tesseract failed")
        return _normalize_text(proc.stdout)


def _safe_crop(img: Image.Image, box: Dict[str, Any]) -> Image.Image:
    w, h = img.size
    x1 = float(box["x1"])
    y1 = float(box["y1"])
    x2 = float(box["x2"])
    y2 = float(box["y2"])
    x1 = max(0.0, min(x1, float(w)))
    x2 = max(0.0, min(x2, float(w)))
    y1 = max(0.0, min(y1, float(h)))
    y2 = max(0.0, min(y2, float(h)))
    if x2 <= x1 or y2 <= y1:
        return img.crop((0, 0, 1, 1))
    return img.crop((x1, y1, x2, y2))


def _get_image_path(image_dir: str, image_path: str) -> str:
    p = Path(image_path)
    if p.is_absolute():
        return str(p)
    return str(Path(image_dir) / image_path)


def _process_record(
    record: Dict[str, Any],
    image_dir: str,
    lang: str,
    psm_page: int,
    psm_bbox: int,
    overwrite: bool,
) -> Dict[str, Any]:
    image_path = record.get("image_path")
    if not image_path:
        return record

    full_image_path = _get_image_path(image_dir=image_dir, image_path=image_path)
    img = Image.open(full_image_path).convert("RGB")

    if overwrite or not record.get("text_description"):
        record["text_description"] = _run_tesseract(img, lang=lang, psm=psm_page)

    bboxes = record.get("bboxes") or []
    for bbox in bboxes:
        if not overwrite and bbox.get("text_description"):
            continue
        box = bbox.get("box")
        if not box:
            bbox["text_description"] = ""
            continue
        crop = _safe_crop(img, box)
        bbox["text_description"] = _run_tesseract(crop, lang=lang, psm=psm_bbox)

    record["bboxes"] = bboxes
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--lang", type=str, default="eng")
    parser.add_argument("--psm_page", type=int, default=6)
    parser.add_argument("--psm_bbox", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="OCR JSONL", unit="lines"):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                record = _process_record(
                    record=record,
                    image_dir=args.image_dir,
                    lang=args.lang,
                    psm_page=args.psm_page,
                    psm_bbox=args.psm_bbox,
                    overwrite=args.overwrite,
                )
            except Exception:
                record.setdefault("text_description", "")
                for bbox in record.get("bboxes") or []:
                    bbox.setdefault("text_description", "")
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
