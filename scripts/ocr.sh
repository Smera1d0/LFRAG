set -e

python scripts/ocr_jsonl.py \
  --input_jsonl datasets/eval/LF_Docmatix.jsonl \
  --output_jsonl datasets/eval/LF_Docmatix_ocr.jsonl \
  --image_dir datasets/eval/LF_Docmatix \
  --lang eng

python scripts/ocr_jsonl.py \
  --input_jsonl datasets/eval/LF_PaperTab.jsonl \
  --output_jsonl datasets/eval/LF_PaperTab_ocr.jsonl \
  --image_dir datasets/eval/LF_PaperTab \
  --lang eng