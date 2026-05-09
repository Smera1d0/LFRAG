#!/usr/bin/env bash
'''
gen.sh —— Run generation and evaluation sequentially (Docmatix + PaperTab)
Rule: Only LFRAG uses bbox-level results; all other methods uniformly use page-level results
Usage: bash scripts/gen.sh
'''

set -e

PYTHONPATH=./scripts/
export PYTHONPATH

TOPK_VALUES=(1 3 5 7)
NUM_WORKERS=32
GEN_MODEL_PORT=8003
JUDGE_MODEL_PORT=8004
API_BASE="http://127.0.0.1"

run_generate() {
  local dataset="$1"
  local method="$2"
  local level="$3"
  local image_dir="$4"
  local input_path="$5"
  local output_path="$6"

  echo "============================================================"
  echo "Dataset     : ${dataset}"
  echo "Method      : ${method}"
  echo "Level       : ${level}"
  echo "results_path: ${input_path}"
  echo "output_path : ${output_path}"
  echo "============================================================"

  python generate.py \
    --results_path "$input_path" \
    --image_dir "$image_dir" \
    --output_path "$output_path" \
    --topk "$TOPK" \
    --level "$level" \
    --num_workers "$NUM_WORKERS" \
    --gen_model_port "$GEN_MODEL_PORT" \
    --judge_model_port "$JUDGE_MODEL_PORT" \
    --api_base "$API_BASE" \
    --disable_bert_score \
    --debug_usage
}

DOC_IMAGE_DIR="./datasets/eval/LF_Docmatix"
DOC_RESULTS_ROOT="./results/LF_Docmatix"

PT_IMAGE_DIR="./datasets/eval/LF_PaperTab"
PT_RESULTS_ROOT="./results/LF_PaperTab"

run_all_for_current_topk() {
  # ============================================================================
  #                             Docmatix
  # ============================================================================

  echo "====== [1/16] Docmatix | siglip (page) ======"
  run_generate \
    "Docmatix" \
    "siglip" \
    "page" \
    "$DOC_IMAGE_DIR" \
    "$DOC_RESULTS_ROOT/siglip/retrieval_results_page.jsonl" \
    "$DOC_RESULTS_ROOT/siglip/topk_${TOPK}_page.jsonl"

  echo "====== [2/16] Docmatix | colpali (page) ======"
  run_generate \
    "Docmatix" \
    "colpali" \
    "page" \
    "$DOC_IMAGE_DIR" \
    "$DOC_RESULTS_ROOT/colpali/retrieval_results_page.jsonl" \
    "$DOC_RESULTS_ROOT/colpali/topk_${TOPK}_page.jsonl"

  echo "====== [3/16] Docmatix | colqwen (page) ======"
  run_generate \
    "Docmatix" \
    "colqwen" \
    "page" \
    "$DOC_IMAGE_DIR" \
    "$DOC_RESULTS_ROOT/colqwen/retrieval_results_page.jsonl" \
    "$DOC_RESULTS_ROOT/colqwen/topk_${TOPK}_page.jsonl"

  echo "====== [4/16] Docmatix | VisRAG (page) ======"
  run_generate \
    "Docmatix" \
    "VisRAG" \
    "page" \
    "$DOC_IMAGE_DIR" \
    "$DOC_RESULTS_ROOT/visrag/retrieval_results_page.jsonl" \
    "$DOC_RESULTS_ROOT/visrag/topk_${TOPK}_page.jsonl"

  echo "====== [5/16] Docmatix | bm25 (page) ======"
  run_generate \
    "Docmatix" \
    "bm25" \
    "page" \
    "$DOC_IMAGE_DIR" \
    "$DOC_RESULTS_ROOT/bm25/retrieval_results_page.jsonl" \
    "$DOC_RESULTS_ROOT/bm25/topk_${TOPK}_page.jsonl"

  echo "====== [6/16] Docmatix | bge-m3 (page) ======"
  run_generate \
    "Docmatix" \
    "bge-m3" \
    "page" \
    "$DOC_IMAGE_DIR" \
    "$DOC_RESULTS_ROOT/bge-m3/retrieval_results_page.jsonl" \
    "$DOC_RESULTS_ROOT/bge-m3/topk_${TOPK}_page.jsonl"

  echo "====== [7/16] Docmatix | nv-embed-v2 (page) ======"
  run_generate \
    "Docmatix" \
    "nv-embed-v2" \
    "page" \
    "$DOC_IMAGE_DIR" \
    "$DOC_RESULTS_ROOT/nv-embed-v2/retrieval_results_page.jsonl" \
    "$DOC_RESULTS_ROOT/nv-embed-v2/topk_${TOPK}_page.jsonl"

  echo "====== [8/16] Docmatix | LFRAG (bbox) ======"
  run_generate \
    "Docmatix" \
    "LFRAG" \
    "bbox" \
    "$DOC_IMAGE_DIR" \
    "$DOC_RESULTS_ROOT/lfrag/retrieval_results_bbox.jsonl" \
    "$DOC_RESULTS_ROOT/lfrag/topk_${TOPK}_bbox.jsonl"

  # ============================================================================
  #                             PaperTab
  # ============================================================================

  echo "====== [9/16] PaperTab | siglip (page) ======"
  run_generate \
    "PaperTab" \
    "siglip" \
    "page" \
    "$PT_IMAGE_DIR" \
    "$PT_RESULTS_ROOT/siglip/retrieval_results_page.jsonl" \
    "$PT_RESULTS_ROOT/siglip/topk_${TOPK}_page.jsonl"

  echo "====== [10/16] PaperTab | colpali (page) ======"
  run_generate \
    "PaperTab" \
    "colpali" \
    "page" \
    "$PT_IMAGE_DIR" \
    "$PT_RESULTS_ROOT/colpali/retrieval_results_page.jsonl" \
    "$PT_RESULTS_ROOT/colpali/topk_${TOPK}_page.jsonl"

  echo "====== [11/16] PaperTab | colqwen (page) ======"
  run_generate \
    "PaperTab" \
    "colqwen" \
    "page" \
    "$PT_IMAGE_DIR" \
    "$PT_RESULTS_ROOT/colqwen/retrieval_results_page.jsonl" \
    "$PT_RESULTS_ROOT/colqwen/topk_${TOPK}_page.jsonl"

  echo "====== [12/16] PaperTab | visrag (page) ======"
  run_generate \
    "PaperTab" \
    "visrag" \
    "page" \
    "$PT_IMAGE_DIR" \
    "$PT_RESULTS_ROOT/visrag/retrieval_results_page.jsonl" \
    "$PT_RESULTS_ROOT/visrag/topk_${TOPK}_page.jsonl"

  echo "====== [13/16] PaperTab | bm25 (page) ======"
  run_generate \
    "PaperTab" \
    "bm25" \
    "page" \
    "$PT_IMAGE_DIR" \
    "$PT_RESULTS_ROOT/bm25/retrieval_results_page.jsonl" \
    "$PT_RESULTS_ROOT/bm25/topk_${TOPK}_page.jsonl"

  echo "====== [14/16] PaperTab | bge-m3 (page) ======"
  run_generate \
    "PaperTab" \
    "bge-m3" \
    "page" \
    "$PT_IMAGE_DIR" \
    "$PT_RESULTS_ROOT/bge-m3/retrieval_results_page.jsonl" \
    "$PT_RESULTS_ROOT/bge-m3/topk_${TOPK}_page.jsonl"

  echo "====== [15/16] PaperTab | nv-embed-v2 (page) ======"
  run_generate \
    "PaperTab" \
    "nv-embed-v2" \
    "page" \
    "$PT_IMAGE_DIR" \
    "$PT_RESULTS_ROOT/nv-embed-v2/retrieval_results_page.jsonl" \
    "$PT_RESULTS_ROOT/nv-embed-v2/topk_${TOPK}_page.jsonl"

  echo "====== [16/16] PaperTab | LFRAG (bbox) ======"
  run_generate \
    "PaperTab" \
    "LFRAG" \
    "bbox" \
    "$PT_IMAGE_DIR" \
    "$PT_RESULTS_ROOT/lfrag/retrieval_results_bbox.jsonl" \
    "$PT_RESULTS_ROOT/lfrag/topk_${TOPK}_bbox.jsonl"
}

for TOPK in "${TOPK_VALUES[@]}"; do
  echo ""
  echo "############################################################"
  echo "### Running generation evaluations with TOPK=${TOPK}"
  echo "############################################################"
  run_all_for_current_topk
done

echo "====== All generation evaluations finished for TOPK values: ${TOPK_VALUES[*]} ======"
