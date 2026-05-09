#!/usr/bin/env bash
# ============================================================================
# retrieval.sh  —— Evaluate all retrieval methods on LF_Docmatix and LF_PaperTab
# Usage:  bash retrieval.sh
# ============================================================================
set -e

PYTHONPATH=./scripts/
export PYTHONPATH

DEVICE="cuda:0"
BATCH_QUERY=8
BATCH_PASSAGE=8

RESULTS_ROOT="./results"
OUTPUTS_ROOT="./outputs"

# ── Model paths ──────────────────────────────────────────────────────────────
SIGLIP_MODEL="./models/siglip-so400m-patch14-384"
COLPALI_MODEL="./models/colpali"
# COLQWEN_MODEL="./models/colqwen2_5_v0_1"
COLQWEN_MODEL="./models/colqwen2_5-v0_2"
VISRAG_MODEL="./models/VisRAG-Ret"
BGEM3_MODEL="./models/bge-m3"
NVEMBED_MODEL="./models/NV-Embed-v2"
LFRAG_MODEL="./models/lfrag"

# ── LF_Docmatix dataset ─────────────────────────────────────────────────────────
DOC_JSONL="./datasets/eval/LF_Docmatix.jsonl"
DOC_JSONL_OCR="./datasets/eval/LF_Docmatix_ocr.jsonl"
DOC_IMAGE_DIR="./datasets/eval/LF_Docmatix"

# ── LF_PaperTab dataset ─────────────────────────────────────────────────────────
PT_JSONL="./datasets/eval/LF_PaperTab.jsonl"
PT_JSONL_OCR="./datasets/eval/LF_PaperTab_ocr.jsonl"
PT_IMAGE_DIR="./datasets/eval/LF_PaperTab"


# ============================================================================
#                             LF_Docmatix
# ============================================================================

echo "====== [1/16] Siglip on LF_Docmatix ======"
python eval.py \
  --level both \
  --retriever siglip \
  --model_path "$SIGLIP_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$DOC_JSONL" \
  --image_dir "$DOC_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --output "$OUTPUTS_ROOT/LF_Docmatix/siglip.json" \
  --save_results_path "$RESULTS_ROOT/LF_Docmatix/siglip/retrieval_results.jsonl"

echo "====== [2/16] ColPali on LF_Docmatix ======"
python eval.py \
  --level both \
  --retriever colpali \
  --model_path "$COLPALI_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$DOC_JSONL" \
  --image_dir "$DOC_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --batch_score 128 \
  --output "$OUTPUTS_ROOT/LF_Docmatix/colpali.json" \
  --save_results_path "$RESULTS_ROOT/LF_Docmatix/colpali/retrieval_results.jsonl"

echo "====== [3/16] ColQwen2.5 on LF_Docmatix ======"
python eval.py \
  --level both \
  --retriever colqwen2_5 \
  --model_path "$COLQWEN_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$DOC_JSONL" \
  --image_dir "$DOC_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --output "$OUTPUTS_ROOT/LF_Docmatix/colqwen2_5.json" \
  --save_results_path "$RESULTS_ROOT/LF_Docmatix/colqwen/retrieval_results.jsonl"

echo "====== [4/16] VisRAG-Ret on LF_Docmatix ======"
python eval.py \
  --level both \
  --retriever visrag-ret \
  --model_path "$VISRAG_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$DOC_JSONL" \
  --image_dir "$DOC_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --output "$OUTPUTS_ROOT/LF_Docmatix/visrag.json" \
  --save_results_path "$RESULTS_ROOT/LF_Docmatix/VisRAG/retrieval_results.jsonl"

echo "====== [5/16] BM25 on LF_Docmatix (OCR) ======"
python eval.py \
  --level both \
  --retriever bm25 \
  --device "$DEVICE" \
  --jsonl_path "$DOC_JSONL_OCR" \
  --image_dir "$DOC_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --output "$OUTPUTS_ROOT/LF_Docmatix/bm25.json" \
  --save_results_path "$RESULTS_ROOT/LF_Docmatix/bm25/retrieval_results.jsonl"

echo "====== [6/16] BGE-M3 on LF_Docmatix (OCR) ======"
python eval.py \
  --level both \
  --retriever bge-m3 \
  --model_path "$BGEM3_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$DOC_JSONL_OCR" \
  --image_dir "$DOC_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --output "$OUTPUTS_ROOT/LF_Docmatix/bge-m3.json" \
  --save_results_path "$RESULTS_ROOT/LF_Docmatix/bge-m3/retrieval_results.jsonl"

echo "====== [7/16] NV-Embed-v2 on LF_Docmatix (OCR) ======"
python eval.py \
  --level both \
  --retriever nv-embed-v2 \
  --model_path "$NVEMBED_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$DOC_JSONL_OCR" \
  --image_dir "$DOC_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --output "$OUTPUTS_ROOT/LF_Docmatix/nv-embed-v2.json" \
  --save_results_path "$RESULTS_ROOT/LF_Docmatix/nv-embed-v2/retrieval_results.jsonl"

echo "====== [8/16] LFRAG Retriever on LF_Docmatix ======"
python eval.py \
  --level both \
  --retriever lfrag_retriever \
  --model_path "$LFRAG_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$DOC_JSONL" \
  --image_dir "$DOC_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --batch_score 128 \
  --output "$OUTPUTS_ROOT/LF_Docmatix/lfrag.json" \
  --save_results_path "$RESULTS_ROOT/LF_Docmatix/lfrag/retrieval_results.jsonl"

# ============================================================================
#                             LF_PaperTab
# ============================================================================

echo "====== [9/16] Siglip on LF_PaperTab ======"
python eval.py \
  --level both \
  --retriever siglip \
  --model_path "$SIGLIP_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$PT_JSONL" \
  --image_dir "$PT_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --output "$OUTPUTS_ROOT/LF_PaperTab/siglip.json" \
  --save_results_path "$RESULTS_ROOT/LF_PaperTab/siglip/retrieval_results.jsonl"

echo "====== [10/16] ColPali on LF_PaperTab ======"
python eval.py \
  --level both \
  --retriever colpali \
  --model_path "$COLPALI_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$PT_JSONL" \
  --image_dir "$PT_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --batch_score 128 \
  --output "$OUTPUTS_ROOT/LF_PaperTab/colpali.json" \
  --save_results_path "$RESULTS_ROOT/LF_PaperTab/colpali/retrieval_results.jsonl"

echo "====== [11/16] ColQwen2.5 on LF_PaperTab ======"
python eval.py \
  --level both \
  --retriever colqwen2_5 \
  --model_path "$COLQWEN_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$PT_JSONL" \
  --image_dir "$PT_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --output "$OUTPUTS_ROOT/LF_PaperTab/colqwen2_5.json" \
  --save_results_path "$RESULTS_ROOT/LF_PaperTab/colqwen/retrieval_results.jsonl"

echo "====== [12/16] VisRAG-Ret on LF_PaperTab ======"
python eval.py \
  --level both \
  --retriever visrag-ret \
  --model_path "$VISRAG_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$PT_JSONL" \
  --image_dir "$PT_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --output "$OUTPUTS_ROOT/LF_PaperTab/visrag.json" \
  --save_results_path "$RESULTS_ROOT/LF_PaperTab/visrag/retrieval_results.jsonl"

echo "====== [13/16] BM25 on LF_PaperTab (OCR) ======"
python eval.py \
  --level both \
  --retriever bm25 \
  --device "$DEVICE" \
  --jsonl_path "$PT_JSONL_OCR" \
  --image_dir "$PT_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --output "$OUTPUTS_ROOT/LF_PaperTab/bm25.json" \
  --save_results_path "$RESULTS_ROOT/LF_PaperTab/bm25/retrieval_results.jsonl"

echo "====== [14/16] BGE-M3 on LF_PaperTab (OCR) ======"
python eval.py \
  --level both \
  --retriever bge-m3 \
  --model_path "$BGEM3_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$PT_JSONL_OCR" \
  --image_dir "$PT_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --output "$OUTPUTS_ROOT/LF_PaperTab/bge-m3.json" \
  --save_results_path "$RESULTS_ROOT/LF_PaperTab/bge-m3/retrieval_results.jsonl"

echo "====== [15/16] NV-Embed-v2 on LF_PaperTab (OCR) ======"
python eval.py \
  --level both \
  --retriever nv-embed-v2 \
  --model_path "$NVEMBED_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$PT_JSONL_OCR" \
  --image_dir "$PT_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --output "$OUTPUTS_ROOT/LF_PaperTab/nv-embed-v2.json" \
  --save_results_path "$RESULTS_ROOT/LF_PaperTab/nv-embed-v2/retrieval_results.jsonl"

echo "====== [16/16] LFRAG Retriever on LF_PaperTab ======"
python eval.py \
  --level both \
  --retriever lfrag_retriever \
  --model_path "$LFRAG_MODEL" \
  --device "$DEVICE" \
  --jsonl_path "$PT_JSONL" \
  --image_dir "$PT_IMAGE_DIR" \
  --batch_query $BATCH_QUERY \
  --batch_passage $BATCH_PASSAGE \
  --batch_score 128 \
  --output "$OUTPUTS_ROOT/LF_PaperTab/lfrag.json" \
  --save_results_path "$RESULTS_ROOT/LF_PaperTab/lfrag/retrieval_results.jsonl"

echo "====== All retrieval experiments done! ======"
