# LFRAG: Layout-oriented Fine-grained Retrieval-Augmented Generation on Multimodal Document Understanding

This repository provides the complete training, inference, and evaluation pipeline of LFRAG, together with the LFDocQA benchmark for fine-grained document retrieval and question answering research.

## Environment Setup
```
cd LFRAG
conda create -n lfrag python=3.10.18 -y
conda activate lfrag
pip install -r requirements.txt
```

## Data Preparation
We provide detailed instructions for preparing training and evaluation datasets for LFRAG. All datasets follow a unified standard format for visual document retrieval and QA tasks.

### 1. Training Dataset
The annotation files for training are already provided in `datasets/train/`.

The corresponding document images should be downloaded separately from the provided cloud storage link: XXXXX. After downloading, extract the image archives and place them under `datasets/train/`.

The final training dataset structure should be organized as follows:
```
datasets/
├── train/
│   ├── BBox_DocVQA_train.jsonl
│   ├── docmatix_train.jsonl
│   ├── docvqa_train.jsonl
│   ├── dude_train.jsonl
│   ├── infovqa_train.jsonl
│   ├── BBox_DocVQA/
│   ├── docmatix/
│   ├── docvqa/
│   ├── dude/
│   └── infographicsvqa/         
```

### 2. Evaluation Dataset
We build a novel visual document benchmark **LFDocQA** for comprehensive evaluation, which consists of two independent subsets: **LF-Docmatix** and **LF-PaperTab**. Each subset contains 500 high-quality annotated samples for region-level retrieval and document QA evaluation.

The evaluation dataset is stored in the following structure:

```
datasets/
├── eval/
│   ├── LF_Docmatix.jsonl
│   ├── LF_PaperTab.jsonl
│   ├── LF_Docmatix/
│   ├── LF_PaperTab/
```

#### Sample Format

Each sample is stored as a JSON object with the following format:
```json
{
  "doc_id": "31093",
  "image_id": "0",
  "image_path": "31093_0.jpg",
  "bboxes": [
    {
      "bbox_id": 0,
      "class_name": "table",
      "box": {
        "x1": 103.06436,
        "y1": 109.12723,
        "x2": 1136.47498,
        "y2": 1453.95398
      },
      "confidence": 0.93078
    }
  ],
  "question": "What are the main events taking place in the vicinity of Belgium according to the ONZK calendar for 2021?",
  "answer": "The main events taking place in the vicinity of Belgium include ...",
  "relevant_bbox_ids": [0,3]
}
```

#### Field Description

- `doc_id`: Document-level identifier.
- `image_id`: Page or image identifier inside the document.
- `image_path`: Relative path of the page image used by the sample.
- `bboxes`: Detected layout regions on the image.
- `question`: Natural-language question about the document page.
- `answer`: Ground-truth answer for the question.
- `relevant_bbox_ids`: IDs of the regions that are relevant to answering the question.

#### Bounding boxes Structure

Each item in `bboxes` contains:

- `bbox_id`: Region identifier referenced by `relevant_bbox_ids`.
- `class_name`: Region type, such as `table`, `figure`, `title`, or `plain text`.
- `box`: Bounding-box coordinates in image space.
- `confidence`: Confidence score for the region.

The `box` object contains:

- `x1`, `y1`: Top-left corner.
- `x2`, `y2`: Bottom-right corner.

## Training
We provide a multi-GPU training pipeline for efficient LoRA fine-tuning of the LFRAG retriever model.

#### Multi-GPU Configuration

The multi-GPU training environment is configured in   `scripts/train_config.yaml`. You can modify this file to specify the available GPU IDs and the number of training processes according to your hardware device conditions.

#### Start Training
The LFRAG retriever is built upon [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), following the same initialization setting as [ColQwen2.5-v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2). Therefore, we start fine-tuning from the released ColQwen2.5 checkpoint.

First, download the pretrained checkpoint from Hugging Face:

```
huggingface-cli download --resume-download vidore/colqwen2.5-v0.2  --local-dir ./models/colqwen2.5-v0.2
```

The downloaded checkpoint is initialized from the base model `vidore/colqwen2.5-base` (i.e., Qwen2.5-VL-3B), which serves as the backbone of our retriever model.

Before launching training, please modify several parameters in `scripts/train.sh`, including:

- `PYTHONPATH`: path to the LFRAG project directory.
- `MODEL_PATH`: local path of the downloaded ColQwen2.5 checkpoint.
- `WANDB_PROJECT`, `WANDB_NAME`, `WANDB_ENTITY`, and `WANDB_API_KEY`: WandB logging configuration.
Training hyperparameters and dataset paths according to your environment.

After configuration, start training with:
```
bash scripts/train.sh
```

## Inference
#### Step 1: Layout Segmentation and Block Aggregation

First, document page images in the retrieval corpus are processed using [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) for layout segmentation. Model can be downloaded from [here](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench). After that, LFRAG performs Block Aggregation to merge semantically related layout regions into larger retrieval units.

We provide `scripts/layout_segmentation_and_merge_sample.py` to visualize layout segmentation and block aggregation results on a single image.

For batch processing all document images in the corpus and generating structured document block annotations in JSONL format, please use the batch preprocessing script: 
```bash
python scripts/layout_segmentation_and_merge.py \
  --model ./models/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt \
  --image_dir /your-corpus-dir \
  --output_dir ./vis_output \
  --jsonl_path ./your-corpus.jsonl
```

This step generates structured document block annotations and saves them in `.jsonl` format for retrieval.

#### Step 2: Merge Base Model Weights

Since the LFRAG LoRA adapter is trained on top of [ColQwen2.5-v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2), which is itself a LoRA adapter on [colqwen2.5-base](https://huggingface.co/vidore/colqwen2.5-base) (i.e. Qwen2.5-VL-3B), you need to merge ColQwen2.5-v0.2 into the base model first:

```bash
# Download the base model and ColQwen2.5-v0.2 adapter
huggingface-cli download --resume-download vidore/colqwen2.5-base --local-dir ./models/colqwen2.5-base
huggingface-cli download --resume-download vidore/colqwen2.5-v0.2 --local-dir ./models/colqwen2.5-v0.2

# Merge into a single model
python scripts/merge_adapter.py \
  --base_model ./models/colqwen2.5-base \
  --adapter ./models/colqwen2.5-v0.2 \
  --output_dir ./models/colqwen2.5-v0.2-merged
```

#### Step 3: Retrieval with LFRAG

The inference script uses a two-step workflow: **build index** (pre-compute passage embeddings once) and **query** (retrieve top-k blocks per query in real time).

**Build Index** (run once per corpus):
```bash
python scripts/inference.py build-index \
  --jsonl_path ./your-corpus.jsonl \
  --image_dir /your-corpus-dir \
  --index_path ./indexes/index.pt
```

**Query** (fast, only encodes the query):
```bash
python scripts/inference.py query \
  --query "user query" \
  --index_path ./indexes/index.pt \
  --topk 3
```

Optional arguments for `query`:
- `--output results.json`: Save results to a JSON file.
- `--save_crops_dir ./crops`: Save top-k cropped block images to a directory.
- `--image_dir /your-corpus-dir`: Required when using `--save_crops_dir`.

Our trained LoRA adapter checkpoint can be downloaded from [Google Drive](https://drive.google.com/file/d/1gqey-X0qKqdQcxvBV51_g37fpnKpYq1o/view?usp=drive_link). After downloading, extract the zip file and place the contents under `./ckpts/lfrag/`.

**Note:** The `lfrag/` directory is a PEFT LoRA adapter trained on top of the merged ColQwen2.5-v0.2 model.

- Base model: `vidore/colqwen2.5-base` (merged with `vidore/colqwen2.5-v0.2`)
- PEFT type: `LORA`
- Task type: `FEATURE_EXTRACTION`

Important: Do not load `lfrag/adapter_model.safetensors` directly.
Please load the entire `lfrag/` directory as a PEFT adapter.

## Evaluation

### Prerequisites

#### Retrieval Models

To reproduce all retrieval baselines, download the following models and place them under `./models/`:

| Model | Source | Notes |
|---|---|---|
| SigLIP | [google/siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384) | Vision-only retriever |
| ColPali | [vidore/colpali](https://huggingface.co/vidore/colpali) | Late-interaction retriever |
| ColQwen2.5 | [vidore/colqwen2.5-v0.2](https://huggingface.co/vidore/colqwen2.5-v0.2) | Late-interaction retriever |
| VisRAG-Ret | [openbmb/VisRAG-Ret](https://huggingface.co/openbmb/VisRAG-Ret) | Vision RAG retriever |
| BGE-M3 | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | Text-based retriever (requires OCR) |
| NV-Embed-v2 | [nvidia/NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2) | Text-based retriever (requires OCR) |
| LFRAG | `./ckpts/lfrag/` | Our method (LoRA adapter) |

For text-based retrievers (BM25, BGE-M3, NV-Embed-v2), you need to first generate OCR text descriptions. Use the provided script:
```bash
python scripts/ocr_jsonl.py --input ./datasets/eval/LF_Docmatix.jsonl --output ./datasets/eval/LF_Docmatix_ocr.jsonl
```

#### LLM Services for Generation and Judging

The generation evaluation pipeline requires two LLM services running via [vLLM](https://github.com/vllm-project/vllm) or [SGLang](https://github.com/sgl-project/sglang):

- **Generation model** (e.g., Qwen2.5-VL-7B): serves on port `8003` for answer generation.
- **Judge model** (e.g., Qwen3-14B): serves on port `8004` for LLM-based scoring.

Example launch commands:
```bash
# Generation model (VLM)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen2.5-VL-7B-Instruct --port 8003

# Judge model (text-only LLM)
CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
  --model-path ./models/qwen3-14b --port 8004
```

### Step 1: Retrieval Evaluation

Run all retrieval methods on both LFDocQA subsets:

```bash
cd scripts
bash retrieval.sh
```

This evaluates each retriever at both **page-level** and **bbox-level**, outputting metrics (NDCG, MAP, Recall, Precision, MRR) and saving retrieval results under `./results/`.

To evaluate a single retriever:
```bash
python scripts/eval.py \
  --retriever lfrag_retriever \
  --model_path ./ckpts/lfrag \
  --jsonl_path ./datasets/eval/LF_Docmatix.jsonl \
  --image_dir ./datasets/eval \
  --level both \
  --batch_query 8 --batch_passage 8 --batch_score 128 \
  --output ./outputs/LF_Docmatix/lfrag.json \
  --save_results_path ./results/LF_Docmatix/lfrag/retrieval_results.jsonl
```

### Step 2: Generation Evaluation

After retrieval, feed the retrieved results into a VLM for answer generation and evaluate with multiple metrics:

```bash
cd scripts
bash gen.sh
```

Or run a single generation evaluation:
```bash
python scripts/generate.py \
  --results_path ./results/LF_Docmatix/lfrag/retrieval_results_bbox.jsonl \
  --image_dir ./datasets/eval/LF_Docmatix \
  --output_path ./results/LF_Docmatix/lfrag/topk_3_bbox.jsonl \
  --topk 3 --level bbox \
  --gen_model_port 8003 --judge_model_port 8004 \
  --api_base http://127.0.0.1
```

### Step 3: LLM Judge (Optional)

To re-score existing generation results with a different judge model:
```bash
python scripts/llm_judge.py \
  --results_dir ./results \
  --datasets LF_Docmatix LF_PaperTab \
  --port 8004
```


## Acknowledgements

Our codebase is built upon and modified from the excellent open-source framework provided by [ColPali](https://github.com/illuin-tech/colpali).
We also use [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO) for document layout segmentation and preprocessing.