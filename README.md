# LFRAG: Layout-oriented Fine-grained Retrieval-Augmented Generation on Multimodal Document Understanding

This repository contains two JSONL datasets under `datasets/` and one LoRA adapter under `lfrag/`.

## Repository Layout

```text
datasets/
  docmatix.jsonl
  PaperTab.jsonl
lfrag/
  adapter_config.json
  adapter_model.safetensors
  ...
```

## Datasets

Both dataset files use the same JSONL schema. Each line is one training or evaluation sample.

### Files

- `datasets/docmatix.jsonl`: 500 samples
- `datasets/PaperTab.jsonl`: 500 samples

### Sample Format

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

### Field Description

- `doc_id`: Document-level identifier.
- `image_id`: Page or image identifier inside the document.
- `image_path`: Relative path of the page image used by the sample.
- `bboxes`: Detected layout regions on the image.
- `question`: Natural-language question about the document page.
- `answer`: Ground-truth answer for the question.
- `relevant_bbox_ids`: IDs of the regions that are relevant to answering the question.

### `bboxes` Structure

Each item in `bboxes` contains:

- `bbox_id`: Region identifier referenced by `relevant_bbox_ids`.
- `class_name`: Region type, such as `table`, `figure`, `title`, or `plain text`.
- `box`: Bounding-box coordinates in image space.
- `confidence`: Confidence score for the region.

The `box` object contains:

- `x1`, `y1`: Top-left corner.
- `x2`, `y2`: Bottom-right corner.

## Using the `lfrag` LoRA Adapter

The `lfrag/` folder is a PEFT LoRA adapter trained on top of:

- Base model: `vidore/colqwen2.5-base`
- PEFT type: `LORA`
- Task type: `FEATURE_EXTRACTION`

Important: do not load `lfrag/adapter_model.safetensors` by itself. Load the whole `lfrag/` directory as a PEFT adapter.

### Install Dependencies

```bash
pip install torch transformers peft
```

### Load the Base Model and Adapter

```python
import torch
from transformers import AutoModel, AutoProcessor
from peft import PeftModel

base_model_name = "vidore/colqwen2.5-base"
adapter_path = "lfrag"

processor = AutoProcessor.from_pretrained(
    adapter_path,
    trust_remote_code=True,
)

base_model = AutoModel.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()
```

