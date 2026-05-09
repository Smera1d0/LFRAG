from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union, cast

import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils.import_utils import is_flash_attn_2_available

from retrievers.base_vision_retriever import BaseVisionRetriever
from retrievers.registry_utils import register_vision_retriever
from utils.data_utils import ListDataset
from utils.torch_utils import get_torch_device

logger = logging.getLogger(__name__)

load_dotenv(override=True)


def _default_tag() -> str:
    return "<abandon>"


@register_vision_retriever("lfrag_retriever")
class LFRAGRetriever(BaseVisionRetriever):
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        device: str = "auto",
        num_workers: int = 0,
        base_model_name_or_path: str = "./models/colqwen2.5-v0.2-merged",
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)

        try:
            from colpali_engine.models import LFRAG_Processor, LFRAG_Retriever
        except ImportError as e:
            raise ImportError(
                "Cannot import colpali_engine. Please ensure it is installed and accessible. "
            ) from e

        self.device = get_torch_device(device)
        self.num_workers = num_workers
        self.base_model_name_or_path = base_model_name_or_path
        self.lora_name_or_path = pretrained_model_name_or_path

        self.processor = cast(
            Any,
            LFRAG_Processor.from_pretrained(self.base_model_name_or_path),
        )

        base_model = cast(
            Any,
            LFRAG_Retriever.from_pretrained(
                self.base_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
            ),
        )
        if hasattr(base_model, "resize_token_embeddings"):
            base_model.resize_token_embeddings(len(self.processor.tokenizer))

        self.model = self._maybe_load_lora(base_model, self.lora_name_or_path).eval()
        self.bbox_only = True

    def _maybe_load_lora(self, model: Any, lora_name_or_path: Optional[str]) -> Any:
        if not lora_name_or_path:
            return model
        if not os.path.exists(lora_name_or_path):
            raise FileNotFoundError(f"LoRA path does not exist: {lora_name_or_path}")

        try:
            from peft import PeftModel
        except Exception as e:
            raise ImportError("Missing peft, cannot load LoRA weights.") from e

        try:
            return cast(Any, PeftModel.from_pretrained(model, lora_name_or_path))
        except Exception:
            adapter_path = os.path.join(lora_name_or_path, "adapter_model")
            if os.path.exists(adapter_path):
                return cast(Any, PeftModel.from_pretrained(model, adapter_path))
            raise

    def _collate_images_with_tags(self, batch: List[Dict[str, object]]):
        images = cast(List[Image.Image], [elt["image"] for elt in batch])
        tags = cast(List[str], [elt["tag"] for elt in batch])
        return self.processor.process_images(images=images, tags=tags).to(self.device)

    def _collate_queries(self, batch: List[str]):
        return self.processor.process_queries(queries=batch).to(self.device)

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        dataloader = DataLoader(
            dataset=ListDataset[str](queries),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_queries,
            num_workers=self.num_workers,
        )

        query_embeddings: List[torch.Tensor] = []
        with torch.no_grad():
            for batch_query in tqdm(dataloader, desc="Forward pass queries...", leave=False):
                embeddings_query = self.model(**batch_query).to("cpu")
                query_embeddings.extend(list(torch.unbind(embeddings_query)))
        return query_embeddings

    def forward_passages(
        self,
        passages: List[Image.Image],
        batch_size: int,
        tags: Optional[List[str]] = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        if tags is None:
            tags = [_default_tag()] * len(passages)
        if len(tags) != len(passages):
            raise ValueError(f"The number of tags ({len(tags)}) does not match the number of passages ({len(passages)})")

        items: List[Dict[str, object]] = [{"image": img, "tag": tag} for img, tag in zip(passages, tags)]
        dataloader = DataLoader(
            dataset=ListDataset[Dict[str, object]](items),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_images_with_tags,
            num_workers=self.num_workers,
        )

        passage_embeddings: List[torch.Tensor] = []
        with torch.no_grad():
            for batch_doc in tqdm(dataloader, desc="Forward pass documents...", leave=False):
                embeddings_doc = self.model(**batch_doc).to("cpu")
                passage_embeddings.extend(list(torch.unbind(embeddings_doc)))
        return passage_embeddings

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = 128,
    ) -> torch.Tensor:
        if batch_size is None:
            batch_size = 128
        return self.processor.score(
            query_embeddings,
            passage_embeddings,
            batch_size=batch_size,
            device="cpu",
        )
