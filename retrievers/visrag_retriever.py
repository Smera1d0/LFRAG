from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import List, Optional, Union, cast

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from retrievers.base_vision_retriever import BaseVisionRetriever
from retrievers.registry_utils import register_vision_retriever
from utils.iter_utils import batched
from utils.torch_utils import get_torch_device


@register_vision_retriever("visrag-ret")
class VisRAGRetriever(BaseVisionRetriever):
    def __init__(
        self,
        pretrained_model_name_or_path: str = "./models/VisRAG-Ret",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(use_visual_embedding=True)
        self.device = get_torch_device(device)

        model_path = Path(pretrained_model_name_or_path)
        if model_path.is_dir():
            adapter_config = model_path / "adapter_config.json"
            base_config = model_path / "config.json"
            if adapter_config.exists() and not base_config.exists():
                raise ValueError(
                    "VisRAGRetriever requires a full VisRAG-Ret checkpoint directory (with config.json). "
                    f"Got a PEFT adapter directory instead: {pretrained_model_name_or_path}. "
                    "Pass the base VisRAG-Ret model path or merge the adapter into the base model before evaluating."
                )

        # Load Model
        # VisRAG recommends bfloat16
        self.model = (
            AutoModel.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16, 
            )
            .to(self.device)
            .eval()
        )

        try:
            forward_params = set(inspect.signature(self.model.forward).parameters.keys())
        except (TypeError, ValueError):
            forward_params = set()
        if not {"text", "image", "tokenizer"}.issubset(forward_params):
            raise ValueError(
                "VisRAGRetriever expects a VisRAG-Ret style model whose forward(...) accepts "
                "`text`, `image`, and `tokenizer`. "
                f"Loaded model type: {type(self.model)} from {pretrained_model_name_or_path}."
            )

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, 
            trust_remote_code=True
        )

        # Instruction required by VisRAG for queries
        self.instruction = "Represent this query for retrieving relevant documents: "

    @staticmethod
    def _weighted_mean_pooling(hidden: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Specific pooling logic for VisRAG.
        """
        # attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        # Note: The reference implementation effectively weights tokens by their position 
        # (or cumulative count) within the mask.
        attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
        s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
        d = attention_mask_.sum(dim=1, keepdim=True).float()
        reps = s / d
        return reps

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_queries: List[torch.Tensor] = []
        
        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)
            
            # Prepend instruction
            query_texts = [self.instruction + query for query in query_batch]
            
            # Prepare inputs in the format VisRAG's custom model expects
            inputs = {
                "text": query_texts,
                "image": [None] * len(query_texts),
                "tokenizer": self.tokenizer
            }

            with torch.no_grad():
                # The model handles tokenization internally if passed raw text + tokenizer
                outputs = self.model(**inputs)
                
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state

                reps = self._weighted_mean_pooling(hidden, attention_mask)
                qs = F.normalize(reps, p=2, dim=1)

            # Ensure output is on the correct device and detached
            query_embeddings = qs.to(self.device)
            list_emb_queries.extend(list(torch.unbind(query_embeddings, dim=0)))

        return list_emb_queries

    def forward_passages(self, passages: List[Image.Image], batch_size: int, **kwargs) -> List[torch.Tensor]:
        list_emb_passages: List[torch.Tensor] = []
        
        for passage_batch in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            passage_batch = cast(List[Image.Image], passage_batch)
            
            # Ensure images are RGB
            images = [img.convert("RGB") for img in passage_batch]

            # Prepare inputs: Text must be empty strings for image encoding
            inputs = {
                "text": [""] * len(images),
                "image": images,
                "tokenizer": self.tokenizer
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
                
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state

                reps = self._weighted_mean_pooling(hidden, attention_mask)
                ps = F.normalize(reps, p=2, dim=1)

            passage_embeddings = ps.to(self.device)
            list_emb_passages.extend(list(torch.unbind(passage_embeddings, dim=0)))

        return list_emb_passages

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Dot-product similarity between queries and passages.
        """
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings)

        # VisRAG embeddings are normalized, so dot product == cosine similarity
        scores = torch.einsum("bd,cd->bc", query_embeddings, passage_embeddings)
        return scores
