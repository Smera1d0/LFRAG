import math
from typing import List, Optional, Union, cast

import numpy as np
import torch
from tqdm import tqdm

from retrievers.base_vision_retriever import BaseVisionRetriever
from retrievers.registry_utils import register_vision_retriever
from utils.iter_utils import batched
from utils.torch_utils import get_torch_device


def _patch_transformers_mistral_position_embeddings() -> None:
    try:
        import inspect

        from transformers.models.mistral.modeling_mistral import MistralAttention, MistralRotaryEmbedding
    except Exception:
        return

    if getattr(MistralAttention, "_mrag_eval_position_embeddings_patch", False):
        return

    try:
        sig = inspect.signature(MistralAttention.forward)
    except Exception:
        return

    pe_param = sig.parameters.get("position_embeddings")
    if pe_param is None:
        return

    if pe_param.default is not inspect._empty:
        return

    original_forward = MistralAttention.forward

    def _forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: Optional[torch.Tensor],
        past_key_value=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        if position_embeddings is None:
            position_ids = kwargs.get("position_ids")
            if position_ids is None:
                batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            position_ids = position_ids.to(device=hidden_states.device, dtype=torch.long)

            rotary_emb = getattr(self, "_mrag_eval_rotary_emb", None)
            if rotary_emb is None:
                rotary_emb = MistralRotaryEmbedding(config=self.config, device=hidden_states.device)
                setattr(self, "_mrag_eval_rotary_emb", rotary_emb)
            position_embeddings = rotary_emb(hidden_states, position_ids)

        return original_forward(
            self,
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    MistralAttention.forward = _forward
    setattr(MistralAttention, "_mrag_eval_position_embeddings_patch", True)


@register_vision_retriever("nv-embed-v2")
class NVEmbedV2Retriever(BaseVisionRetriever):
    """
    NVEmbedV2Retriever class to retrieve embeddings using the nvidia/NV-Embed-v2 model.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str = "nvidia/NV-Embed-v2",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(use_visual_embedding=False)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                'Install the missing dependencies with `pip install sentence-transformers` to use NVEmbedV2Retriever.'
            )

        self.device = get_torch_device(device)
        _patch_transformers_mistral_position_embeddings()

        # Load the model using SentenceTransformer as per the example
        self.model = SentenceTransformer(
            pretrained_model_name_or_path, 
            trust_remote_code=True, 
            device=self.device
        )

        try:
            first_module = self.model._first_module()
            auto_model = getattr(first_module, "auto_model", None)
            if auto_model is not None and getattr(auto_model, "config", None) is not None:
                auto_model.config.use_cache = False
                embedding_model = getattr(auto_model, "embedding_model", None)
                if embedding_model is not None and getattr(embedding_model, "config", None) is not None:
                    embedding_model.config.use_cache = False
        except Exception:
            pass
        
        # Specific configurations for NV-Embed-v2
        self.model.max_seq_length = 32768
        self.model.tokenizer.padding_side = "right"
        
        # Define the instruction task. 
        # For general retrieval benchmarks, we use a standard generic instruction.
        self.task_instruction = "Given a question, retrieve passages that answer the question"
        self.query_prefix = f"Instruct: {self.task_instruction}\nQuery: "

    def _add_eos(self, texts: List[str]) -> List[str]:
        """Helper to append EOS token to inputs as required by NV-Embed-v2"""
        return [t + self.model.tokenizer.eos_token for t in texts]

    def forward_queries(self, queries: List[str], batch_size: int, **kwargs) -> torch.Tensor:
        list_emb_queries: List[float] = []

        for query_batch in tqdm(
            batched(queries, batch_size),
            desc="Forwarding query batches",
            total=math.ceil(len(queries) / batch_size),
            leave=False,
        ):
            query_batch = cast(List[str], query_batch)
            
            # NV-Embed-v2 requires EOS token appended
            processed_queries = self._add_eos(query_batch)

            with torch.no_grad():
                # Encode with the specific prompt prefix and normalization
                query_embeddings = self.model.encode(
                    processed_queries,
                    prompt=self.query_prefix,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

            list_emb_queries.extend(query_embeddings.tolist())

        return torch.tensor(list_emb_queries)

    def forward_passages(self, passages: List[str], batch_size: int, **kwargs) -> torch.Tensor:
        list_emb_passages: List[torch.Tensor] = []

        for passage_batch in tqdm(
            batched(passages, batch_size),
            desc="Forwarding passage batches",
            total=math.ceil(len(passages) / batch_size),
            leave=False,
        ):
            passage_batch = cast(List[str], passage_batch)

            # NV-Embed-v2 requires EOS token appended for passages too
            processed_passages = self._add_eos(passage_batch)

            with torch.no_grad():
                # Passages do not need the prompt prefix, but still need normalization
                passage_embeddings = self.model.encode(
                    processed_passages,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )

            list_emb_passages.extend(passage_embeddings.tolist())

        return torch.tensor(list_emb_passages)

    def get_scores(
        self,
        query_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        passage_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Dot-product similarity between queries and passages.
        Since embeddings are normalized, this is equivalent to Cosine Similarity.
        """
        if isinstance(query_embeddings, list):
            query_embeddings = torch.stack(query_embeddings)
        if isinstance(passage_embeddings, list):
            passage_embeddings = torch.stack(passage_embeddings)

        # Calculate Dot Product
        # Note: The official example multiplies by 100, but standard benchmarks usually 
        # expect raw scores (0-1 range for cosine). We keep it standard here.
        scores = torch.einsum("bd,cd->bc", query_embeddings, passage_embeddings)
        
        return scores
