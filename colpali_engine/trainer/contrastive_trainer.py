from functools import partial
from typing import Optional

import datasets
import torch
from datasets import DatasetDict
from torch.distributed.nn.functional import all_gather  # PyTorch ≥ 2.1
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from transformers import Trainer, is_datasets_available
from transformers.trainer_utils import seed_worker

# from colpali_engine.data.sampler import SingleDatasetBatchSampler


def concat_all_gather(t: torch.Tensor) -> torch.Tensor:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.cat(all_gather(t), dim=0)  # keeps grad graph
    return t


class ContrastiveTrainer(Trainer):
    def __init__(self, loss_func, is_vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        self.is_vision_model = is_vision_model  # Unused argument, will be removed in 0.4.0
        self.args.remove_unused_columns = False  # Safety, don't remove dataset columns from dataloader
        self.dataset_list = kwargs["train_dataset"]  # <class 'colpali_engine.utils.dataset_transformation.JsonlCropDataset'>

    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[torch.utils.data.Sampler]:
        return super()._get_train_sampler(train_dataset=train_dataset)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer needs a train_dataset")

        dataset = self.train_dataset
        sampler = self._get_train_sampler(dataset)

        dataloader_params = {
            "sampler": sampler,
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,  
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
        }
        if self.args.dataloader_prefetch_factor is not None:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index
            )

        dataloader = DataLoader(dataset, **dataloader_params)
        return self.accelerator.prepare(dataloader)
    

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None,):
        # 1. forward query
        query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])  # [B, Nq, D]

        # 2. forward all doc crops (pos + neg)
        doc_embeddings = model(
            **{k[4:]: v for k, v in inputs.items() if k.startswith("doc")}
        )  # [sum(Kᵢ+Lᵢ), Nd, D]

        pos_doc_crop_counts = inputs["pos_doc_crop_counts"]
        neg_doc_crop_counts = inputs["neg_doc_crop_counts"]

        # 3. sample ONE positive per query
        device = doc_embeddings.device
        sampled_indices = []
        # print("pos_doc_crop_counts:", pos_doc_crop_counts)
        # print("neg_doc_crop_counts:", neg_doc_crop_counts)

        cursor = 0
        for i, (k, l) in enumerate(zip(pos_doc_crop_counts, neg_doc_crop_counts)):
            if k <= 0:
                raise ValueError(f"Query {i} has no positive samples")

            pos_offset = torch.randint(0, k, (1,), device=device).item()
            sampled_indices.append(cursor + pos_offset)

            cursor += k + l

        sampled_indices = torch.tensor(sampled_indices, device=device)
        # print("sampled_indices:", sampled_indices)

        doc_outputs = doc_embeddings.index_select(0, sampled_indices)
        # print("doc_outputs.shape:", doc_outputs.shape)
        # doc_outputs: [B, Nd, D]


        # 4. multi-GPU gather
        offset = 0
        if self.accelerator.num_processes > 1 and self.accelerator.sync_gradients:
            if num_items_in_batch is None:
                num_items_in_batch = doc_outputs.shape[0]

            doc_outputs = self.accelerator.pad_across_processes(
                doc_outputs,
                dim=1,
                pad_index=0,
                pad_first=True,
            )
            doc_outputs = concat_all_gather(doc_outputs)

            rank = self.accelerator.process_index
            offset = rank * num_items_in_batch

        # 5. standard n×n contrastive loss
        # print("offset:", offset)
        loss = self.loss_func(
            query_outputs,
            doc_outputs,
            offset=offset,
        )
        return (loss, (query_outputs, doc_outputs)) if return_outputs else loss


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=True):
        """This function is used to generate predictions and return the loss for the given inputs."""
        if not prediction_loss_only:
            raise ValueError("prediction_step is only called with prediction_loss_only=True")

        with torch.no_grad():
            # feed only kwargs with 'doc_' prefix
            doc_outputs = model(**{k[4:]: v for k, v in inputs.items() if k.startswith("doc")})
            query_outputs = model(input_ids=inputs["query_input_ids"], attention_mask=inputs["query_attention_mask"])
            pos_doc_crop_counts = inputs["pos_doc_crop_counts"]
            neg_doc_crop_counts = inputs["neg_doc_crop_counts"]

            loss = self.loss_func(query_outputs, doc_outputs, pos_doc_crop_counts, neg_doc_crop_counts, offset=0)
            return loss, None, None
