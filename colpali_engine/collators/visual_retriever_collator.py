import random
from typing import Any, Dict, List, Union

from PIL.Image import Image

from colpali_engine.data.dataset import ColPaliEngineDataset
from colpali_engine.models.paligemma import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


def prefix_keys(data: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Add a prefix to all keys in the dictionary to avoid key conflicts between different types of samples"""
    return {f"{prefix}{k}": v for k, v in data.items()}


class VisualRetrieverCollator:
    """Visual retrieval collator adapted for multi-query, separate global image processing + local crops"""

    # Prefix definitions (distinguish queries, positive samples, negative samples, global images)
    query_prefix = "query_"
    pos_doc_prefix = "doc_"
    neg_doc_prefix = "neg_doc_"
    global_prefix = "global_" 

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = None

        # Adapt to special tokens and padding settings for ColPaliProcessor
        if isinstance(self.processor, (ColPaliProcessor,)):
            image_token = "<image>"
            try:
                idx = self.processor.tokenizer.additional_special_tokens.index(image_token)
                self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[idx]
            except ValueError:
                self.image_token_id = None

        if isinstance(self.processor, ColPaliProcessor) and self.processor.tokenizer.padding_side != "right":
            print("Setting padding side to right")
            self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:

        queries = []                  # All query texts
        images, labels = [], []        # Positive and negative sample crops + labels
        pos_crop_counts = []          # Number of positive sample crops for each query
        neg_crop_counts = []          # Number of negative sample crops for each query
        crop_counts = []      # Number of all sample crops for each query
        questions = []
        answers = []

        # First pass: extract queries, global images, and crops (with labels) for all examples
        for example in examples:
            questions.append(example["question"])
            answers.append(example["answer"])
            
            # Extract global image and assign an empty label (since global image has no specific class label)
            global_image = example["image"]  # PIL Image
            global_label = ""  # Global image label is empty
            images.append(global_image)  # Add global image to the sample set
            labels.append(global_label)

            # Extract query text
            query = example["question"]
            queries.append(query)

            # Extract relevant bbox_ids for the current QA pair (positive sample identifiers)
            relevant_bbox_ids = example["relevant_bbox_ids"]
            all_crops = example.get("crops", [])  # All crops for the current sample

            # Split positive/negative sample crops based on relevant_bbox_ids
            # Positive samples: crops corresponding to relevant_bbox_ids
            pos_crops = [crop for crop in all_crops if crop["bbox_id"] in relevant_bbox_ids]
            # Negative samples: crops not corresponding to relevant_bbox_ids
            neg_crops = [crop for crop in all_crops if crop["bbox_id"] not in relevant_bbox_ids]

            # Collect positive sample crops + labels
            pos_current_images = [crop["crop_image"] for crop in pos_crops]
            pos_current_labels = [f'<{crop["class_name"]}>' for crop in pos_crops]
            images.extend(pos_current_images)
            labels.extend(pos_current_labels)
            pos_crop_counts.append(len(pos_current_images))  

            # Collect negative sample crops + labels
            neg_current_images = [crop["crop_image"] for crop in neg_crops]
            neg_current_labels = [f'<{crop["class_name"]}>' for crop in neg_crops]
            images.extend(neg_current_images)
            labels.extend(neg_current_labels)
            neg_crop_counts.append(len(neg_current_images))  

            assert len(pos_current_images) + len(neg_current_images) == example["crop_counts"], "Inconsistent crop counts"

            crop_counts.append(example["crop_counts"])  

        assert all(isinstance(q, str) for q in queries), "All queries must be strings, this collator does not support images in queries."

        # Process query
        queries = [self.processor.query_prefix + q + self.processor.query_augmentation_token * 10 for q in queries]
        batch_query = self.auto_collate_texts(queries, key_prefix=self.query_prefix)

        # Process images + labels (global image + crops)
        batch_pos = self.auto_collate_images(
            images=images,
            labels=labels,
            crop_counts=crop_counts,
            key_prefix=self.pos_doc_prefix
        )

        return {
            **batch_query,
            **batch_pos,
            'pos_doc_crop_counts': pos_crop_counts,
            'neg_doc_crop_counts': neg_crop_counts,
            'questions': questions,
            'answers': answers,
        }
    
    ## process text queries
    def auto_collate_texts(self, texts: List[str], key_prefix: str) -> Dict[str, Any]:
        
        proc_batch = self.processor.process_queries(texts)
        return prefix_keys(proc_batch, key_prefix)

    ## process images + labels (global image + crops)
    def auto_collate_images(self, images, labels: List[str], 
                           crop_counts: List[int], key_prefix: str) -> Dict[str, Any]:

        assert len(images) == len(labels), f"Number of images ({len(images)}) does not match number of labels ({len(labels)})"

        proc_batch = self.processor.process_images(images, labels)

        proc_batch["crop_counts"] = crop_counts

        return prefix_keys(proc_batch, key_prefix)
    


class VisualRetrieverCollator_only_crop:
    query_prefix = "query_"
    pos_doc_prefix = "doc_"

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = None

        if isinstance(self.processor, (ColPaliProcessor,)):
            image_token = "<image>"
            try:
                idx = self.processor.tokenizer.additional_special_tokens.index(image_token)
                self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[idx]
            except ValueError:
                self.image_token_id = None

        if isinstance(self.processor, ColPaliProcessor) and self.processor.tokenizer.padding_side != "right":
            print("Setting padding side to right")
            self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        queries = []                 
        images = []        # crops
        crop_counts = []      

        for example in examples:
            queries.append(example["question"])

            all_crops = example.get("crops", [])  
            crop_images = [crop["crop_image"] for crop in all_crops]
            images.extend(crop_images)

            crop_counts.append(len(all_crops))  

        assert all(isinstance(q, str) for q in queries), "All queries must be strings, this collator does not support images in queries."

        queries = [self.processor.query_prefix + q + self.processor.query_augmentation_token * 10 for q in queries]
        batch_query = self.auto_collate_texts(queries, key_prefix=self.query_prefix)
        batch_crops = self.auto_collate_images(images=images,key_prefix=self.pos_doc_prefix)

        return {
            **batch_query,
            **batch_crops,
            'crop_counts': crop_counts,
        }
    
    def auto_collate_texts(self, texts: List[str], key_prefix: str) -> Dict[str, Any]:   
        proc_batch = self.processor.process_queries(texts)
        return prefix_keys(proc_batch, key_prefix)

    def auto_collate_images(self, images, key_prefix: str) -> Dict[str, Any]:
        proc_batch = self.processor.process_images(images)
        return prefix_keys(proc_batch, key_prefix)


class VisualRetrieverCollator_for_colqwen:
    query_prefix = "query_"
    pos_doc_prefix = "doc_"

    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_length: int = 2048,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_token_id = None

        if isinstance(self.processor, (ColPaliProcessor,)):
            image_token = "<image>"
            try:
                idx = self.processor.tokenizer.additional_special_tokens.index(image_token)
                self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[idx]
            except ValueError:
                self.image_token_id = None

        if isinstance(self.processor, ColPaliProcessor) and self.processor.tokenizer.padding_side != "right":
            print("Setting padding side to right")
            self.processor.tokenizer.padding_side = "right"

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:

        queries = []             
        images = []               
        pos_crop_counts = []          
        neg_crop_counts = []          

        for example in examples:
            query = example["question"]
            queries.append(query)

            relevant_bbox_ids = example["relevant_bbox_ids"]
            all_crops = example.get("crops", [])  

            pos_crops = [crop for crop in all_crops if crop["bbox_id"] in relevant_bbox_ids]
            neg_crops = [crop for crop in all_crops if crop["bbox_id"] not in relevant_bbox_ids]

            pos_current_images = [crop["crop_image"] for crop in pos_crops]
            images.extend(pos_current_images)
            pos_crop_counts.append(len(pos_current_images)) 

            neg_current_images = [crop["crop_image"] for crop in neg_crops]
            images.extend(neg_current_images)
            neg_crop_counts.append(len(neg_current_images)) 

        assert all(isinstance(q, str) for q in queries), "All queries must be strings, this collator does not support images in queries."

        queries = [self.processor.query_prefix + q + self.processor.query_augmentation_token * 10 for q in queries]
        batch_query = self.auto_collate_texts(queries, key_prefix=self.query_prefix)
        batch_crops = self.auto_collate_images(images=images,key_prefix=self.pos_doc_prefix)

        return {
            **batch_query,
            **batch_crops,
            'pos_doc_crop_counts': pos_crop_counts,
            'neg_doc_crop_counts': neg_crop_counts,
        }
    
    def auto_collate_texts(self, texts: List[str], key_prefix: str) -> Dict[str, Any]:   
        proc_batch = self.processor.process_queries(texts)
        return prefix_keys(proc_batch, key_prefix)

    def auto_collate_images(self, images, key_prefix: str) -> Dict[str, Any]:
        proc_batch = self.processor.process_images(images)
        return prefix_keys(proc_batch, key_prefix)