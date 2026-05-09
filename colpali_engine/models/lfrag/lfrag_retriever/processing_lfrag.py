from typing import ClassVar, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature
from transformers.models.qwen2_vl import Qwen2VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor


class LFRAG_Processor(BaseVisualRetrieverProcessor, Qwen2VLProcessor):  # noqa: N801
    """
    Processor for LFRAG

    Args:
        *args: Variable length argument list to be passed to the parent `Qwen2VLProcessor` class.
        max_num_visual_tokens: The maximum number of visual tokens that can be processed by the model.
        **kwargs: Arbitrary keyword arguments to be passed to the parent `Qwen2VLProcessor` class.
    """

    # visual_prompt_prefix: ClassVar[str] = (
    #     "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    # )
    def visual_prompt_prefix(self, tag): 
        return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.{tag}<|im_end|><|endoftext|>"
    
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        new_special_tokens = [
            "<title>", '<plain text>', '<figure_caption>', '<table_caption>', '<table_footnote>','<formula_caption>','<isolate_formula>','<abandon>', '<figure>', '<table>'
        ]
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": new_special_tokens
        })

        self.tokenizer.padding_side = "left"

    @classmethod
    def from_pretrained(
        cls,
        *args,
        device_map: Optional[str] = None,
        **kwargs,
    ):
        instance = super().from_pretrained(
            *args,
            device_map=device_map,
            **kwargs,
        )

        if "max_num_visual_tokens" in kwargs:
            instance.image_processor.max_pixels = kwargs["max_num_visual_tokens"] * 28 * 28
            instance.image_processor.size["longest_edge"] = instance.image_processor.max_pixels

        return instance

    def process_images(
        self,
        images: List[Image.Image],
        tags: List[str],
    ) -> Union[BatchFeature, BatchEncoding]:
        """
        Process images for LFRAG

        Args:
            images: List of PIL images.
        """

        images = [image.convert("RGB") for image in images]
        # Ensure all images meet the minimum size requirement (28x28)
        min_size = 28
        processed_images = []
        for img in images:
            width, height = img.size
            if width < min_size or height < min_size:
                # Calculate new size maintaining aspect ratio
                if width < min_size and height < min_size:
                    # Both dimensions too small, resize to min_size x min_size
                    new_width = min_size
                    new_height = min_size
                elif width < min_size:
                    # Only width too small
                    scale = min_size / width
                    new_width = min_size
                    new_height = max(min_size, int(height * scale))
                else:
                    # Only height too small
                    scale = min_size / height
                    new_height = min_size
                    new_width = max(min_size, int(width * scale))

                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            processed_images.append(img)

        images = processed_images

        batch_doc = self(
            text=[self.visual_prompt_prefix(tag) for tag in tags],
            images=images,
            padding="longest",
            return_tensors="pt",
        )
        
        # NOTE: The following adjustment ensures correct behavior with DDP on multiple GPUs.
        offsets = batch_doc["image_grid_thw"][:, 1] * batch_doc["image_grid_thw"][:, 2]  # (batch_size,)

        # Split the pixel_values tensor into a list of tensors, one per image
        pixel_values = list(
            torch.split(batch_doc["pixel_values"], offsets.tolist())
        )  # [(num_patches_image_0, pixel_values), ..., (num_patches_image_n, pixel_values)]

        # Pad the list of pixel_value tensors to the same length along the sequence dimension
        batch_doc["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )  # (batch_size, max_num_patches, pixel_values)

        # print("processor_batch_doc:")
        # print({k: v.shape for k, v in batch_doc.items()})
        return batch_doc

    def process_texts(self, texts: List[str]) -> Union[BatchFeature, BatchEncoding]:
        """
        Process texts for LFRAG

        Args:
            texts: List of input texts.

        Returns:
            Union[BatchFeature, BatchEncoding]: Processed texts.
        """
        batch_query = self(
            text=texts,
            return_tensors="pt",
            padding="longest",
        )
        return batch_query

    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        The `spatial_merge_size` is the number of patches that will be merged spatially. It is stored in
        as a `Qwen2VLForConditionalGeneration` attribute under `model.spatial_merge_size`.
        """
        patch_size = self.image_processor.patch_size

        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=patch_size * self.image_processor.merge_size,
            min_pixels=self.image_processor.size["shortest_edge"],
            max_pixels=self.image_processor.size["longest_edge"],
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
