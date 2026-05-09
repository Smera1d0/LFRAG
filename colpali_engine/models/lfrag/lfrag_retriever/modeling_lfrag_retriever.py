from typing import ClassVar, Optional, Dict, List, Union, Tuple
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLModelOutputWithPast,
    Unpack,
    KwargsForCausalLM,
    
    is_torchdynamo_compiling
)

class LFRAG_Retriever(Qwen2_5_VLModel):  # noqa: N801
    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: Qwen2_5_VLConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,  
            num_heads=self.config.num_attention_heads,  
            batch_first=True  # [batch_size, seq_len, embed_dim]
        )
        
        # self._initialize_cross_attention_weights()
        self.post_init()

    def _initialize_cross_attention_weights(self):
        nn.init.xavier_uniform_(self.cross_attention.in_proj_weight, gain=1.0)
        if self.cross_attention.in_proj_bias is not None:
            nn.init.constant_(self.cross_attention.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.cross_attention.out_proj.weight, gain=1.0)
        if self.cross_attention.out_proj.bias is not None:
            nn.init.constant_(self.cross_attention.out_proj.bias, 0.0)
    
    def _initialize_weights(self, module):
        super()._initialize_weights(module)
        if module is self:
            self._initialize_cross_attention_weights()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        key_mapping = kwargs.pop("key_mapping", None)
        if key_mapping is None:
            key_mapping = super()._checkpoint_conversion_mapping

        model = super().from_pretrained(*args, **kwargs, key_mapping=key_mapping)
        return model
    
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[list[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_values: Optional[torch.Tensor] = None,
            pixel_values_videos: Optional[torch.FloatTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            rope_deltas: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            second_per_grid_ts: Optional[torch.Tensor] = None,
            **kwargs: Unpack[KwargsForCausalLM],
        ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        
        if pixel_values is not None and image_grid_thw is not None:
            if pixel_values.dim() > 2: 
                patch_counts = image_grid_thw[:, 1] * image_grid_thw[:, 2]
                pixel_values = torch.cat(
                    [pix[:count] for pix, count in zip(pixel_values, patch_counts)],
                    dim=0
                )
        
        # 1. process crop_counts 
        crop_counts = kwargs.pop("crop_counts", None)
        if crop_counts is None and "doc_crop_counts" in kwargs:
            crop_counts = kwargs.pop("doc_crop_counts")

        # 2. process inputs_embeds 
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_embeds = self.get_image_features(pixel_values, image_grid_thw)
                if crop_counts is not None and len(crop_counts) > 0:
                    enhanced_image_embeds = []
                    start_idx = 0
                    
                    for num_crop in crop_counts:
                        global_feat = image_embeds[start_idx] # [seq_len_global, dim]
                        crop_feats_list = image_embeds[start_idx + 1 : start_idx + 1 + num_crop]
                        
                        original_lengths = [feat.shape[0] for feat in crop_feats_list]
                        padded_crops = pad_sequence(crop_feats_list, batch_first=True, padding_value=0.0)

                        # Expand global embedding to match the batch dimension of crop embeddings: [Num_Crops, Seq_Len_Global, Dim]
                        k = global_feat.unsqueeze(0).expand(num_crop, -1, -1)
                        v = k

                        # Cross Attention (Mid-Fusion)
                        # Q=Crops(padded), K=Global, V=Global
                        # attn_out: [Num_Crops, Max_Len, Dim]
                        attn_out, _ = self.cross_attention(padded_crops, k, v)

                        enhanced_image_embeds.append(global_feat)
                        for i, length in enumerate(original_lengths):
                                valid_attn = attn_out[i, :length, :]
                                original_crop = crop_feats_list[i]
                                enhanced_crop = original_crop + valid_attn
                                enhanced_image_embeds.append(enhanced_crop)

                        start_idx += 1 + num_crop
                    
                    image_embeds = enhanced_image_embeds

                # 3. Fill in the enhanced image features into inputs_embedds
                image_embeds = torch.cat(image_embeds, dim=0)
                n_image_tokens = (input_ids == self.config.image_token_id).sum()
                n_image_features = image_embeds.shape[0]
                
                if not is_torchdynamo_compiling() and n_image_tokens != n_image_features:
                     pass 

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
                video_embeds = torch.cat(video_embeds, dim=0)
                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)      

        # 4. Forward (LLM)
        outputs = super().forward(
            input_ids=input_ids, 
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True, 
            return_dict=True,
            pixel_values=None, # Prohibit the parent class from processing the image again
            image_grid_thw=image_grid_thw, 
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs
        )
        
        last_hidden_states = outputs.last_hidden_state
        
        # 5. Post processing: Split and filter out Global, keep only Crop
        if image_grid_thw is not None and input_ids is not None and crop_counts is not None:
            image_token_mask = input_ids == self.config.image_token_id
            all_image_tokens = last_hidden_states[image_token_mask]
            
            merge_size = self.visual.spatial_merge_size
            split_lengths = (image_grid_thw[:, 0] * (image_grid_thw[:, 1] // merge_size) * (image_grid_thw[:, 2] // merge_size)).tolist()
            all_visual_units = torch.split(all_image_tokens, split_lengths, dim=0)
            
            global_units = []
            all_crop_units = []
            cursor = 0
            
            for num_crop in crop_counts:
                global_units.append(all_visual_units[cursor])
                if num_crop > 0:
                    sample_crops = all_visual_units[cursor + 1 : cursor + 1 + num_crop]
                    all_crop_units.extend(sample_crops)
                cursor += (1 + num_crop)
            
            if len(all_crop_units) > 0:
                # [Total_Crops, Seq_Len, Dim]
                crop_emb = pad_sequence(all_crop_units, batch_first=True, padding_value=0.0)
                crop_emb = self.custom_text_proj(crop_emb)
                crop_emb = crop_emb / (crop_emb.norm(dim=-1, keepdim=True) + 1e-6)
                proj = crop_emb
            else:
                proj = torch.zeros(0, self.dim, device=last_hidden_states.device)

            return proj

        else:
            # Fallback (Query or Text-only)
            proj = last_hidden_states
            proj = self.custom_text_proj(proj)  # [B, Seq_Len, Dim=128]
            proj = proj / (proj.norm(dim=-1, keepdim=True) + 1e-6)     # L2 Norm
            
            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    proj = proj * attention_mask.unsqueeze(-1)
                elif attention_mask.ndim == 4:
                    pass
            return proj
    
    @property
    def patch_size(self) -> int:
        return self.visual.config.patch_size

    @property
    def spatial_merge_size(self) -> int:
        return self.visual.config.spatial_merge_size
      