import argparse
import shutil
from pathlib import Path
import os
import wandb
import torch
from peft import LoraConfig
from transformers import TrainingArguments
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.loss.late_interaction_losses import ColbertLoss
from colpali_engine.models import LFRAG_Retriever, LFRAG_Processor 
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig 
from colpali_engine.utils.dataset_transformation import load_multi_jsonl_datasets, load_docmatix_eval_datasets


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="./ckpts", help="where to write model + script copy")

    p.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    p.add_argument("--tau", type=float, default=0.02, help="temperature for loss function")
    p.add_argument("--peft", action="store_true", help="use PEFT for training")

    p.add_argument("--dataset_jsonl_path", type=str, default="./datasets/train/docmatix_train.jsonl", help="path to the jsonl file containing the dataset") 
    p.add_argument("--dataset_image_path", type=str, default="./datasets/train", help="path to the directory containing the images") 
    p.add_argument("--eval_size", type=int, default=1000, help="number of testing samples") 

    p.add_argument("--model_path", type=str, default="models/colqwen2.5-v0.2", help="path to the pretrained model")
    p.add_argument("--per_device_train_batch_size", type=int, default=8, help="batch size per device during training")
    p.add_argument("--epoch", type=int, default=2)

    p.add_argument("--wandb_name", type=str, help="wandb run name")
    p.add_argument("--wandb_project", type=str, help="wandb project name")
    p.add_argument("--wandb_entity", type=str, help="wandb entity name")
    p.add_argument("--wandb_api_key", type=str, help="wandb api key")
    
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_API_KEY"] = args.wandb_api_key

    wandb.init(entity=args.wandb_entity,
               project=args.wandb_project, 
               name=args.wandb_name
               ) 

    loss_func = ColbertLoss(temperature=args.tau,
                           normalize_scores=True,
                           use_smooth_max=False,
                           pos_aware_negative_filtering=False,
                           )
    
    config = ColModelTrainingConfig( 
        output_dir=args.output_dir,
        processor=LFRAG_Processor.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            max_num_visual_tokens=768,
        ),
        model=LFRAG_Retriever.from_pretrained(
            pretrained_model_name_or_path=args.model_path,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
        ),
        train_dataset = load_multi_jsonl_datasets(
            dataset_root=args.dataset_image_path,
            split="train",
            seed=42,
        ),
        eval_dataset = load_docmatix_eval_datasets(
            args.dataset_jsonl_path, 
            args.dataset_image_path, 
            args.eval_size
        ), 
        loss_func=loss_func,
        tr_args=TrainingArguments(
            output_dir=None,
            overwrite_output_dir=True,
            num_train_epochs=args.epoch,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            per_device_eval_batch_size=16,
            eval_strategy="steps",
            dataloader_num_workers=8,
            save_steps=50,
            logging_steps=1,
            eval_steps=1000000,
            warmup_steps=100,
            learning_rate=args.lr,
            save_total_limit=10,
            report_to="wandb", 
        ),
        peft_config=LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.1,
            init_lora_weights="gaussian",
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules="(.*(model)(?!.*visual).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
            modules_to_save=["cross_attention"],
        )
    )

    trainer = ColModelTraining(config)
    print(trainer)
    trainer.train()
    trainer.save()

    # accelerate launch --config_file scripts/train_config.yaml scripts/train_lfrag_sample.py
